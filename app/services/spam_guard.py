"""Pre-send spam-risk gate.

``deliverability_optimizer.scan_content_heuristic`` already knows how to score a
draft for spam-filter risk, but nothing on the send path ever asked it. The scan
existed only as an endpoint a human could click — so AI-composed copy, spintax
that resolved into a bad combination, or a template someone edited at 6 PM all
went out unexamined, and the first signal was the bounce rate two days later.

This module is the gate. It scores the *final* rendered subject and body,
immediately before dispatch, and returns one of three verdicts:

* ``pass``  — under the flag threshold. Nothing recorded, nothing changed.
* ``flag``  — over the flag threshold, under the block threshold. The send goes
  ahead and the score rides along on the send record, so a rising trend is
  visible before it becomes a deliverability incident.
* ``block`` — at or over the block threshold. The send is refused as a *soft*
  skip: the enrollment is held, not failed, because the fix is to edit the copy,
  not to burn the contact.

**Blocking is off by default** (``spam_guard_block_threshold`` high enough to be
unreachable in normal operation is *not* the mechanism — the explicit
``spam_guard_block_enabled`` flag is). A content heuristic that silently stops
outreach is worse than the thing it prevents; an operator turns blocking on once
they trust the scores they've been watching in the flag lane.

The scoring itself is deliberately the *deterministic* heuristic, never the AI
one: this runs on every send, must not cost an LLM call, and must return the
same verdict for the same copy every time. Nothing here raises — a scanner bug
must never stop the mail.

**Jev second opinion** (:func:`scan_async`). The send paths call the async
variant, which runs the same heuristic and then asks TypeSafe's Jev for a
yes/no spam probability. Jev can only make a verdict *stricter*: a probability
at ``jev_spam_flag_threshold`` turns a pass into a flag, and one at
``jev_spam_block_threshold`` blocks — but only when ``spam_guard_block_enabled``
is on, the same switch that governs heuristic blocking. With no
``TYPESAFE_API_KEY``, or Jev failing, :func:`scan_async` returns exactly what
:func:`scan` would. Jev isn't asked about copy the heuristic already blocked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.services import jev_service

logger = logging.getLogger(__name__)

# Verdicts.
PASS = "pass"
FLAG = "flag"
BLOCK = "block"


@dataclass
class SpamVerdict:
    """The gate's decision for one message."""

    verdict: str = PASS
    score: int = 0
    risk: str = "low"
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    # False only for ``block``. Kept separate from ``verdict`` so call sites read
    # as ``if not verdict.allowed`` rather than comparing strings.
    allowed: bool = True
    # Set when scanning itself failed; the send is always allowed in that case.
    error: Optional[str] = None
    # Jev's spam probability (0-1), when it was asked. None = not asked / no answer.
    jev_spam_probability: Optional[float] = None

    @property
    def flagged(self) -> bool:
        return self.verdict in (FLAG, BLOCK)

    def as_metadata(self) -> dict:
        """Compact form for ``OutreachEvent.event_metadata`` / ``EmailSend``.

        Only recorded when the message was flagged or blocked — writing a
        ``score: 0`` onto every clean send would triple the metadata volume for
        no signal.
        """
        meta = {
            "verdict": self.verdict,
            "score": self.score,
            "risk": self.risk,
            "issues": self.issues[:6],
        }
        if self.jev_spam_probability is not None:
            meta["jev_spam_probability"] = round(self.jev_spam_probability, 4)
        return meta

    def reason(self) -> str:
        detail = "; ".join(self.issues[:3]) or "no specific trigger"
        return f"spam risk {self.score}/100 ({self.risk}) — {detail}"


def _thresholds() -> tuple[int, int, bool, bool]:
    s = get_settings()
    return (
        int(s.spam_guard_flag_threshold),
        int(s.spam_guard_block_threshold),
        bool(s.spam_guard_enabled),
        bool(s.spam_guard_block_enabled),
    )


def scan(
    subject: str,
    body: str,
    *,
    has_unsubscribe: bool = True,
) -> SpamVerdict:
    """Score ``subject``/``body`` and decide whether the send may proceed.

    Synchronous and dependency-free by design: this sits directly in front of the
    SMTP call, so it must not do IO and must not be able to hang a send. Returns
    an allowing verdict on any internal failure — a broken scanner degrades to
    the behaviour we had before it existed (send it), never to a mail stoppage.
    """
    flag_at, block_at, enabled, block_enabled = _thresholds()
    if not enabled:
        return SpamVerdict()

    try:
        from app.services.deliverability_optimizer import get_deliverability_optimizer

        raw = get_deliverability_optimizer().scan_content_heuristic(
            subject or "", body or "", has_unsubscribe
        )
    except Exception as exc:  # noqa: BLE001 — never block mail on a scanner bug
        logger.warning("Spam scan failed, allowing send: %s", exc)
        return SpamVerdict(error=str(exc))

    score = int(raw.get("spam_score") or 0)
    verdict = SpamVerdict(
        score=score,
        risk=str(raw.get("risk") or "low"),
        issues=list(raw.get("issues") or []),
        suggestions=list(raw.get("suggestions") or []),
    )

    if score >= block_at and block_enabled:
        verdict.verdict = BLOCK
        verdict.allowed = False
    elif score >= flag_at:
        verdict.verdict = FLAG
    else:
        verdict.verdict = PASS

    if verdict.flagged:
        logger.warning(
            "Pre-send spam scan: %s (%s) — subject=%r",
            verdict.verdict, verdict.reason(), (subject or "")[:120],
        )
    return verdict


def scan_for_send(
    subject: str,
    body: str,
    *,
    has_unsubscribe: bool = True,
) -> tuple[bool, SpamVerdict]:
    """``scan`` in the shape a send path wants: ``(may_send, verdict)``."""
    verdict = scan(subject, body, has_unsubscribe=has_unsubscribe)
    return verdict.allowed, verdict


def _plain_text(body: str) -> str:
    """Visible text of an HTML body — what a reader (and Jev) judges."""
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(body or "", "html.parser").get_text(" ", strip=True)
    except Exception:  # noqa: BLE001
        return body or ""


async def scan_async(
    subject: str,
    body: str,
    *,
    has_unsubscribe: bool = True,
) -> SpamVerdict:
    """:func:`scan`, then Jev's spam probability as a second opinion.

    Jev can only tighten the verdict (pass → flag, → block when blocking is
    enabled), never loosen it. Never raises; any Jev problem leaves the
    heuristic verdict as it was.
    """
    verdict = scan(subject, body, has_unsubscribe=has_unsubscribe)
    s = get_settings()
    if not s.spam_guard_enabled or verdict.verdict == BLOCK:
        return verdict

    try:
        p = await jev_service.spam_probability(subject, _plain_text(body))
    except Exception as exc:  # noqa: BLE001 — never block mail on a scanner bug
        logger.warning("Jev spam check failed, keeping heuristic verdict: %s", exc)
        return verdict
    if p is None:
        return verdict

    verdict.jev_spam_probability = p
    before = verdict.verdict
    if p >= float(s.jev_spam_block_threshold) and s.spam_guard_block_enabled:
        verdict.verdict = BLOCK
        verdict.allowed = False
    elif p >= float(s.jev_spam_flag_threshold) and verdict.verdict == PASS:
        verdict.verdict = FLAG

    if verdict.verdict != before:
        verdict.issues.insert(0, f"Jev spam probability {p:.2f}")
        logger.warning(
            "Jev raised pre-send spam verdict %s → %s (p=%.2f) — subject=%r",
            before, verdict.verdict, p, (subject or "")[:120],
        )
    return verdict


async def scan_for_send_async(
    subject: str,
    body: str,
    *,
    has_unsubscribe: bool = True,
) -> tuple[bool, SpamVerdict]:
    """:func:`scan_async` in the shape a send path wants: ``(may_send, verdict)``."""
    verdict = await scan_async(subject, body, has_unsubscribe=has_unsubscribe)
    return verdict.allowed, verdict
