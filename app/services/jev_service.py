"""Shared client for TypeSafe AI's Jev (System One) decision model.

Jev answers *structured* questions about a piece of state — pick one label
(``Choice``), rate on a rubric (``Score``), or give a yes/no probability
(``Noul``) — and returns calibrated probabilities instead of prose. That makes it
a better fit than the LLM chain for the three routing decisions it serves here:

* lead temperature (hot / warm / cold)        — :func:`classify_lead_temperature`
* reply intent (interested, unsubscribe, ...) — :func:`classify_reply_intent`
* outbound spam likelihood                    — :func:`spam_probability`

**Nothing here raises, and nothing here is required.** Every helper returns
``None`` when Jev is unconfigured (no ``TYPESAFE_API_KEY``), the SDK isn't
installed, the call times out, or the response is malformed. Callers treat
``None`` as "no opinion" and keep their pre-Jev behaviour. Jev is consulted on
the send path and in batch rescoring, so after a failure it is skipped for
``jev_failure_cooldown_seconds`` — a rescoring batch of 200 companies would
otherwise pay the timeout 200 times during an outage.

The SDK's async client is created per call rather than cached: Celery tasks run
each pass on a fresh event loop (``_run_async``), and a pooled client bound to a
closed loop fails on its next request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

LEAD_TEMPERATURES: dict[str, str] = {
    "hot": "Strong fit and active buying signals — worth contacting now.",
    "warm": "Plausible fit or some engagement, but no clear buying signal yet.",
    "cold": "Poor fit, missing data, or no engagement at all.",
}

# ``wrong_person`` is not in the original brief's list but is a first-class
# routing category in ``app.tasks.replies`` (it flags the contact for cleanup and
# stashes any referral). Without it Jev would be forced to file "try our CTO"
# under one of the other labels, most likely ``not_interested``.
REPLY_INTENTS: dict[str, str] = {
    "interested": "Positive intent: wants to talk, learn more, asks a qualifying question, or requests a call or demo.",
    "not_interested": "A clear rejection such as 'no thanks' or 'not a fit', without opt-out language.",
    "out_of_office": "An automated out-of-office or vacation autoresponder.",
    "bounce": "An automated delivery-failure notification from a mail server, not a human reply.",
    "unsubscribe": "An explicit request to stop emailing: 'unsubscribe', 'remove me', 'take me off your list'.",
    "wrong_person": "The recipient is not the right contact, has left the company, or points to someone else.",
}

SPAM_INSTRUCTIONS = (
    "This outbound B2B sales email would likely be treated as spam by email "
    "filters or by its recipient."
)
SPAM_CRITERIA = {
    "true": "Spam-like: pushy or deceptive claims, excessive urgency or capitals, link-heavy, or generic mass-mail copy.",
    "false": "A plausible, specific, personal business email a recipient would read as legitimate.",
}

# Long bodies cost tokens and add nothing to these decisions.
MAX_STATE_CHARS = 6000

# Monotonic timestamp until which Jev is skipped after a failure. Process-local
# on purpose: each worker discovers an outage on its own first call.
_cooldown_until: float = 0.0


@dataclass
class JevChoice:
    """A Jev ``Choice`` answer in a plain, JSON-serialisable shape."""

    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    model: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "model": self.model,
        }


def is_configured() -> bool:
    """True when a TypeSafe API key is set. Doesn't check the SDK is installed."""
    return bool((get_settings().typesafe_api_key or "").strip())


def _in_cooldown() -> bool:
    return time.monotonic() < _cooldown_until


def _start_cooldown() -> None:
    global _cooldown_until
    _cooldown_until = time.monotonic() + max(0, int(get_settings().jev_failure_cooldown_seconds))


def reset_cooldown() -> None:
    """Clear the failure cooldown (tests, and an operator re-enabling Jev)."""
    global _cooldown_until
    _cooldown_until = 0.0


def _truncate(text: Optional[str], limit: int = MAX_STATE_CHARS) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


async def system_one(state: str, questions: Mapping[str, Any]) -> Optional[Any]:
    """Ask Jev ``questions`` about ``state``. Returns the SDK response or ``None``.

    ``questions`` maps answer names to SDK question objects (``Choice`` /
    ``Score`` / ``Noul``). Never raises.
    """
    settings = get_settings()
    api_key = (settings.typesafe_api_key or "").strip()
    if not api_key or _in_cooldown():
        return None
    try:
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy
    except ImportError:
        logger.warning("TYPESAFE_API_KEY is set but typesafe-sdk is not installed; skipping Jev")
        return None

    try:
        async with AsyncTypeSafeClient(
            api_key=api_key,
            model=settings.typesafe_model,
            timeout=float(settings.jev_timeout_seconds),
            retry=RetryPolicy(max_retries=0),
        ) as client:
            return await client.system_one(state=_truncate(state), questions=dict(questions))
    except Exception as exc:  # noqa: BLE001 — Jev is advisory; never break the caller
        _start_cooldown()
        logger.warning(
            "Jev call failed (%s: %s); skipping Jev for %ss",
            type(exc).__name__, exc, settings.jev_failure_cooldown_seconds,
        )
        return None


async def ask_choice(
    state: str,
    name: str,
    criteria: Mapping[str, str],
    instructions: Optional[str] = None,
) -> Optional[JevChoice]:
    """One ``Choice`` question → :class:`JevChoice`, or ``None``.

    An answer whose label isn't one of ``criteria`` is discarded rather than
    passed on — callers dispatch on the label.
    """
    try:
        from typesafe_sdk import Choice
    except ImportError:
        return None
    response = await system_one(
        state, {name: Choice(criteria=dict(criteria), instructions=instructions)}
    )
    if response is None:
        return None
    try:
        answer = response.choices[name]
        label = str(answer.choice)
        if label not in criteria:
            logger.warning("Jev returned out-of-vocabulary label %r for %s", label, name)
            return None
        return JevChoice(
            label=label,
            confidence=max(0.0, min(1.0, float(answer.confidence))),
            probabilities={str(k): float(v) for k, v in (answer.probabilities or {}).items()},
            model=getattr(response, "model", None),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Malformed Jev choice response for %s: %s", name, exc)
        return None


async def ask_noul(
    state: str,
    name: str,
    instructions: str,
    criteria: Optional[Mapping[str, str]] = None,
) -> Optional[float]:
    """One ``Noul`` (yes/no) question → probability of "yes" in [0, 1], or ``None``."""
    try:
        from typesafe_sdk import Noul
    except ImportError:
        return None
    response = await system_one(
        state,
        {name: Noul(instructions=instructions, criteria=dict(criteria) if criteria else None)},
    )
    if response is None:
        return None
    try:
        return max(0.0, min(1.0, float(response.nouls[name].noul)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Malformed Jev noul response for %s: %s", name, exc)
        return None


# ─── Use-case helpers ────────────────────────────────────────────────────────


async def classify_lead_temperature(state: str) -> Optional[JevChoice]:
    """hot / warm / cold for a lead described by ``state``."""
    if not get_settings().jev_lead_scoring_enabled:
        return None
    return await ask_choice(
        state,
        "lead_temperature",
        LEAD_TEMPERATURES,
        instructions="How likely is this B2B lead to convert if contacted now?",
    )


def reply_state(
    subject: Optional[str],
    body_text: Optional[str],
    *,
    from_email: Optional[str] = None,
    contact_name: str = "",
    company_name: str = "",
) -> str:
    """The text Jev classifies a reply from."""
    lines = ["An email reply received in response to a cold B2B sales email."]
    if from_email:
        lines.append(f"From: {from_email}")
    if contact_name:
        lines.append(f"Contact: {contact_name}")
    if company_name:
        lines.append(f"Company: {company_name}")
    lines.append(f"Subject: {subject or ''}")
    lines.append("Body:")
    lines.append(_truncate(body_text, 4000))
    return "\n".join(lines)


async def classify_reply_intent(
    subject: Optional[str],
    body_text: Optional[str],
    *,
    from_email: Optional[str] = None,
    contact_name: str = "",
    company_name: str = "",
) -> Optional[JevChoice]:
    """Routing intent for an inbound reply."""
    if not get_settings().jev_reply_classification_enabled:
        return None
    state = reply_state(
        subject, body_text,
        from_email=from_email, contact_name=contact_name, company_name=company_name,
    )
    return await ask_choice(
        state,
        "reply_intent",
        REPLY_INTENTS,
        instructions="What is the intent of this reply?",
    )


async def spam_probability(subject: Optional[str], body: Optional[str]) -> Optional[float]:
    """Probability (0-1) that an outbound email reads as spam."""
    if not get_settings().jev_spam_check_enabled:
        return None
    state = f"Subject: {subject or ''}\n\nBody:\n{_truncate(body, 5000)}"
    return await ask_noul(state, "is_spam", SPAM_INSTRUCTIONS, SPAM_CRITERIA)
