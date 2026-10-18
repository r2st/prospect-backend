"""Hunter.io email verification — the paid per-mailbox layer (P3).

``email_validation`` already proves a *domain* can receive mail (syntax + MX).
It cannot tell ``ceo@acme.com`` (real) from ``nobody@acme.com`` (hard bounce) —
both pass. Hunter's SMTP-level check closes exactly that gap, which matters
because Feature 2 auto-pauses **all** sending at a >5% bounce rate or 5 hard
bounces in 24h: one bad Apollo batch can pause the whole system. Verification is
upstream insurance for that safety rail.

**Layered cheapest-first.** Syntax (free, local) → MX (free, cached per domain)
→ Hunter (paid, per address). Hunter is only called for addresses the free
layers couldn't already condemn, so a syntactically broken address never costs a
credit.

**Fail-open is non-negotiable.** A missing key, a timeout, a 429 that outlives
its retries, or any other error leaves the contact *exactly as it was* —
``email_verified_at`` stays NULL so the next sweep retries it, and the contact
stays sendable. A Hunter outage must degrade to today's MX-only behaviour, never
suppress good contacts. Only a definitive ``undeliverable`` ever writes
``email_status='invalid'``.

``email_status`` remains the single gate every send path reads; the
``email_verification_*`` columns only explain why it holds that value.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import httpx
from sqlalchemy import func as sqla_func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.contact import Contact
from app.services.email_validation import check_mx, domain_of, is_valid_syntax

logger = logging.getLogger(__name__)

# Hunter's four verdicts.
DELIVERABLE = "deliverable"
UNDELIVERABLE = "undeliverable"
RISKY = "risky"
UNKNOWN = "unknown"
RESULTS = frozenset({DELIVERABLE, UNDELIVERABLE, RISKY, UNKNOWN})

# Retry policy for 429 / 5xx. Small and bounded: this runs inside a Celery task
# with a 300s soft limit, and a rate-limited run must be reported as such rather
# than block the queue.
_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.5
_MAX_RETRY_AFTER = 30.0
_TIMEOUT = 30.0


class _Budget:
    """A run's allowance of Hunter calls.

    ``take()`` reserves a credit **before** the HTTP call, so an exhausted budget
    never spends one. Shared across a batch so concurrent verifications draw down
    the same allowance.
    """

    def __init__(self, remaining: float):
        self.remaining = remaining

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0

    def cap(self, limit: int) -> int:
        """Clamp a batch size to what the budget can actually pay for, so a
        sweep never selects rows it will only skip. Unlimited budget → ``limit``
        (``int(inf)`` would raise)."""
        if math.isinf(self.remaining):
            return limit
        return max(0, min(limit, int(self.remaining)))


def _verdict(
    status: str,
    *,
    result: Optional[str] = None,
    score: Optional[int] = None,
    source: Optional[str] = None,
    email_status: Optional[str] = None,
    meta: Optional[dict] = None,
    reason: Optional[str] = None,
) -> dict:
    """Build a verification verdict.

    ``status`` is about the *call*, not the address:
      ``ok``      — a layer reached a verdict; safe to persist.
      ``skipped`` — nothing was attempted (no key / budget exhausted). Persist
                    nothing.
      ``error``   — a layer tried and failed. Persist nothing; retry later.
    Only ``ok`` verdicts are ever written to a contact.
    """
    return {
        "status": status,
        "result": result,
        "score": score,
        "source": source,
        "email_status": email_status,
        "meta": meta or {},
        "reason": reason,
    }


class HunterVerificationService:
    """Wraps Hunter.io's Email Verifier + Email Finder APIs."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = (self.settings.hunter_base_url or "").rstrip("/")
        # Override slot for callers/tests that assign ``svc.api_key``. None means
        # "no override"; an assigned "" is a deliberate "act as if unkeyed".
        self._api_key_override: Optional[str] = None

    @property
    def api_key(self) -> str:
        """The Hunter key, resolved at *access* time, never at construction.

        A key read once in ``__init__`` is the environment as it was when the
        object was built — wrong for a long-lived worker, wrong after a config
        reload, and it turns "no key yet" into a property of the process rather
        than of the request. Resolving on access keeps a missing key exactly what
        the fail-open design says it is: a runtime skip, not a startup failure.
        """
        if self._api_key_override is not None:
            return self._api_key_override
        return get_settings().hunter_api_key or ""

    @api_key.setter
    def api_key(self, value: Optional[str]) -> None:
        self._api_key_override = value

    @property
    def enabled(self) -> bool:
        """False when no key is configured — the capability is simply off."""
        return bool(self.api_key)

    # ── budget ──────────────────────────────────────────────────────────────

    async def hunter_calls_today(self, db: AsyncSession) -> int:
        """Hunter-sourced verifications recorded so far in the current UTC day.

        This is a *guardrail*, not billing truth — it counts contacts currently
        carrying a Hunter verdict stamped today, so a re-verification of the same
        contact still counts once, and calls that errored (which Hunter generally
        does not charge for) don't count at all. Good enough to stop a
        misconfigured sweep from burning a month of credits; reconcile against
        Hunter's own dashboard. P2's ``ApiCreditLedger`` supersedes this when it
        lands.
        """
        start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        result = await db.execute(
            select(sqla_func.count())
            .select_from(Contact)
            .where(
                Contact.email_verification_source == "hunter",
                Contact.email_verified_at >= start,
            )
        )
        return int(result.scalar() or 0)

    async def make_budget(self, db: AsyncSession) -> _Budget:
        """Today's remaining Hunter allowance. Budget 0 = unlimited."""
        limit = self.settings.hunter_daily_credit_budget
        if limit <= 0:
            return _Budget(math.inf)
        spent = await self.hunter_calls_today(db)
        return _Budget(max(0, limit - spent))

    # ── HTTP ────────────────────────────────────────────────────────────────

    @staticmethod
    def _retry_delay(response: httpx.Response, fallback: float) -> float:
        """Honour Retry-After when Hunter sends one, clamped so a hostile or
        absurd value can't stall the worker."""
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return min(float(raw), _MAX_RETRY_AFTER)
            except (TypeError, ValueError):
                pass
        return fallback

    async def _get(self, path: str, params: dict) -> tuple[Optional[dict], Optional[str]]:
        """GET a Hunter endpoint with bounded backoff.

        Returns ``(data, None)`` on success or ``(None, reason)`` on failure.
        Never raises — every caller treats a failure as fail-open.
        """
        # The key goes in a header, never the query string: httpx logs every
        # request URL at INFO, so ``?api_key=`` put the credential in every log
        # line of every sweep — and logs are shipped and retained.
        headers = {"X-API-KEY": self.api_key}
        delay = _BACKOFF_BASE

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                    response = await client.get(
                        f"{self.base_url}{path}", params=params, headers=headers
                    )

                # Retryable: rate limit or a server-side wobble.
                if response.status_code == 429 or response.status_code >= 500:
                    reason = (
                        "rate_limited"
                        if response.status_code == 429
                        else f"http_{response.status_code}"
                    )
                    if attempt == _MAX_ATTEMPTS:
                        logger.warning(
                            "Hunter %s failed after %d attempts: %s",
                            path, attempt, reason,
                        )
                        return None, reason
                    await asyncio.sleep(self._retry_delay(response, delay))
                    delay *= 2
                    continue

                # Other 4xx won't improve on retry (bad key, malformed address,
                # plan exhausted). Fail fast rather than spend the retries.
                if response.status_code >= 400:
                    logger.warning(
                        "Hunter %s returned %d", path, response.status_code
                    )
                    return None, f"http_{response.status_code}"

                return response.json(), None

            except Exception as e:  # network error, timeout, bad JSON
                if attempt == _MAX_ATTEMPTS:
                    logger.warning("Hunter %s errored: %s", path, e)
                    return None, str(e) or e.__class__.__name__
                await asyncio.sleep(delay)
                delay *= 2

        return None, "exhausted"  # pragma: no cover - loop always returns above

    # ── verification ────────────────────────────────────────────────────────

    def _map_hunter(self, payload: dict) -> dict:
        """Map a Hunter email-verifier payload onto a verdict.

        The ``accept_all`` case is the one that matters: a catch-all domain
        accepts every address at SMTP time and bounces later, so Hunter
        genuinely cannot tell a real mailbox from a typo. Those come back as
        ``risky`` — sendable by a human, never auto-enrollable.
        """
        data = (payload or {}).get("data") or {}
        raw_result = (data.get("result") or "").strip().lower()
        result = raw_result if raw_result in RESULTS else UNKNOWN

        score = data.get("score")
        try:
            score = int(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        accept_all = bool(data.get("accept_all"))
        # Hunter reports a catch-all as deliverable-with-accept_all often enough
        # that trusting `result` alone would auto-enroll a domain that bounces
        # later. Downgrade it to what it actually is: unknowable.
        if accept_all and result == DELIVERABLE:
            result = RISKY

        meta = {
            "accept_all": accept_all,
            "disposable": bool(data.get("disposable")),
            "webmail": bool(data.get("webmail")),
            "gibberish": bool(data.get("gibberish")),
            "smtp_check": data.get("smtp_check"),
            "mx_records": data.get("mx_records"),
            "block": bool(data.get("block")),
            "hunter_result": raw_result or None,
            "hunter_status": data.get("status"),
        }

        if result == DELIVERABLE:
            email_status = "valid"
        elif result == UNDELIVERABLE:
            email_status = "invalid"
        else:
            # risky / unknown → sendable, but honestly uncertain. Never 'invalid'.
            email_status = "unknown"

        return _verdict(
            "ok",
            result=result,
            score=score,
            source="hunter",
            email_status=email_status,
            meta=meta,
        )

    async def _precheck(self, address: str) -> Optional[dict]:
        """The free layers. Returns a verdict to short-circuit on, or None when
        the address survives to the paid layer.

        Note an MX result of ``unknown`` does NOT short-circuit: dnspython may be
        absent or DNS may be down locally, neither of which says anything about
        the mailbox. Only a definitive ``invalid`` stops us — otherwise a missing
        optional dependency would silently disable Hunter entirely.
        """
        if not is_valid_syntax(address):
            return _verdict(
                "ok",
                result=UNDELIVERABLE,
                source="mx",
                email_status="invalid",
                meta={"reason": "syntax"},
            )

        mx = await asyncio.to_thread(check_mx, domain_of(address))
        if mx == "invalid":
            return _verdict(
                "ok",
                result=UNDELIVERABLE,
                source="mx",
                email_status="invalid",
                meta={"reason": "no_mx"},
            )
        return None

    async def verify_email(
        self, email: Optional[str], *, budget: Optional[_Budget] = None
    ) -> dict:
        """Verify one address: syntax → MX → Hunter.

        ``budget`` gates the paid layer and is checked **before** the HTTP call.
        Pass None for no gate (callers with a ``db`` should use ``make_budget``).
        Never raises.
        """
        address = (email or "").strip()

        precheck = await self._precheck(address)
        if precheck is not None:
            return precheck

        if not self.api_key:
            # Not an error — the capability is off. Fall back to MX-only.
            return _verdict("skipped", reason="no_api_key")

        if budget is not None and not budget.take():
            logger.warning("Hunter daily credit budget exhausted — skipping %s", address)
            return _verdict("skipped", reason="budget_exhausted")

        payload, error = await self._get("/email-verifier", {"email": address})
        if error is not None or payload is None:
            # Fail open: the address is not condemned, just unconfirmed.
            return _verdict("error", reason=error or "empty_response")

        return self._map_hunter(payload)

    def _apply(self, contact: Contact, verdict: dict) -> bool:
        """Persist a verdict onto a contact. Returns True if anything was written.

        Non-``ok`` verdicts write nothing at all: leaving ``email_verified_at``
        NULL is what makes the next sweep retry the contact instead of writing it
        off on the strength of an outage.
        """
        if verdict.get("status") != "ok":
            return False

        now = datetime.now(timezone.utc)
        contact.email_verification_result = verdict["result"]
        contact.email_verification_score = verdict["score"]
        contact.email_verification_source = verdict["source"]
        contact.email_verification_meta = verdict["meta"] or {}
        contact.email_verified_at = now
        if verdict["email_status"]:
            contact.email_status = verdict["email_status"]
            contact.email_checked_at = now

        if verdict["result"] == UNDELIVERABLE:
            logger.info(
                "Contact %s verified undeliverable (%s) — will never be emailed",
                contact.id, contact.email,
            )
        return True

    async def verify_contact(
        self, db: AsyncSession, contact: Contact, *, budget: Optional[_Budget] = None
    ) -> dict:
        """Verify a contact and persist the verdict. Does NOT commit — the caller
        owns the transaction."""
        if budget is None:
            budget = await self.make_budget(db)
        verdict = await self.verify_email(contact.email, budget=budget)
        self._apply(contact, verdict)
        return verdict

    async def verify_batch(
        self,
        db: AsyncSession,
        contacts: list[Contact],
        *,
        budget: Optional[_Budget] = None,
    ) -> list[dict]:
        """Verify many contacts under one shared budget, with bounded concurrency.

        Persists each verdict. Does NOT commit. One contact's failure never sinks
        the batch.
        """
        if not contacts:
            return []
        if budget is None:
            budget = await self.make_budget(db)

        semaphore = asyncio.Semaphore(max(1, self.settings.hunter_concurrency))

        async def _one(contact: Contact) -> dict:
            async with semaphore:
                return await self.verify_email(contact.email, budget=budget)

        verdicts = await asyncio.gather(
            *(_one(c) for c in contacts), return_exceptions=True
        )

        results: list[dict] = []
        for contact, verdict in zip(contacts, verdicts):
            if isinstance(verdict, BaseException):
                logger.warning(
                    "Verification raised for contact %s: %s", contact.id, verdict
                )
                verdict = _verdict("error", reason=str(verdict))
            self._apply(contact, verdict)
            results.append(
                {
                    "contact_id": str(contact.id),
                    "email": contact.email,
                    "status": verdict["status"],
                    "result": verdict["result"],
                    "score": verdict["score"],
                    "source": verdict["source"],
                    "email_status": contact.email_status,
                    "reason": verdict["reason"],
                }
            )
        return results

    async def bulk_verify(
        self, emails: Iterable[str], *, db: Optional[AsyncSession] = None
    ) -> list[dict]:
        """Verify a list of raw addresses (no persistence).

        Pass ``db`` to apply the daily credit gate; without it the batch is
        ungated. Order matches the input.
        """
        addresses = list(emails)
        if not addresses:
            return []
        budget = await self.make_budget(db) if db is not None else None
        semaphore = asyncio.Semaphore(max(1, self.settings.hunter_concurrency))

        async def _one(email: str) -> dict:
            async with semaphore:
                verdict = await self.verify_email(email, budget=budget)
                return {**verdict, "email": email}

        verdicts = await asyncio.gather(
            *(_one(e) for e in addresses), return_exceptions=True
        )
        out: list[dict] = []
        for email, verdict in zip(addresses, verdicts):
            if isinstance(verdict, BaseException):
                logger.warning("Verification raised for %s: %s", email, verdict)
                verdict = {**_verdict("error", reason=str(verdict)), "email": email}
            out.append(verdict)
        return out

    # ── recovery ────────────────────────────────────────────────────────────

    async def find_email(
        self,
        db: AsyncSession,
        company,
        first_name: Optional[str],
        last_name: Optional[str],
        *,
        budget: Optional[_Budget] = None,
    ) -> Optional[str]:
        """Recover an address from a name + company domain via Hunter's finder.

        Apollo regularly returns a person with ``email: null``, and
        ``find_contacts`` inserts them un-emailable. This converts some of those
        dead rows back into usable contacts. The returned address is a *guess* —
        the caller must still verify it before sending.
        """
        domain = getattr(company, "domain", None)
        if not self.api_key or not domain or not (first_name and last_name):
            return None

        if budget is None:
            budget = await self.make_budget(db)
        if not budget.take():
            logger.warning("Hunter budget exhausted — skipping email-finder for %s", domain)
            return None

        payload, error = await self._get(
            "/email-finder",
            {"domain": domain, "first_name": first_name, "last_name": last_name},
        )
        if error is not None or not payload:
            return None

        data = payload.get("data") or {}
        email = (data.get("email") or "").strip() or None
        if email:
            logger.info(
                "Hunter recovered %s for %s %s at %s (confidence %s)",
                email, first_name, last_name, domain, data.get("score"),
            )
        return email

    async def domain_search(
        self,
        db: AsyncSession,
        domain: str,
        *,
        budget: Optional[_Budget] = None,
    ) -> dict:
        """Find addresses for a domain via Hunter's ``/domain-search``.

        The bridge P4 needs: Google Places gives a business a website and no
        email, and ``find_email`` can't help because Places gives us no *names*
        either. Domain-search is the one Hunter endpoint that goes from a bare
        domain to addresses.

        Returns ``{"email", "score", "emails", "status"}`` — ``email`` is the
        best candidate or None. Never raises; a failure returns
        ``email: None`` so the caller falls back to scraping.

        **Personal beats generic regardless of confidence.** Hunter scores a
        catch-all ``info@`` at 90+ and a real ``priya.sharma@`` at 70, but the
        generic one lands in a shared inbox nobody answers — and it's precisely
        the kind of address that accept_all domains bounce later. Sorting on
        confidence alone would pick the wrong one almost every time.
        """
        domain = (domain or "").strip().lower()
        if not self.api_key:
            return {"status": "skipped", "reason": "no_api_key", "email": None, "emails": []}
        if not domain:
            return {"status": "skipped", "reason": "no_domain", "email": None, "emails": []}

        if budget is None:
            budget = await self.make_budget(db)
        if not budget.take():
            logger.warning("Hunter budget exhausted — skipping domain-search for %s", domain)
            return {"status": "skipped", "reason": "budget_exhausted", "email": None, "emails": []}

        payload, error = await self._get("/domain-search", {"domain": domain, "limit": 10})
        if error is not None or not payload:
            return {"status": "error", "reason": error, "email": None, "emails": []}

        entries = ((payload.get("data") or {}).get("emails")) or []
        emails = [
            {
                "email": (e.get("value") or "").strip().lower(),
                "type": e.get("type"),
                "score": e.get("confidence"),
                "first_name": e.get("first_name"),
                "last_name": e.get("last_name"),
                "position": e.get("position"),
            }
            for e in entries
            if (e.get("value") or "").strip()
        ]
        if not emails:
            return {"status": "not_found", "email": None, "emails": []}

        ranked = sorted(
            emails,
            key=lambda e: (e.get("type") == "personal", e.get("score") or 0),
            reverse=True,
        )
        best = ranked[0]
        logger.info(
            "Hunter domain-search found %d address(es) for %s; best %s (%s, confidence %s)",
            len(emails), domain, best["email"], best.get("type"), best.get("score"),
        )
        return {
            "status": "ok",
            "email": best["email"],
            "score": best.get("score"),
            "emails": ranked,
        }

    # ── routing helper ──────────────────────────────────────────────────────

    def is_auto_enroll_eligible(self, contact: Contact) -> bool:
        """Whether a contact may be enrolled into outreach **automatically**.

        Only a Hunter-confirmed ``deliverable`` at or above ``hunter_min_score``
        qualifies. ``risky`` (which includes every catch-all/accept_all domain),
        ``unknown``, and never-verified contacts are enrollable by a human but
        never by the machine — auto-enrolling a catch-all is how a batch of
        later-bouncing addresses reaches the sender and trips Feature 2.

        Intended as P5's high-confidence band; see the spec's routing table.
        """
        if getattr(contact, "suppressed", False) or getattr(contact, "unsubscribed", False):
            return False
        if contact.email_status == "invalid":
            return False
        if contact.email_verification_result != DELIVERABLE:
            return False
        score = contact.email_verification_score
        return score is not None and score >= self.settings.hunter_min_score

    # ── sweeps ──────────────────────────────────────────────────────────────

    def _emailable(self):
        """Filters shared by both sweeps: only contacts we could actually email.

        Verifying someone we will never contact again is pure credit burn.
        """
        return (
            Contact.email.isnot(None),
            Contact.email != "",
            Contact.suppressed.is_(False),
            Contact.unsubscribed.is_(False),
        )

    async def select_pending(self, db: AsyncSession, limit: int) -> list[Contact]:
        """Contacts that have never reached a verification verdict."""
        result = await db.execute(
            select(Contact)
            .where(*self._emailable(), Contact.email_verified_at.is_(None))
            .order_by(Contact.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def select_stale(self, db: AsyncSession, limit: int) -> list[Contact]:
        """Contacts whose verification has aged past ``hunter_reverify_days``.

        Oldest-first so a capped run always re-verifies the rottenest addresses.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.settings.hunter_reverify_days
        )
        result = await db.execute(
            select(Contact)
            .where(
                *self._emailable(),
                Contact.email_verified_at.isnot(None),
                Contact.email_verified_at < cutoff,
                # A known-bad address is never emailed, so re-verifying it can
                # only burn credits. (NULL-safe: NULL != 'invalid' is NULL, which
                # would silently drop unverified rows.)
                or_(
                    Contact.email_status.is_(None),
                    Contact.email_status != "invalid",
                ),
            )
            .order_by(Contact.email_verified_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())


def get_hunter_service() -> HunterVerificationService:
    return HunterVerificationService()
