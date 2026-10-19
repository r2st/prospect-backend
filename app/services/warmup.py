"""Automated email warm-up ramp for new sending accounts.

The ramp (spec):

    day 1–3   → 5/day
    day 4–7   → 10/day
    day 8–14  → 20/day
    day 15–21 → 35/day
    day 22–30 → 50/day
    day 31+   → graduated (the org's normal daily limit applies)

This module owns three responsibilities:

* :func:`limit_for_day` — the pure ramp function (day → (limit, status)).
* :func:`ensure_warmup_plan` — idempotently create a plan when a sender is added.
* :func:`check_send_allowed` / :func:`advance_all` — the runtime gate the send
  path calls before every send, and the daily beat task that steps each plan
  forward.

A plan supersedes the older config-only per-account warm-up in
:mod:`app.services.send_limiter` (``account_warmup_limit``): when a plan row
exists it is authoritative for that account.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.sender_warmup import SenderWarmupPlan
from app.services import send_day

logger = logging.getLogger(__name__)

# (inclusive_last_day, daily_limit) steps of the ramp. The final "graduated"
# stage is handled separately since its limit is the org's normal cap.
_RAMP: tuple[tuple[int, int], ...] = (
    (3, 5),
    (7, 10),
    (14, 20),
    (21, 35),
    (30, 50),
)

# Day at which an account is considered fully warmed.
GRADUATION_DAY = 31

# Ceiling on the ramp day a *pre-existing* sender may be credited with when its
# plan is backfilled (see :func:`credited_start_date`). Day 14 is the top of the
# 20/day band: enough that a mailbox with real history isn't throttled to a
# brand-new 5/day, low enough that it still climbs the last two bands under
# observation rather than jumping straight to full volume.
BACKFILL_MAX_CREDIT_DAY = 14


def org_normal_limit() -> int:
    """The daily cap a graduated account reverts to (the org's normal limit)."""
    return get_settings().max_emails_per_day


def limit_for_day(day: int, graduated_limit: Optional[int] = None) -> tuple[int, str]:
    """Ramp: map a 1-based warm-up day to ``(daily_limit, status)``.

    Days before 1 are clamped to day 1 (a plan can't be pre-start). Day 31+
    graduates to ``graduated_limit`` (defaults to the org's normal daily cap).
    """
    if graduated_limit is None:
        graduated_limit = org_normal_limit()
    d = max(1, int(day))
    for last_day, limit in _RAMP:
        if d <= last_day:
            return limit, "warming"
    return graduated_limit, "graduated"


def day_for(start: date, today: Optional[date] = None) -> int:
    """1-based day on the ramp for ``start`` as of ``today`` (day 1 == start)."""
    if today is None:
        today = send_day.local_date()
    if isinstance(start, datetime):
        start = start.date()
    return max(1, (today - start).days + 1)


def _transport_for(account) -> str:
    """gmail | smtp for a sending-account object."""
    return "smtp" if type(account).__name__ == "SmtpAccount" else "gmail"


def used_today(account) -> int:
    """Today's send count for an account, honouring the daily rollover.

    ``daily_send_count`` is only "today's" count when ``daily_count_reset_at``
    falls inside the current sending day; a stale reset stamp means the stored
    number is yesterday's and will be zeroed on the next send, so it reads as 0
    for warm-up accounting.

    "Today" is the sender's local day (:mod:`app.services.send_day`), matching
    the boundary ``record_account_send`` rolls the counter on and the one the
    org daily cap keys itself by. On the UTC boundary this read, a ramp for a
    mailbox sending into US Pacific hours reset an hour before that contact's
    window closed — so a day-1 mailbox on a 5/day plan sent five, rolled over,
    sent five more, and still reported ``5/5``.
    """
    reset_at = getattr(account, "daily_count_reset_at", None)
    if reset_at is not None and reset_at.tzinfo is None:
        reset_at = reset_at.replace(tzinfo=timezone.utc)
    if reset_at is None or reset_at < send_day.day_start():
        return 0
    try:
        return int(account.daily_send_count or 0)
    except (TypeError, ValueError):
        return 0


async def get_plan(
    db: AsyncSession, account_id: UUID, account_type: str
) -> Optional[SenderWarmupPlan]:
    """The warm-up plan for one account, or ``None`` if it has none."""
    return (
        await db.execute(
            select(SenderWarmupPlan).where(
                SenderWarmupPlan.account_type == account_type,
                SenderWarmupPlan.account_id == account_id,
            )
        )
    ).scalar_one_or_none()


async def ensure_warmup_plan(
    db: AsyncSession,
    *,
    account_id: UUID,
    account_type: str,
    account_email: Optional[str] = None,
    start: Optional[date] = None,
    org_id: Optional[UUID] = None,
) -> SenderWarmupPlan:
    """Create a warm-up plan for a sender if it doesn't already have one.

    Idempotent: returns the existing plan when one is present, so it is safe to
    call every time an account is (re)connected. ``start`` defaults to today, so
    a newly added mailbox begins on day 1 of the ramp.

    ``org_id`` is passed explicitly at account-creation time because that path
    may run without a bound org context (OAuth callback); when omitted the
    org-stamp hook fills it from context.
    """
    existing = await get_plan(db, account_id, account_type)
    if existing is not None:
        return existing

    if start is None:
        start = send_day.local_date()
    day = day_for(start)
    limit, status = limit_for_day(day)

    plan = SenderWarmupPlan(
        account_id=account_id,
        account_type=account_type,
        account_email=account_email,
        start_date=start,
        current_day=day,
        daily_limit=limit,
        status=status,
    )
    if org_id is not None:
        plan.org_id = org_id
    db.add(plan)
    await db.flush()
    logger.info(
        "Created warm-up plan for %s:%s (day %d, limit %d, %s)",
        account_type, account_email or account_id, day, limit, status,
    )
    return plan


async def check_send_allowed(db: AsyncSession, account) -> tuple[bool, str]:
    """Whether ``account`` may send one more email right now under its warm-up plan.

    * No plan, or a graduated plan → allowed (warm-up doesn't constrain it).
    * ``paused`` plan → blocked (the operator is holding this sender).
    * ``warming`` plan → allowed only while today's sends are under the plan's
      ``daily_limit``.

    Returns ``(allowed, reason)``. Fail-open on any error so a warm-up hiccup
    never wedges sending.
    """
    try:
        plan = await get_plan(db, account.id, _transport_for(account))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("warm-up lookup failed for %s: %s", getattr(account, "email", "?"), exc)
        return True, "warm-up check unavailable"

    if plan is None or plan.status == "graduated":
        return True, "no active warm-up limit"
    if plan.status == "paused":
        return False, "warm-up paused for this sender"

    sent = used_today(account)
    if sent >= plan.daily_limit:
        return False, (
            f"warm-up daily limit reached "
            f"(day {plan.current_day}: {sent}/{plan.daily_limit})"
        )
    return True, f"warm-up day {plan.current_day}: {sent}/{plan.daily_limit}"


async def plan_daily_limit(db: AsyncSession, account) -> Optional[int]:
    """The warm-up daily cap for ``account`` from its plan, or ``None``.

    ``None`` means "warm-up doesn't cap this account" (no plan, or graduated), so
    callers fall back to their own limit. A ``paused`` plan reports 0 (send
    nothing).
    """
    plan = await get_plan(db, account.id, _transport_for(account))
    if plan is None or plan.status == "graduated":
        return None
    if plan.status == "paused":
        return 0
    return plan.daily_limit


def recompute(plan: SenderWarmupPlan, today: Optional[date] = None) -> bool:
    """Step a plan forward to ``today``. Returns True if anything changed.

    Recomputes ``current_day`` from ``start_date`` and re-derives the ramp limit.
    A ``paused`` plan is left untouched (its day/limit are frozen until resumed).
    """
    if plan.status == "paused":
        return False
    day = day_for(plan.start_date, today)
    limit, status = limit_for_day(day)
    changed = (
        plan.current_day != day
        or plan.daily_limit != limit
        or plan.status != status
    )
    plan.current_day = day
    plan.daily_limit = limit
    plan.status = status
    return changed


def credited_start_date(account, today: Optional[date] = None) -> date:
    """The ramp start date to give a sender whose plan is being backfilled.

    A mailbox that has never sent (``last_used_at`` unset) is genuinely new and
    starts at day 1 — today.

    A mailbox that *has* sent is not new, and starting it at day 1 throttles an
    already-warm sender to 5/day. That is what happened when plans were first
    backfilled onto senders that had been running for weeks: throughput
    collapsed overnight. So credit its existing history by back-dating the ramp
    to ``connected_at`` — but clamp the credit to
    :data:`BACKFILL_MAX_CREDIT_DAY` so an old account resumes partway up the
    ramp under observation instead of graduating instantly on a date field.
    """
    if today is None:
        today = send_day.local_date()

    last_used = getattr(account, "last_used_at", None)
    if last_used is None:
        return today

    connected = getattr(account, "connected_at", None)
    if connected is None:
        return today
    if isinstance(connected, datetime):
        connected = connected.date()

    credited_day = min(day_for(connected, today), BACKFILL_MAX_CREDIT_DAY)
    return today - timedelta(days=credited_day - 1)


async def backfill_warmup_plans(db: AsyncSession) -> dict:
    """Create a warm-up plan for every connected sender that lacks one.

    ``ensure_warmup_plan`` is wired into the Gmail/SMTP connect paths, so *new*
    accounts get a plan. But accounts connected before that wiring existed have
    none — which is why ``sender_warmup_plans`` sat empty — so a mailbox that
    should be ramping instead sends at full volume. This closes that gap
    idempotently: it is safe to run on every warm-up beat and only ever adds the
    plans that are missing.

    An account that has never sent starts its ramp *today* (day 1). One that has
    already been sending is credited for that history via
    :func:`credited_start_date` — capped partway up the ramp, so it is neither
    throttled back to a brand-new 5/day nor handed full volume on the strength of
    a date field alone.
    """
    from app.models.gmail_account import GmailAccount
    from app.models.smtp_account import SmtpAccount

    created = 0

    gmail_accounts = (
        await db.execute(
            select(GmailAccount).where(GmailAccount.status == "connected")
        )
    ).scalars().all()
    for account in gmail_accounts:
        if await get_plan(db, account.id, "gmail") is None:
            await ensure_warmup_plan(
                db,
                account_id=account.id,
                account_type="gmail",
                account_email=account.email,
                start=credited_start_date(account),
                org_id=getattr(account, "org_id", None),
            )
            created += 1

    smtp_accounts = (
        await db.execute(
            select(SmtpAccount).where(SmtpAccount.status == "connected")
        )
    ).scalars().all()
    for account in smtp_accounts:
        if await get_plan(db, account.id, "smtp") is None:
            await ensure_warmup_plan(
                db,
                account_id=account.id,
                account_type="smtp",
                account_email=account.email,
                start=credited_start_date(account),
                org_id=getattr(account, "org_id", None),
            )
            created += 1

    if created:
        await db.commit()
    return {
        "gmail_accounts": len(gmail_accounts),
        "smtp_accounts": len(smtp_accounts),
        "plans_created": created,
    }


async def repair_uncredited_plans(db: AsyncSession) -> dict:
    """Re-credit plans that were backfilled *before* history-crediting existed.

    :func:`backfill_warmup_plans` only ever creates plans that are **missing**,
    so it cannot repair a plan that already exists with a wrong start date. That
    left a real hole: plans backfilled by the earlier code started every sender
    at day 1 (5/day) regardless of history, and because the row then existed the
    later :func:`credited_start_date` fix never touched it. Six mailboxes that
    had been sending for weeks were pinned to 5/day each, collapsing daily
    capacity to 30 — the throughput floor behind "we send almost nothing".

    This closes that hole by re-deriving the credited start date for every
    connected sender that has real sending history and moving its plan's
    ``start_date`` back to match.

    Deliberately conservative — it only ever moves a plan **forward on the
    ramp** (``start_date`` earlier). It never pushes a sender backwards, never
    touches a ``paused`` plan, and never credits a mailbox that has never sent.
    So it is safe to run on every warm-up beat and is idempotent: once a plan
    holds its credited date, later runs are no-ops.
    """
    from app.models.gmail_account import GmailAccount
    from app.models.smtp_account import SmtpAccount

    repaired = 0

    async def _repair(accounts, account_type: str) -> None:
        nonlocal repaired
        for account in accounts:
            # Never credit a mailbox with no send history — it really is new.
            if getattr(account, "last_used_at", None) is None:
                continue
            plan = await get_plan(db, account.id, account_type)
            if plan is None or plan.status == "paused":
                continue

            credited = credited_start_date(account)
            start = plan.start_date
            if isinstance(start, datetime):
                start = start.date()
            # Only ever move the ramp *earlier* (i.e. further along).
            if start is None or credited >= start:
                continue

            plan.start_date = credited
            recompute(plan)
            repaired += 1
            logger.info(
                "Re-credited warm-up plan for %s:%s — start %s → %s "
                "(now day %d, limit %d)",
                account_type, plan.account_email or account.id,
                start, credited, plan.current_day, plan.daily_limit,
            )

    gmail_accounts = (
        await db.execute(
            select(GmailAccount).where(GmailAccount.status == "connected")
        )
    ).scalars().all()
    await _repair(gmail_accounts, "gmail")

    smtp_accounts = (
        await db.execute(
            select(SmtpAccount).where(SmtpAccount.status == "connected")
        )
    ).scalars().all()
    await _repair(smtp_accounts, "smtp")

    if repaired:
        await db.commit()
    return {"plans_repaired": repaired}


async def advance_all(db: AsyncSession) -> dict:
    """Daily task body: step every plan in the current org forward one day.

    Backfills any missing plans first, so a sender connected before the
    ensure-plan wiring existed still gets a ramp (and the plans table stops being
    empty) rather than silently sending at full volume. Then re-credits any plan
    that an earlier backfill started at day 1 despite the mailbox having real
    sending history (see :func:`repair_uncredited_plans`).

    Returns a small summary ({plans, advanced, graduated, backfilled, repaired})
    for the task log.
    """
    backfill = await backfill_warmup_plans(db)
    repair = await repair_uncredited_plans(db)

    plans = (await db.execute(select(SenderWarmupPlan))).scalars().all()
    advanced = 0
    graduated = 0
    for plan in plans:
        if recompute(plan):
            advanced += 1
        if plan.status == "graduated":
            graduated += 1
    await db.commit()
    return {
        "plans": len(plans),
        "advanced": advanced,
        "graduated": graduated,
        "backfilled": backfill["plans_created"],
        "repaired": repair["plans_repaired"],
    }
