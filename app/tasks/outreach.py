"""Celery tasks for outreach email sequences."""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database_celery import celery_session
from app.models.company import Company
from app.models.contact import Contact
from app.models.consent import ConsentRecord
from app.models.follow_up import FollowUp
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.models.sequence import OutreachSequence, SequenceStep
from app.services.consent import check_consent
from app.services.email_compose import (
    render,
    build_merge_context,
    clean_body,
    clean_subject,
    subject_context,
)
from app.services.email_dispatch import send_outreach_email
from app.services.sequence_optimizer import get_sequence_optimizer, record_decision
from app.tasks.celery_app import celery_app
from app.tasks.org_runner import for_each_org, org_scope
from app.services import send_claim
from app.services.send_limiter import get_send_limiter
from app.services.send_time_optimizer import get_send_time_optimizer

logger = logging.getLogger(__name__)

# How far to push ``next_send_at`` when a send is soft-skipped (warm-up cap hit,
# suppression, kill switch). These conditions clear on a timescale of hours, but
# ``process_outreach_queue`` re-dispatches anything whose ``next_send_at`` is due
# every 5 minutes — and each re-dispatch pays for contact research and an AI
# composition *before* reaching the gate that declines the send. Without a
# back-off a single capped mailbox turns into a hot loop that burns the LLM
# quota (and, once the provider starts answering 429, produces nothing else).
SOFT_SKIP_BACKOFF = timedelta(hours=1)


async def _defer_enrollment(
    db: AsyncSession, enrollment_id: str, delay: timedelta
) -> None:
    """Push an enrollment's ``next_send_at`` out by ``delay``.

    Issued as a Core UPDATE keyed by id rather than through the ORM object: the
    callers run right after a ``rollback()``, which expires every instance in the
    session, so touching the loaded enrollment would trigger a lazy refresh.
    Best-effort — a failure here only costs an earlier retry, so it must never
    turn a benign soft-skip into a task failure.
    """
    try:
        await db.execute(
            update(OutreachEnrollment)
            .where(OutreachEnrollment.id == UUID(enrollment_id))
            .values(next_send_at=datetime.now(timezone.utc) + delay)
            .execution_options(skip_org_filter=True)
        )
        await db.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not defer enrollment %s: %s", enrollment_id, exc)
        await db.rollback()


async def _handle_duplicate_claim(
    db: AsyncSession, enrollment_id: str, step_number: int, claim
) -> dict:
    """What to do when this step is already claimed by another attempt.

    The refusal itself is the safety property — no second message goes out. What
    is left is the enrollment, which is still sitting on a step it will never be
    allowed to send, and would otherwise be re-dispatched every five minutes for
    the rest of its life.

    Told apart by the age of the claim:

    * fresh — a sibling worker is mid-send. Back off by the same interval a soft
      skip uses and let it finish; it advances the enrollment itself. If it
      already has, ``_defer_enrollment`` would drag a step scheduled days out
      back to an hour from now, so the deferral is applied only while the
      enrollment is still on the claimed step.
    * stale — nothing came back from the attempt that took this claim. The
      message went out and its bookkeeping is lost, so
      :func:`send_claim.reconcile` writes the send that is missing and moves the
      cadence on rather than leaving the enrollment wedged.

    ``db`` is the send's own session, already rolled back by the caller. It is
    reused rather than opened fresh because that rollback is what released the
    enrollment row; a second connection here would only add another lock holder
    to a row this task is still finishing with.
    """
    if claim.stale:
        reconciled = await send_claim.reconcile(db, enrollment_id, step_number)
        if reconciled:
            return {
                "status": "reconciled",
                "reason": (
                    "step was claimed for the wire and never recorded — "
                    "recovered rather than re-sent"
                ),
                "step": step_number,
                "enrollment_id": enrollment_id,
            }
        # The original attempt came back after all; nothing to fix.
        return {
            "status": "skipped",
            "reason": "duplicate_send_suppressed",
            "step": step_number,
            "enrollment_id": enrollment_id,
        }

    still_here = (
        await db.execute(
            select(OutreachEnrollment.current_step)
            .where(OutreachEnrollment.id == UUID(enrollment_id))
            .execution_options(skip_org_filter=True)
        )
    ).scalar_one_or_none()
    if still_here == step_number:
        await _defer_enrollment(db, enrollment_id, SOFT_SKIP_BACKOFF)

    logger.warning(
        "Enrollment %s step %s already claimed for the wire — not sending again",
        enrollment_id, step_number,
    )
    return {
        "status": "skipped",
        "reason": "duplicate_send_suppressed",
        "step": step_number,
        "enrollment_id": enrollment_id,
    }


def _run_async(coro):
    """Run async code in Celery's sync context with a fresh event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _advance_step(db: AsyncSession, enrollment: OutreachEnrollment) -> None:
    """Advance an enrollment to its next step without sending.

    Increments ``current_step`` and either schedules ``next_send_at`` from the
    upcoming step's delay, or completes the enrollment when no step remains.
    Used by optimizer skip/switch-channel paths (the send path advances inline).
    """
    now = datetime.now(timezone.utc)
    enrollment.current_step += 1
    next_step = (
        await db.execute(
            select(SequenceStep).where(
                and_(
                    SequenceStep.sequence_id == enrollment.sequence_id,
                    SequenceStep.step_number == enrollment.current_step,
                )
            )
        )
    ).scalar_one_or_none()
    if next_step is not None:
        # One resolver for every due date in the system (``services.step_delay``):
        # the step's days + hours, counted in business days when it asks for it,
        # scaled by the enrollment's engagement-driven adaptive factor, spread by
        # a stable per-enrollment jitter.
        from app.services import step_delay

        enrollment.next_send_at = step_delay.due_at(
            next_step,
            now,
            factor=enrollment.adaptive_delay_factor,
            enrollment_id=enrollment.id,
        )
    else:
        enrollment.status = "completed"
        enrollment.completed_at = now


async def _prior_step_engagement(
    db: AsyncSession, enrollment: OutreachEnrollment, prior_step_number: int
) -> dict:
    """Whether the enrollment's prior step was opened / the contact ever replied.

    Reads ``OutreachEvent`` rows for this enrollment: an ``opened``/``clicked``
    event on ``prior_step_number`` means the prior touch was opened; a
    ``replied`` event on any step means they responded. Used to evaluate a
    step's ``send_condition``.

    Also returns ``prior_tracked``: whether the prior step's send actually
    carried an open pixel. Absence of an open only means "not opened" if we were
    looking — for a step sent untracked (tracking disabled, or sent before
    tracking existed) a missing open means *unknown*, and the caller must not
    read it as a negative.
    """
    rows = (
        await db.execute(
            select(
                OutreachEvent.event_type,
                OutreachEvent.step_number,
                OutreachEvent.event_metadata,
            ).where(OutreachEvent.enrollment_id == enrollment.id)
        )
    ).all()
    opened = any(
        et in ("opened", "clicked") and sn == prior_step_number for et, sn, _ in rows
    )
    # Clicks are counted across the whole enrollment, not just the prior step.
    # An open is about one message ("did that subject land?"); a click is about
    # the person ("are they interested enough to look?"), and interest shown at
    # step 2 is still interest when step 4 decides how direct to be.
    clicked = any(et == "clicked" for et, _, _ in rows)
    replied = any(et == "replied" for et, _, _ in rows)
    prior_tracked = any(
        et == "sent" and sn == prior_step_number and bool((md or {}).get("tracked"))
        for et, sn, md in rows
    )
    return {
        "opened": opened,
        "clicked": clicked,
        "replied": replied,
        "prior_tracked": prior_tracked,
    }


async def evaluate_step_condition(
    db: AsyncSession, enrollment: OutreachEnrollment, step: SequenceStep
) -> tuple[bool, str]:
    """Decide whether a step should send given its ``send_condition``.

    * ``always`` (or unknown) → always send.
    * ``if_not_opened`` → send only if the prior step was NOT opened.
    * ``if_opened_no_reply`` → send only if the prior step WAS opened and the
      contact has not replied.
    * ``if_clicked_no_reply`` → send only if they clicked a link anywhere in this
      enrollment and have not replied.
    * ``if_not_clicked`` → send only if they have never clicked.

    The click branches exist because the open signal is the weakest one we have.
    Apple Mail Privacy Protection pre-fetches images, so a meaningful share of
    "opens" are a proxy server, not a person; Gmail's image proxy inflates the
    rest. A click is a deliberate act by a human. Keying the branch that decides
    *how direct to be* on clicks rather than opens is the difference between
    reacting to interest and reacting to a mail server.

    The first step (no prior touch) has nothing to branch on: ``if_not_opened``
    and ``if_not_clicked`` send (nothing has happened yet); ``if_opened_no_reply``
    and ``if_clicked_no_reply`` do not. Returns ``(should_send, reason)``.

    **Fails open on untracked sends.** ``if_opened_no_reply`` gates on a positive
    open signal, so if the prior step went out with no pixel there is no way for
    that gate to ever pass and the step would be skipped silently, forever — the
    exact failure this branch had before sequence tracking existed. When the
    prior send was untracked we send instead of skipping: a step that goes out
    when it maybe shouldn't is recoverable; one that can never go out is not.
    """
    condition = getattr(step, "send_condition", "always") or "always"
    if condition == "always":
        return True, "always"

    eng = await _prior_step_engagement(db, enrollment, step.step_number - 1)
    if condition == "if_not_opened":
        if eng["opened"]:
            return False, "prior step was opened — skipping the not-opened branch"
        return True, "prior step not opened"
    if condition == "if_opened_no_reply":
        if eng["replied"]:
            return False, "contact already replied — skipping"
        if not eng["opened"]:
            if not eng["prior_tracked"]:
                return True, (
                    "prior step was sent without open tracking — open state unknown, "
                    "sending rather than skipping a step that could never fire"
                )
            return False, "prior step not opened — skipping the opened-no-reply branch"
        return True, "prior step opened, no reply yet"
    if condition == "if_clicked_no_reply":
        if eng["replied"]:
            return False, "contact already replied — skipping"
        if not eng["clicked"]:
            # Same fail-open rule as ``if_opened_no_reply``, for the same reason:
            # a click can only be observed on a tracked send, so gating on a
            # positive click after an untracked one is a branch that could never
            # fire. Untracked → unknown, and unknown sends.
            if not eng["prior_tracked"]:
                return True, (
                    "prior step was sent without link tracking — click state "
                    "unknown, sending rather than skipping a step that could "
                    "never fire"
                )
            return False, "no click yet — skipping the clicked-no-reply branch"
        return True, "contact clicked a link, no reply yet"
    if condition == "if_not_clicked":
        if eng["clicked"]:
            return False, "contact already clicked — skipping the not-clicked branch"
        return True, "no click yet"
    return True, "unknown condition — defaulting to send"


@celery_app.task(name="app.tasks.outreach.send_sequence_step", bind=True, max_retries=3)
def send_sequence_step(self, enrollment_id: str):
    """Send the current step of a sequence to the enrolled contact.

    Checks consent, renders the template, sends via SES, and records the event.
    """
    logger.info("Processing sequence step for enrollment %s", enrollment_id)

    async def _run():
        # This task is dispatched per-enrollment, so the org context set by the
        # caller is lost over the Celery wire. Resolve the enrollment's org
        # up front (unfiltered) and bind it for the whole send, so the kill
        # switch, reads and the recorded event all belong to the right tenant.
        async with celery_session() as db0:
            org_id = (
                await db0.execute(
                    select(OutreachEnrollment.org_id)
                    .where(OutreachEnrollment.id == UUID(enrollment_id))
                    .execution_options(skip_org_filter=True)
                )
            ).scalar_one_or_none()
        async with org_scope(org_id):
            return await _run_inner()

    async def _run_inner():
        async with celery_session() as db:
            try:
                # Global kill switch — never send while sending is halted. Checked
                # first so a paused system does no work and takes no side effects.
                from app.services import system_settings

                if not await system_settings.is_sending_enabled(db):
                    logger.warning(
                        "send_sequence_step halted for %s: kill switch engaged", enrollment_id
                    )
                    return {"status": "halted", "reason": "sending kill switch engaged",
                            "enrollment_id": enrollment_id}

                # Load enrollment with related data
                result = await db.execute(
                    select(OutreachEnrollment)
                    .where(OutreachEnrollment.id == UUID(enrollment_id))
                )
                enrollment = result.scalar_one_or_none()
                if not enrollment:
                    return {"status": "not_found", "enrollment_id": enrollment_id}

                if enrollment.status != "active":
                    return {"status": "skipped", "reason": f"enrollment status is {enrollment.status}"}

                # Pausing a campaign has to stop the campaign's mail.
                #
                # ``PATCH /sequences/{id}/status`` is the operator's "pause this
                # campaign" control: it sets ``status='paused'`` and syncs
                # ``is_active=False``, and the console renders the sequence as
                # paused. Nothing on the send path had ever read either column.
                # ``is_active`` gated *enrolling* (``POST /outreach/enrollments``
                # 400s on an inactive sequence) and listing, so the control
                # stopped new contacts entering — and every contact already
                # enrolled kept receiving steps 2..N on schedule, from a campaign
                # the UI said was stopped. Same shape as the kill switch and the
                # unsubscribe link before them: a control that reads as enforced
                # everywhere except where the mail leaves.
                #
                # A hold, not a stop. The enrollment stays ``active`` and its
                # ``next_send_at`` is untouched, so resuming the sequence resumes
                # the cadence rather than requiring every contact to be re-enrolled
                # — pausing is reversible by construction and un-pausing must not
                # need a second, undocumented repair step. The queue sweep filters
                # these out in SQL, so the refusal here is the interlock for a task
                # already in the broker when the pause landed, not the hot path.
                sequence = (
                    await db.execute(
                        select(OutreachSequence).where(
                            OutreachSequence.id == enrollment.sequence_id
                        )
                    )
                ).scalar_one_or_none()
                if sequence is not None and not sequence.is_active:
                    logger.info(
                        "Enrollment %s held: sequence %s is paused",
                        enrollment_id, enrollment.sequence_id,
                    )
                    return {
                        "status": "skipped",
                        "reason": "sequence_paused",
                        "enrollment_id": enrollment_id,
                    }

                # Follow-ups stop on reply. SmartReply flips the enrollment to
                # 'replied' when it matches the inbound message, but a reply can
                # also be recorded by another path (tracking webhook, manual
                # log) that only writes the event. Either one ends the cadence:
                # a follow-up to someone who already answered is the worst
                # email the sequence can send.
                if (enrollment.current_step or 1) > 1:
                    has_reply = (
                        await db.execute(
                            select(OutreachEvent.id)
                            .where(
                                OutreachEvent.enrollment_id == enrollment.id,
                                OutreachEvent.event_type == "replied",
                            )
                            .limit(1)
                        )
                    ).first()
                    if has_reply is not None:
                        enrollment.status = "replied"
                        enrollment.completed_at = datetime.now(timezone.utc)
                        await db.commit()
                        logger.info(
                            "Enrollment %s stopped: contact already replied", enrollment_id
                        )
                        return {"status": "skipped", "reason": "contact_replied",
                                "enrollment_id": enrollment_id}

                # Load contact
                result = await db.execute(
                    select(Contact).where(Contact.id == enrollment.contact_id)
                )
                contact = result.scalar_one_or_none()
                if not contact or not contact.email:
                    enrollment.status = "paused"
                    await db.commit()
                    return {"status": "paused", "reason": "no_contact_email"}

                # Hard suppression (Feature 1): never send to a suppressed contact.
                # Three sources: the per-contact ``suppressed`` flag, the legacy
                # ``unsubscribed`` flag, and the org do-not-contact suppression
                # LIST keyed by email (Feature 2 — a bulk compliance import that
                # may name an address which isn't even a contact). Any one of them
                # stops the enrollment as unsubscribed (NOT bounced — a compliance
                # stop must not dent sender metrics).
                #
                # ``unsubscribed`` is checked because this gate not checking it is
                # how the footer unsubscribe link — the only opt-out a recipient
                # can actually reach — failed to stop anything for as long as it
                # existed. ``opt_out.record`` now sets both flags and the list, so
                # this is redundant by design: three writers have set this column
                # alone at various times, and the cost of reading it is an
                # attribute access.
                from app.services import suppression as _suppression

                if (
                    getattr(contact, "suppressed", False)
                    or getattr(contact, "unsubscribed", False)
                    or await _suppression.is_suppressed(db, contact.email)
                ):
                    enrollment.status = "unsubscribed"
                    enrollment.completed_at = datetime.now(timezone.utc)
                    await db.commit()
                    logger.info(
                        "Enrollment %s stopped: contact %s is suppressed",
                        enrollment_id, contact.email,
                    )
                    return {"status": "skipped", "reason": "contact_suppressed"}

                # Load the sequence step before the deliverability gates below,
                # because most of those gates are about *email* and this step may
                # not be one. A "call" step is not limited by a Gmail mailbox's
                # warm-up ceiling, does not consume the daily send budget, and
                # cannot bounce off a missing MX record — gating it on any of
                # those blocks a phone call on the state of an inbox.
                result = await db.execute(
                    select(SequenceStep).where(
                        and_(
                            SequenceStep.sequence_id == enrollment.sequence_id,
                            SequenceStep.step_number == enrollment.current_step,
                        )
                    )
                )
                step = result.scalar_one_or_none()
                if not step:
                    enrollment.status = "completed"
                    enrollment.completed_at = datetime.now(timezone.utc)
                    await db.commit()
                    return {"status": "completed", "reason": "no_more_steps"}

                # A step with no channel set is an email step — that is the
                # column's default and every step predating multi-channel.
                is_email_step = not step.channel or step.channel == "email"

                # Consent and suppression deliberately stay outside this branch
                # and apply to every channel: "do not contact us" is about the
                # contact, not about the transport.

                # Timezone-aware send window. Also checked by the queue sweep,
                # and re-checked here for the same reason suppression is: the
                # sweep is not the only way to reach this task. ``POST
                # /sequences/process`` calls ``send_sequence_step.delay()``
                # directly, and its docstring calls itself "the HTTP-triggered
                # alternative to the process_outreach_queue Celery beat task" —
                # but the window lived only in the sweep, so triggering it at
                # 3 AM on a Sunday sent the mail at 3 AM on a Sunday. A gate that
                # one of two dispatch paths can route around is not a gate; the
                # only place it holds for every caller is here, at the send.
                #
                # Applies to every channel, like the sweep's copy: the window is
                # business hours in the contact's own timezone, and "don't ring
                # them at 3 AM" is at least as true of a phone call.
                #
                # Deferred rather than dropped — the step is still due, just not
                # right now — so the enrollment comes back around after the
                # backoff instead of losing its turn.
                _in_window, _window_reason = get_send_limiter().within_send_window(
                    getattr(contact, "timezone", None)
                )
                if not _in_window:
                    await _defer_enrollment(db, enrollment_id, SOFT_SKIP_BACKOFF)
                    logger.info(
                        "Enrollment %s held: %s", enrollment_id, _window_reason
                    )
                    return {
                        "status": "skipped",
                        "reason": _window_reason,
                        "outside_send_window": True,
                    }

                # Recipient validation — a domain with no MX can't receive mail,
                # so sending guarantees a hard bounce. Validate once (lazily) and
                # pause the enrollment on a definitively invalid domain.
                #
                # Sending capacity is checked HERE, before the recipient
                # validation / Hunter verification / research / AI-composition
                # work below, because all of that is wasted when the mailbox that
                # would carry the send has already hit its warm-up cap. A
                # follow-up step is pinned to the mailbox that started the thread
                # (``enrollment.sender_account_id``) and cannot rotate away from
                # it, so a capped sender means this enrollment is going nowhere
                # until tomorrow — paying an LLM call every 5 minutes to
                # rediscover that is what exhausted the provider quota.
                if is_email_step and enrollment.sender_account_id is not None:
                    from app.services import warmup as _warmup
                    from app.services.email_dispatch import get_account as _get_account

                    _pinned = await _get_account(db, enrollment.sender_account_id)
                    if _pinned is not None:
                        _ok, _cap_reason = await _warmup.check_send_allowed(db, _pinned)
                        if not _ok:
                            await _defer_enrollment(
                                db, enrollment_id, SOFT_SKIP_BACKOFF
                            )
                            logger.info(
                                "Enrollment %s held: pinned sender %s has no capacity (%s)",
                                enrollment_id, getattr(_pinned, "email", "?"), _cap_reason,
                            )
                            return {
                                "status": "skipped",
                                "reason": _cap_reason,
                                "sender_capped": True,
                            }

                # Global daily cap — the cheap read, for the same reason as the
                # warm-up check above. This is the advisory row count, so it does
                # NOT make the send safe (the atomic reservation right before the
                # wire does that); what it does is stop us paying for research,
                # verification and an LLM composition to rediscover, every five
                # minutes, that today's volume is already spent.
                if is_email_step:
                    _pre_ok, _pre_reason = await get_send_limiter().can_send_today(db)
                    if not _pre_ok:
                        await _defer_enrollment(db, enrollment_id, SOFT_SKIP_BACKOFF)
                        logger.info("Enrollment %s held: %s", enrollment_id, _pre_reason)
                        return {
                            "status": "skipped",
                            "reason": _pre_reason,
                            "daily_limit_reached": True,
                        }

                # Account-level burst control. The per-contact gates are keyed on
                # an address and cannot see that info@, sales@ and two named
                # people are one buyer — so four cold emails can land on one
                # company within seconds, which is both the canonical spam
                # pattern and the fastest way to lose the account's goodwill.
                # Deferring (rather than cancelling) staggers the colleagues
                # instead of dropping them, and the wait comes from the gate so a
                # 20-hour gap costs one wake-up rather than twenty.
                if is_email_step:
                    (
                        _co_ok,
                        _co_reason,
                        _co_wait,
                    ) = await get_send_limiter().can_send_to_company(
                        db, contact.company_id, exclude_contact_id=contact.id
                    )
                    if not _co_ok:
                        await _defer_enrollment(
                            db, enrollment_id, max(_co_wait, SOFT_SKIP_BACKOFF)
                        )
                        logger.info(
                            "Enrollment %s held: %s", enrollment_id, _co_reason
                        )
                        return {
                            "status": "skipped",
                            "reason": _co_reason,
                            "company_throttled": True,
                        }

                from app.services.email_validation import validate_contact

                if is_email_step and contact.email_status is None:
                    await validate_contact(db, contact)

                # Mailbox verification (P3) — the MX check above only proves the
                # *domain* accepts mail; it can't tell a real mailbox from one
                # that hard-bounces. Normally app.tasks.verification has already
                # verified this contact within 15 min of import, so this is the
                # backstop for a contact that raced the sweep to its first send.
                # Fail-open by construction: a Hunter outage / missing key writes
                # nothing, so email_status keeps its MX verdict and the send goes
                # ahead — only a definitive 'undeliverable' stops it below.
                from app.config import get_settings
                from app.services.hunter_verification import get_hunter_service

                if (
                    is_email_step
                    and get_settings().hunter_verify_before_send
                    and contact.email_status != "invalid"
                    and contact.email_verified_at is None
                ):
                    await get_hunter_service().verify_contact(db, contact)

                if is_email_step and contact.email_status == "invalid":
                    enrollment.status = "paused"
                    await db.commit()
                    logger.warning(
                        "Enrollment %s paused: %s has an undeliverable domain",
                        enrollment_id, contact.email,
                    )
                    return {"status": "skipped", "reason": "invalid_recipient_domain",
                            "contact_email": contact.email}

                # The gate above proves the domain *can* receive mail. This one
                # asks whether it has ever received ours: several distinct
                # addresses there hard-bounced and nothing at the domain has
                # ever opened, clicked or replied. A folded company keeps paying
                # for DNS and a host that has decided to 550 us still answers,
                # so both look healthy to an MX lookup — the evidence is our own
                # bounce history, not the resolver.
                #
                # ``bounce_handler`` normally suppresses these contacts the
                # moment the domain crosses the line, which is why this is a
                # backstop rather than the mechanism: it catches the contact
                # discovered *after* the write-off, whose address has no
                # suppression entry because it did not exist yet.
                if is_email_step:
                    from app.services import domain_reputation

                    _dead, _dead_reason = await domain_reputation.is_domain_dead(
                        db, domain_reputation.domain_of(contact.email)
                    )
                    if _dead:
                        enrollment.status = "paused"
                        await db.commit()
                        logger.warning(
                            "Enrollment %s paused: %s", enrollment_id, _dead_reason
                        )
                        return {"status": "skipped", "reason": "dead_recipient_domain",
                                "detail": _dead_reason,
                                "contact_email": contact.email}

                # Load company for consent check
                result = await db.execute(
                    select(Company).where(Company.id == contact.company_id)
                )
                company = result.scalar_one_or_none()
                if not company:
                    return {"status": "error", "reason": "company_not_found"}

                # Check consent
                consent_result = await check_consent(db, contact, company)
                if not consent_result.can_email:
                    enrollment.status = "paused"
                    await db.commit()
                    logger.warning(
                        "Cannot email %s: %s", contact.email, consent_result.reason
                    )
                    return {
                        "status": "blocked",
                        "reason": consent_result.reason,
                        "regime": consent_result.regime,
                    }

                # (The step is loaded above, before the email-only gates.)

                # Delay guard: verify enough time has passed since the last
                # send before firing the next step. This prevents premature
                # sends when the task is dispatched by both enroll and the
                # periodic processor (double-dispatch race).
                #
                # Gated on the step's *total* configured wait, not on
                # ``delay_days`` alone — an hours-only step has ``delay_days ==
                # 0``, and the old gate would have skipped the double-dispatch
                # guard entirely for exactly the steps whose whole point is a
                # short, precisely-placed wait.
                from app.services import step_delay

                if enrollment.current_step > 1 and step_delay.base_delay(step) > timedelta(0):
                    now = datetime.now(timezone.utc)

                    # Prefer next_send_at (set by previous step completion)
                    if enrollment.next_send_at is not None:
                        due_at = enrollment.next_send_at
                        if due_at.tzinfo is None:
                            due_at = due_at.replace(tzinfo=timezone.utc)
                        if now < due_at:
                            logger.info(
                                "Enrollment %s step %d not yet due (due %s)",
                                enrollment_id, enrollment.current_step, due_at,
                            )
                            return {
                                "status": "deferred",
                                "reason": f"step not due until {due_at.isoformat()}",
                                "step": enrollment.current_step,
                            }
                    else:
                        # Fallback: check last sent event timestamp
                        last_sent = await db.execute(
                            select(OutreachEvent)
                            .where(
                                and_(
                                    OutreachEvent.enrollment_id == enrollment.id,
                                    OutreachEvent.event_type == "sent",
                                )
                            )
                            .order_by(OutreachEvent.occurred_at.desc())
                            .limit(1)
                        )
                        last_event = last_sent.scalar_one_or_none()
                        if last_event:
                            occurred = last_event.occurred_at
                            if occurred.tzinfo is None:
                                occurred = occurred.replace(tzinfo=timezone.utc)
                            # Same resolver, same factor as the path that would
                            # have written ``next_send_at``. This fallback used
                            # the raw delay while the primary path used the
                            # factored one, so the two disagreed about when a
                            # step was due for any enrollment whose factor had
                            # moved off 1.0.
                            due_at = step_delay.due_at(
                                step,
                                occurred,
                                factor=enrollment.adaptive_delay_factor,
                                enrollment_id=enrollment.id,
                            )
                            if now < due_at:
                                # Set next_send_at for efficiency on next check
                                enrollment.next_send_at = due_at
                                await db.commit()
                                logger.info(
                                    "Enrollment %s step %d not yet due (due %s, from last event)",
                                    enrollment_id, enrollment.current_step, due_at,
                                )
                                return {
                                    "status": "deferred",
                                    "reason": f"step not due until {due_at.isoformat()}",
                                    "step": enrollment.current_step,
                                }

                # --- Adaptive Step Delays (Feature 3): silence back-off ---
                # This step is now due, so the PRIOR step's engagement window has
                # fully elapsed. If that prior step was tracked and drew no
                # open/click/reply, extend the cadence (×1.5) — the enrollment is
                # cold, so back off rather than pile on. Engagement-driven
                # shortening happens live on the open pixel; this handles the
                # silent case, which has no event to hook. Applied once per step
                # (it moves the factor used to schedule the NEXT send).
                if enrollment.current_step > 1:
                    from app.services import adaptive_delay

                    prior_eng = await _prior_step_engagement(
                        db, enrollment, enrollment.current_step - 1
                    )
                    if (
                        prior_eng["prior_tracked"]
                        and not prior_eng["opened"]
                        and not prior_eng["replied"]
                    ):
                        adaptive_delay.register_silence(enrollment)

                # --- Non-email channel step ---
                # A "call"/"linkedin"/"video"/"direct_mail" step isn't an email:
                # it's a human touch. File the task and HOLD the cadence on this
                # step until somebody does it (or the hold expires), rather than
                # advancing on the filing of the reminder.
                #
                # Holding is what makes the spacing mean anything. "Call, then
                # email two days later" used to send that email two days after
                # the reminder was created — so a call made on Thursday was
                # followed up on Tuesday, referring to a conversation that had
                # not happened. See ``services.manual_touch``.
                if step.channel and step.channel != "email":
                    from app.services import manual_touch

                    existing = await manual_touch.open_task_for(
                        db, enrollment, step.step_number
                    )
                    if existing is None:
                        follow_up, deadline = await manual_touch.create_task(
                            db, enrollment, step, contact.id, company.id if company else None
                        )
                        await db.commit()
                        return {
                            "status": "channel_task_created",
                            "channel": step.channel,
                            "step": step.step_number,
                            "follow_up_id": str(follow_up.id),
                            "expires_at": deadline.isoformat(),
                        }

                    # Already filed — this dispatch is the deadline waking us up
                    # (or a completion the API hook could not action). Resolve it.
                    outcome = await manual_touch.resolve_task(db, enrollment, step)
                    await db.commit()
                    return {
                        "status": f"channel_task_{outcome['status']}",
                        "channel": step.channel,
                        "step": step.step_number,
                    }

                # --- Conditional branching ---
                # Honour the step's send_condition against the prior step's
                # engagement. When the condition isn't met, advance past the
                # step without sending.
                should_send, cond_reason = await evaluate_step_condition(db, enrollment, step)
                if not should_send:
                    await _advance_step(db, enrollment)
                    await db.commit()
                    logger.info(
                        "Condition '%s' not met for enrollment %s step %d: %s",
                        step.send_condition, enrollment_id, step.step_number, cond_reason,
                    )
                    return {
                        "status": "skipped_by_condition",
                        "condition": step.send_condition,
                        "reason": cond_reason,
                        "step": step.step_number,
                    }

                # Build template context (includes demo_link resolved by industry)
                template_context = build_merge_context(contact, company)
                # Keep backward compat fields used by older sequences
                template_context["company_domain"] = company.domain or ""
                template_context["company_industry"] = company.industry_segment or ""

                # --- Calendar Booking Injection (Feature 4) ---
                # Resolve {{booking_link}} to the sender's booking URL only when
                # the interest gate is met (or the step opts out of the gate via
                # ``always_inject``). Otherwise it renders empty and its
                # scaffolding line is stripped, so we never send a dangling
                # "Grab a time here:" with no link.
                from app.services.booking import (
                    apply_booking_placeholder,
                    resolve_booking_url,
                    should_inject_booking_link,
                )

                always_inject = bool(getattr(step, "always_inject", False))
                booking_url = ""
                if always_inject or await should_inject_booking_link(contact, enrollment, db):
                    booking_url = await resolve_booking_url(db, enrollment.sender_account_id)
                template_context["booking_link"] = booking_url

                # --- A/B testing (Feature 3) ---
                # Resolve which copy variant to send BEFORE rendering. For a step
                # with an active A/B test (≥2 active variants) this assigns a
                # variant and persists the assignment before the send (crash
                # safety); for every other step it returns the step's own
                # template, i.e. the exact current behaviour (backwards compat).
                from app.services.ab_test_resolver import (
                    backfill_assignment_send,
                    resolve_variant,
                )

                resolution = await resolve_variant(db, enrollment, step)

                # Send email via the active connected Gmail account (falls back
                # to SES when no account is connected). Render the chosen variant
                # copy, with the booking placeholder applied to it; the dispatcher
                # appends the CAN-SPAM footer.
                subject = clean_subject(
                    render(
                        apply_booking_placeholder(resolution.subject_template, booking_url),
                        subject_context(template_context),
                    )
                )
                body = clean_body(
                    render(
                        apply_booking_placeholder(resolution.body_template, booking_url),
                        template_context,
                    )
                )

                # --- Spintax (manual copy variation) ---
                # Resolve {a|b|c} groups down to one variant for THIS send, and
                # record which alternatives were chosen so per-variant response
                # can be attributed later. Runs after the Jinja merge so merge
                # fields inside a spintax option are already substituted, and
                # before AI composition (which, when it fires, supersedes the
                # static copy entirely).
                from app.services import spintax

                _subj_spin = spintax.resolve_tracked(subject)
                _body_spin = spintax.resolve_tracked(body)
                subject = _subj_spin.text
                body = _body_spin.text
                spintax_meta = None
                if _subj_spin.has_choices or _body_spin.has_choices:
                    spintax_meta = {
                        "subject_signature": _subj_spin.signature,
                        "body_signature": _body_spin.signature,
                        "subject_choices": _subj_spin.choices,
                        "body_choices": _body_spin.choices,
                        "variant_key": f"s:{_subj_spin.signature}|b:{_body_spin.signature}",
                    }

                # --- AI composition (first touch) ---
                # Replace the static mail-merge with copy grounded in the
                # company's research brief. Fail-safe: any problem returns the
                # static render untouched. On a first batch per sequence the copy
                # is queued for a human instead of sent — cold outreach is
                # irreversible and charged against the sending domain.
                # A sequence marked ``send_as_written`` carries copy a person
                # already reviewed and approved word for word (the Discovery
                # document flow): neither the composer nor the optimizer may
                # rewrite it, and it must not be parked for a second review.
                send_as_written = bool(getattr(sequence, "send_as_written", False))
                from app.services.ai_compose import ComposeResult, compose_step

                if send_as_written:
                    composed = ComposeResult("send", subject, body, reason="send as written")
                else:
                    composed = await compose_step(
                        db, enrollment, contact, company, step, subject, body
                    )
                if composed.action == "review":
                    # Park the enrollment. Deliberately NOT _advance_step: the
                    # step has not been sent, and advancing would silently skip
                    # the touch a reviewer is about to approve.
                    enrollment.status = "pending_review"
                    await db.commit()
                    logger.info(
                        "AI draft %s queued for review — enrollment %s step %d parked (%s)",
                        composed.draft_id, enrollment_id, step.step_number, composed.reason,
                    )
                    return {
                        "status": "pending_review",
                        "step": step.step_number,
                        "draft_id": str(composed.draft_id) if composed.draft_id else None,
                        "reason": composed.reason,
                    }
                # ``clean_subject`` again: the static render was cleaned before
                # composition, but an AI subject is a *different* string that
                # never passed through it. The model is prompted with the static
                # copy as an exemplar, so it can echo a leading "{{first_name},"
                # shape and emit ", who eats it when a part runs out of tol?" for
                # a contact with no first name. Idempotent on an already-clean
                # subject, so this only ever removes the artifact.
                # ``clean_body`` for the same reason: an AI body is a different
                # string too, and echoing the exemplar's "Hi {{first_name}},"
                # shape for a nameless contact yields a literal "Hi ,".
                subject, body = clean_subject(composed.subject), clean_body(composed.body)
                if composed.ai_generated:
                    logger.info(
                        "AI-composed step %d for enrollment %s (research=%s)",
                        step.step_number, enrollment_id, composed.research_used,
                    )

                # --- AI-Optimized Sequence Engine (feature 1.2) ---
                # Ask the optimizer what to do with this step before sending. It
                # may adjust delay, rewrite content, switch channel, or skip. The
                # optimizer is fail-safe: it returns a plain "send" on any error
                # or when disabled, so this never blocks a send.
                optimizer = get_sequence_optimizer()
                if send_as_written:
                    decision = {"action": "send", "reasoning": "sequence sends as written"}
                else:
                    decision = await optimizer.decide(
                        db, enrollment, contact, company, step, subject, body
                    )
                record_decision(enrollment, enrollment.current_step, decision)
                action = decision.get("action", "send")

                if action == "skip":
                    # They're in a live conversation — don't pile on a scheduled
                    # touch. Advance past this step without sending.
                    await _advance_step(db, enrollment)
                    await db.commit()
                    logger.info(
                        "Optimizer SKIP for enrollment %s step %d: %s",
                        enrollment_id, step.step_number, decision.get("reasoning"),
                    )
                    return {
                        "status": "skipped_by_optimizer",
                        "step": step.step_number,
                        "reasoning": decision.get("reasoning"),
                    }

                if action == "delay":
                    delay_days = decision.get("delay_days", 1)
                    enrollment.next_send_at = datetime.now(timezone.utc) + timedelta(days=delay_days)
                    await db.commit()
                    logger.info(
                        "Optimizer DELAY %dd for enrollment %s step %d: %s",
                        delay_days, enrollment_id, step.step_number, decision.get("reasoning"),
                    )
                    return {
                        "status": "delayed_by_optimizer",
                        "step": step.step_number,
                        "delay_days": delay_days,
                        "reasoning": decision.get("reasoning"),
                    }

                if action == "switch_channel":
                    # Hold the email and surface a human task to reach out on the
                    # recommended channel. Advance the step so the email sequence
                    # continues from the next touch.
                    channel = decision.get("channel", "call")
                    reminder = FollowUp(
                        contact_id=contact.id,
                        company_id=company.id,
                        due_date=datetime.now(timezone.utc),
                        reason=f"AI: switch to {channel} — {decision.get('reasoning', '')}"[:500],
                        status="pending",
                    )
                    db.add(reminder)
                    await _advance_step(db, enrollment)
                    await db.commit()
                    logger.info(
                        "Optimizer SWITCH_CHANNEL→%s for enrollment %s step %d: %s",
                        channel, enrollment_id, step.step_number, decision.get("reasoning"),
                    )
                    return {
                        "status": "channel_switch_by_optimizer",
                        "step": step.step_number,
                        "channel": channel,
                        "reasoning": decision.get("reasoning"),
                    }

                if action == "rewrite":
                    # Send now, but with the AI-adapted copy.
                    if decision.get("subject"):
                        subject = decision["subject"]
                    if decision.get("body"):
                        body = decision["body"]
                    logger.info(
                        "Optimizer REWRITE for enrollment %s step %d: %s",
                        enrollment_id, step.step_number, decision.get("reasoning"),
                    )
                # action == "send" (or anything else): fall through and send as-is.

                # Thread-linking: for steps after the first, chain into the
                # same Gmail thread so the prospect sees a single conversation.
                thread_id = None
                in_reply_to = None
                references = None
                send_account_id = None

                if enrollment.current_step > 1 and enrollment.gmail_thread_id:
                    thread_id = enrollment.gmail_thread_id
                    if enrollment.gmail_message_id:
                        # Gmail message IDs need angle brackets for RFC 2822
                        msg_ref = f"<{enrollment.gmail_message_id}@mail.gmail.com>"
                        in_reply_to = msg_ref
                        references = msg_ref
                    # Prefix "Re: " to match Gmail threading convention
                    if not subject.lower().startswith("re:"):
                        subject = f"Re: {subject}"

                if enrollment.sender_account_id:
                    send_account_id = enrollment.sender_account_id

                # --- Engagement tracking ---
                # Instrument the final copy (after any optimizer rewrite) so the
                # body that actually ships is the body we measure. Keyed by
                # (enrollment, step), which is this path's send identity — there
                # is no EmailSend row here. Applied last so nothing downstream
                # rewrites the pixel or the wrapped links.
                from app.config import get_settings as _get_settings
                from app.services.email_tracking import apply_tracking, should_track_step

                # The human copy, before the open-pixel / link-wrapping is baked
                # in. Stored on the event so the operator can read back exactly
                # what Dialtone said, without the tracking artifacts.
                human_body = body

                # Drop tracking on the first touch. An open pixel and rewrapped
                # links are a bulk-mail tell, and they hurt most on the cold
                # first email to an address with no prior relationship — exactly
                # where deliverability decides whether the sequence ever gets a
                # reply. Steps 2+ stay tracked, so the engagement signal (and the
                # send_condition / optimizer / A/B stats that read it) is only
                # deferred one step, not lost. ``track_first_step`` overrides.
                tracking_applied = should_track_step(
                    enrollment.current_step, _get_settings()
                )
                body = apply_tracking(
                    body, str(enrollment.id), enrollment.current_step,
                    enabled=tracking_applied,
                )

                # --- Pre-send spam gate ---
                # Score the copy that is actually about to ship — after the
                # merge, after spintax resolved, after any AI rewrite, but before
                # the tracking pixel (which is ours, not the author's, and would
                # otherwise show up as "another link"). A flag rides along on the
                # send event; a block soft-skips so the enrollment is held for a
                # copy edit rather than spent on a message filters would eat.
                from app.services import spam_guard

                _spam_ok, _spam_verdict = spam_guard.scan_for_send(subject, human_body)
                if not _spam_ok:
                    _spam_reason = _spam_verdict.reason()
                    await db.rollback()
                    await _defer_enrollment(db, enrollment_id, SOFT_SKIP_BACKOFF)
                    logger.error(
                        "Enrollment %s blocked by spam gate: %s",
                        enrollment_id, _spam_reason,
                    )
                    return {
                        "status": "skipped",
                        "reason": _spam_reason,
                        "spam_blocked": True,
                        "spam_scan": _spam_verdict.as_metadata(),
                    }

                # --- Per-mailbox pacing: no mailbox bursts. ---
                # The warm-up ramp says how many a mailbox may send today; this
                # says how fast. The queue sweep dispatches every due enrollment
                # in one loop, so without this a mailbox puts its whole backlog
                # on the wire in seconds — the burst shape bulk-send detection
                # looks for, at a daily volume that passes every warm-up check.
                #
                # Resolve the sender *first* when the step isn't pinned to one:
                # a first touch has no ``sender_account_id`` (rotation picks at
                # dispatch), and pacing a mailbox we can't name paces nothing —
                # which would leave the highest-volume path in the system, cold
                # first touches, completely unpaced. Resolving here also pins the
                # send to the mailbox we paced, so the reservation and the send
                # can't disagree about which inbox this went out from.
                from app.services import send_pacing

                if send_account_id is None:
                    from app.services.email_dispatch import resolve_sender_info

                    send_account_id = (await resolve_sender_info(db)).account_id

                _pace = await send_pacing.claim_send_slot(send_account_id)
                if not _pace.allowed:
                    await db.rollback()
                    # Wait out the mailbox's window, plus a small cushion so the
                    # retry doesn't land in the same second the slot frees and
                    # lose the race to another worker.
                    await _defer_enrollment(
                        db, enrollment_id, timedelta(seconds=_pace.wait_seconds + 15)
                    )
                    logger.info(
                        "Enrollment %s paced: %s", enrollment_id, _pace.reason
                    )
                    return {"status": "skipped", "reason": _pace.reason, "paced": True}

                # --- Daily cap: claim the slot atomically, right before the wire.
                # The queue sweep's ``can_send_today`` is advisory — it counts
                # rows, and every concurrent worker reads the same pre-send count,
                # so a burst of dispatches all pass a cap they collectively blow
                # through. This INCR is per-intended-send, so exactly `limit`
                # workers get a slot no matter how many ask at once. The slot is
                # handed back below on any outcome that isn't a real send.
                _limiter = get_send_limiter()
                _reserved, _reserve_reason = await _limiter.reserve_daily_send(db)
                if not _reserved:
                    # Nothing went out, so the mailbox shouldn't sit out its
                    # spacing window on account of a cap it never touched.
                    await send_pacing.release_send_slot(send_account_id)
                    await db.rollback()
                    await _defer_enrollment(db, enrollment_id, SOFT_SKIP_BACKOFF)
                    logger.warning(
                        "Enrollment %s held: %s", enrollment_id, _reserve_reason
                    )
                    return {"status": "skipped", "reason": _reserve_reason,
                            "daily_limit_reached": True}

                # --- At-most-once: claim this step for the wire.
                #
                # Everything this task is about to write — the event, the
                # advanced step, the thread ids — commits *after* Gmail already
                # has the message. A failure anywhere in that window (a dropped
                # COMMIT, a worker killed, bookkeeping that raises) rolls the
                # record back and leaves the mail, and the Celery retry finds an
                # enrollment that still reads "step N, active, due" and sends the
                # same cold email to the same person again. Every other gate here
                # asks the database, which in that window has not been told.
                #
                # The claim *commits*, which is the whole point: a committed fact
                # outlives the rollback of the transaction that loses the send.
                # It is also what separates two workers racing on one enrollment
                # — the conditional UPDATE's row lock decides, not the order they
                # reached the dispatcher.
                #
                # On *this* session, deliberately, not a second one.
                # ``record_decision`` dirties ``enrollment.optimization_context``
                # earlier in this transaction, and the next autoflush turns that
                # into an uncommitted UPDATE holding a row lock on this very row.
                # A claim issued from another connection would wait on that lock
                # while this task waited on the claim — a worker wedged until the
                # statement timeout, on every send. Committing here also lands
                # the pre-send work (the decision log, the validated address)
                # that was going to be committed a moment later anyway.
                _claim_step = step.step_number
                _claim = await send_claim.acquire(db, enrollment_id, _claim_step)
                if not _claim.acquired:
                    # Nothing of ours goes out, so both reservations go back.
                    await _limiter.release_daily_send()
                    await send_pacing.release_send_slot(send_account_id)
                    await db.rollback()
                    return await _handle_duplicate_claim(
                        db, enrollment_id, _claim_step, _claim
                    )

                try:
                    send_result = await send_outreach_email(
                        db,
                        to_email=contact.email,
                        subject=subject,
                        body_html=body,
                        sender_account_id=send_account_id,
                        thread_id=thread_id,
                        in_reply_to=in_reply_to,
                        references=references,
                        # Already scanned above, on the pre-tracking body.
                        skip_spam_scan=True,
                        # And already reserved above. The dispatcher claims the
                        # daily slot for every other caller; this path has to
                        # claim earlier, ahead of the send claim and the pacing
                        # window, so all three release together.
                        daily_slot_reserved=True,
                    )
                except Exception:
                    # The dispatcher blew up — nothing went out, so the slot must
                    # not stay spent or a flapping SMTP host silently eats the
                    # day's allowance. Same for the pacing window: a transient
                    # SMTP failure must not enforce a minute of silence on every
                    # enrollment queued behind this one.
                    #
                    # The send *claim* is deliberately not released here. Those
                    # two counters are cheap to be wrong about; this is not.
                    # ``send_outreach_email`` does real work after the transport
                    # returns — the per-account daily counter, the warm-up start
                    # — so an exception out of it does not prove the message
                    # stayed home. Handing the claim back on a maybe is how one
                    # uncertain send becomes two certain ones.
                    await _limiter.release_daily_send()
                    await send_pacing.release_send_slot(send_account_id)
                    raise

                # A send that didn't happen (soft skip) or bounced outright never
                # consumed real sending volume; return its slot.
                if send_result.get("status") != "sent":
                    await _limiter.release_daily_send()
                    # Pacing, unlike the daily cap, is released only for a *soft
                    # skip* — the dispatcher declined before the wire. A bounce
                    # means we did open a connection and a server did reject us,
                    # which is exactly the moment to keep slowing down rather
                    # than let the next message follow immediately.
                    if send_result.get("status") in ("skipped", "suppressed", "halted"):
                        await send_pacing.release_send_slot(send_account_id)

                # Soft-skip statuses: the dispatcher declined to send but this is
                # NOT a bounce (warm-up cap hit, suppression, kill switch). Record
                # nothing and DON'T advance — the step hasn't been sent, so the
                # queue will retry it once the block clears. Recording a bounce
                # here would falsely dent sender metrics and end the enrollment.
                if send_result.get("status") in ("skipped", "suppressed", "halted"):
                    # Read every ORM attribute we still need BEFORE rolling back.
                    # ``rollback()`` expires all instances in the session, and an
                    # expired attribute read on an AsyncSession attempts sync IO
                    # to refresh itself, which raises MissingGreenlet. That turned
                    # this benign soft-skip into a task crash → Celery retry →
                    # re-dispatch, so a single warm-up cap wedged the whole queue.
                    skip_step = step.step_number
                    skip_status = send_result.get("status")
                    skip_reason = send_result.get("reason")
                    await db.rollback()

                    # Hand the claim back. Unlike the exception path above, this
                    # is not a maybe: the dispatcher returned a status saying it
                    # declined, so the step is genuinely unsent and the retry
                    # this defers to has to be allowed to make it.
                    await send_claim.release(db, enrollment_id, skip_step)

                    # Back the enrollment off so the 5-minute queue sweep stops
                    # re-dispatching a send that is going to be declined again.
                    await _defer_enrollment(db, enrollment_id, SOFT_SKIP_BACKOFF)

                    logger.info(
                        "Step %d for enrollment %s not sent (%s): %s — retrying after %s",
                        skip_step, enrollment_id, skip_status, skip_reason,
                        SOFT_SKIP_BACKOFF,
                    )
                    return {
                        "status": skip_status,
                        "reason": skip_reason,
                        "step": skip_step,
                        "deferred_until": (
                            datetime.now(timezone.utc) + SOFT_SKIP_BACKOFF
                        ).isoformat(),
                    }

                # Record event. ``tracked`` records whether this send carried an
                # open pixel — ``evaluate_step_condition`` reads it to decide
                # whether a missing open means "not opened" or "we never looked".
                event_type = "sent" if send_result["status"] == "sent" else "bounced"
                event = OutreachEvent(
                    enrollment_id=enrollment.id,
                    step_number=enrollment.current_step,
                    event_type=event_type,
                    event_metadata={
                        **send_result,
                        "tracked": bool(tracking_applied),
                        # Persist what actually went out. The sequence path has
                        # no EmailSend row, so without this the sent copy lives
                        # only in the sender's Gmail — the Console feed can show
                        # who/when but never *what*. These make the send readable.
                        "to_email": contact.email,
                        "subject": subject,
                        "body_html": human_body,
                        # Whether this send was the AI-personalized copy or the
                        # static sequence template, and why. Without this, "AI
                        # composition is configured and enabled" and "AI
                        # composition is actually shipping" are indistinguishable
                        # from the outside — every metric the sequence reports
                        # (sent, opened, reply rate) looks identical either way.
                        "ai_generated": bool(composed.ai_generated),
                        "compose_reason": composed.reason,
                        "research_used": bool(composed.research_used),
                        # The prospect data points the copy was built from
                        # (``app.services.personalization``). Recorded on the
                        # send, not just the draft, because most sends never
                        # become a draft — the review gate is a first-N sample —
                        # and this is what lets campaign analytics ask whether
                        # depth of personalization moves the reply rate.
                        **(
                            {"personalization": composed.personalization}
                            if composed.personalization
                            else {}
                        ),
                        # Which spintax variant shipped (when the copy used any),
                        # so downstream engagement can be attributed per variant.
                        **({"spintax": spintax_meta} if spintax_meta else {}),
                        # Only recorded when the scan actually had something to
                        # say — a score of 0 on every clean send is noise, and
                        # this metadata is read by hand during incidents.
                        **(
                            {"spam_scan": _spam_verdict.as_metadata()}
                            if _spam_verdict.flagged
                            else {}
                        ),
                    },
                )
                db.add(event)

                if event_type == "sent":
                    # Close out an approved draft so it can't be re-proposed and
                    # so the review queue reflects what actually shipped.
                    if composed.draft_id is not None:
                        from app.models.ai_email_draft import AiEmailDraft as _Draft

                        draft = (
                            await db.execute(
                                select(_Draft).where(_Draft.id == composed.draft_id)
                            )
                        ).scalar_one_or_none()
                        if draft is not None and draft.status == "approved":
                            draft.status = "sent"
                            draft.send_result = send_result

                    # A/B backfill: link the assignment to its send. The sequence
                    # path records no EmailSend row, so this is a no-op here
                    # (send_result carries none) and per-variant metrics attribute
                    # through the assignment's enrollment + step number instead.
                    if resolution.assignment_id is not None:
                        await backfill_assignment_send(
                            db, resolution.assignment_id, send_result.get("email_send_id")
                        )

                    # Store threading info for subsequent steps
                    if send_result.get("gmail_thread_id"):
                        enrollment.gmail_thread_id = send_result["gmail_thread_id"]
                    if send_result.get("gmail_message_id"):
                        enrollment.gmail_message_id = send_result["gmail_message_id"]
                    if send_result.get("sender_account_id"):
                        from uuid import UUID as _UUID
                        enrollment.sender_account_id = _UUID(send_result["sender_account_id"])

                    # Advance to next step and calculate next_send_at
                    enrollment.current_step += 1

                    # Look ahead to see if there's a next step and set next_send_at
                    next_step_result = await db.execute(
                        select(SequenceStep).where(
                            and_(
                                SequenceStep.sequence_id == enrollment.sequence_id,
                                SequenceStep.step_number == enrollment.current_step,
                            )
                        )
                    )
                    next_step = next_step_result.scalar_one_or_none()
                    if next_step:
                        # ``services.step_delay`` resolves the whole cadence:
                        # days + hours, business-day counting, the enrollment's
                        # adaptive factor (clamped), and the stable jitter.
                        from app.services import step_delay

                        enrollment.next_send_at = step_delay.due_at(
                            next_step,
                            datetime.now(timezone.utc),
                            factor=enrollment.adaptive_delay_factor,
                            enrollment_id=enrollment.id,
                        )
                    else:
                        enrollment.status = "completed"
                        enrollment.completed_at = datetime.now(timezone.utc)
                elif event_type == "bounced":
                    enrollment.status = "bounced"

                # A send that reached the wire clears the failure state. Done in
                # this transaction so the reset and the send it describes commit
                # together — a counter cleared by a send that then rolled back
                # would forgive a failure that still happened.
                from app.services import send_failures

                send_failures.clear_failures(enrollment)

                await db.commit()
                logger.info(
                    "Sent step %d to %s (thread=%s) — result: %s",
                    step.step_number, contact.email,
                    enrollment.gmail_thread_id, send_result["status"],
                )
                return {
                    "status": event_type,
                    "contact_email": contact.email,
                    "step": step.step_number,
                    "send_result": send_result,
                    "gmail_thread_id": enrollment.gmail_thread_id,
                }

            except Exception as e:
                await db.rollback()
                raise

    # Decided out here, where ``self`` is: Celery's retry counter lives on the
    # request, and the attempt below needs to know whether this is the last one.
    retries_left = self.request.retries < self.max_retries

    async def _attempt():
        """One send, plus what to do with a failure that has run out of retries."""
        try:
            return await _run()
        except Exception as exc:
            if retries_left:
                # Let it out to the sync wrapper, which turns it into a Celery
                # retry. Most faults are transient and clear inside the chain,
                # and recording every attempt would make ``consecutive_failures``
                # count attempts rather than incidents.
                raise

            # Out of retries: this is one failed *incident*, and the enrollment
            # has to remember it. Nothing did until now, so the row was left
            # ``active`` with a due date in the past and the queue sweep re-sent
            # it every five minutes for as long as the fault lasted.
            #
            # On its own session: the send's session died with the transaction
            # that failed, and the whole point is that this record outlives it.
            from app.services import send_failures

            async with celery_session() as db:
                outcome = await send_failures.record_failure(db, enrollment_id, exc)

            # Returned rather than re-raised. The failure is now durably on the
            # enrollment and reported by ``GET /dialtone/failures``; re-raising
            # would add a Celery traceback nobody reads and mark a task failed
            # that has, by this point, been handled.
            return {
                "status": "failed",
                "enrollment_id": enrollment_id,
                "error": str(exc)[:500],
                "consecutive_failures": (outcome or {}).get("consecutive_failures"),
                "quarantined": (outcome or {}).get("quarantined", False),
            }

    try:
        return _run_async(_attempt())
    except Exception as exc:
        logger.error("Outreach step failed for enrollment %s: %s", enrollment_id, str(exc))
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


# An org that pauses sending stays paused for days, and the queue sweep runs
# every five minutes: a WARNING per pass buried every other warning in the
# worker log. Once an hour per org is enough to know it is still engaged.
KILL_SWITCH_LOG_INTERVAL_SECONDS = 3600
_kill_switch_last_logged: dict = {}


def _log_kill_switch_halt(now: float | None = None) -> bool:
    """Log the halted-queue warning at most once an hour per org. True if logged."""
    import time

    from app.org_context import get_current_org_id

    org_id = get_current_org_id()
    now = time.monotonic() if now is None else now
    last = _kill_switch_last_logged.get(org_id)
    if last is not None and now - last < KILL_SWITCH_LOG_INTERVAL_SECONDS:
        logger.debug("Dialtone paused for org %s (kill switch still engaged)", org_id)
        return False
    _kill_switch_last_logged[org_id] = now
    logger.warning(
        "Dialtone paused: outreach queue halted for org %s — sending kill switch engaged "
        "(repeats suppressed for %d min)", org_id, KILL_SWITCH_LOG_INTERVAL_SECONDS // 60,
    )
    return True


@celery_app.task(name="app.tasks.outreach.process_outreach_queue")
def process_outreach_queue():
    """Periodic task: find all active enrollments due for their next step and dispatch sends.

    Runs every 5 minutes via Celery Beat. Checks delay_days on each step
    against the last sent event to determine if the next step is due.
    """
    logger.info("Processing outreach queue")

    async def _run():
        # Per-org: enrollments/contacts are org-scoped and each org has its own
        # kill switch, sending window and send limits.
        async def _body(db):
            # Global kill switch pre-check — when engaged, dispatch nothing.
            from app.services import system_settings

            if not await system_settings.is_sending_enabled(db):
                _log_kill_switch_halt()
                return {"status": "halted", "reason": "sending kill switch engaged", "dispatched": 0}

            # Enrollments of *live* campaigns only. A paused sequence
            # (``PATCH /sequences/{id}/status`` → ``is_active=False``) is the
            # operator saying "stop sending this", and until this join existed
            # the sweep went on dispatching every one of its enrollments —
            # steps 2..N kept arriving from a campaign the console showed as
            # paused. Filtered in SQL rather than in the loop below so a paused
            # campaign costs the sweep nothing: an org that shelved a 20k-contact
            # blast should not pay to rank and re-skip it every five minutes.
            result = await db.execute(
                select(OutreachEnrollment)
                .join(
                    OutreachSequence,
                    OutreachEnrollment.sequence_id == OutreachSequence.id,
                )
                .where(
                    OutreachEnrollment.status == "active",
                    OutreachSequence.is_active.is_(True),
                )
            )
            enrollments = list(result.scalars().all())

            # Capacity is finite and this sweep spends it in list order: the
            # daily quota latches off partway through, per-contact limits ration
            # the rest. Unordered, that spends it on whoever the planner happened
            # to return first — and a sequential scan returns the *same* rows
            # first tomorrow, so anything past the cap line is starved rather
            # than delayed. Rank decides only who reaches the gates below first;
            # every gate still runs, on every enrollment.
            from app.services import queue_priority

            now = datetime.now(timezone.utc)
            enrollments = await queue_priority.rank(db, enrollments, now=now)

            dispatched = 0
            skipped_window = 0
            scheduled_smart = 0
            # Enrollments that were due and lost the day to a spent quota. The
            # count is the demand signal behind the ranking — "30 sent, 84 asked"
            # is what says the cap is the constraint.
            skipped_quota = 0
            # Enrollments whose send task from an earlier sweep is still in the
            # broker. Reported because a number that climbs sweep after sweep is
            # the readable form of "the worker is not keeping up" — which used to
            # surface only as a mysteriously large LLM bill.
            skipped_in_flight = 0
            # Enrollments this sweep closed because their contact has opted out.
            # Reported rather than silent: it is the count that says the queue
            # shrank for a compliance reason and not because sending stalled.
            stopped_suppressed = 0

            # Anti-spam: don't send outside business hours / on weekends. Cold
            # mail that lands at 3 AM reads as a bot and tanks reply rates. The
            # window is evaluated per contact in THEIR timezone when known (see
            # the per-enrollment check below), so a lead in Los Angeles isn't
            # emailed at 5 AM their time even when the global window is open.
            limiter = get_send_limiter()

            # Channel of the step each enrollment is currently on, in one query
            # for the whole sweep.
            #
            # Needed because most of the gates below are about email — the daily
            # send cap, the recipient's inbox window, per-contact send limits,
            # send-time optimization — and a "call"/"linkedin" step is none of
            # those. Without this the daily email cap ``break``s out of the sweep
            # and a call task that was waiting to expire sits blocked behind an
            # inbox quota it does not use.
            #
            # Batched rather than read per enrollment: the fast path (a set
            # ``next_send_at``) deliberately avoids loading the step at all, and
            # reintroducing a per-enrollment query to read one column would undo
            # that for every email enrollment to serve the few manual ones.
            step_channels: dict[tuple, str] = {}
            if enrollments:
                channel_rows = (
                    await db.execute(
                        select(
                            SequenceStep.sequence_id,
                            SequenceStep.step_number,
                            SequenceStep.channel,
                        ).where(
                            SequenceStep.sequence_id.in_(
                                {e.sequence_id for e in enrollments}
                            )
                        )
                    )
                ).all()
                step_channels = {
                    (seq_id, num): channel for seq_id, num, channel in channel_rows
                }

            # Latched once the org's daily email quota is found spent, so the
            # remaining email enrollments are skipped without re-asking.
            email_quota_spent: Optional[str] = None
            # A dispatch is only useful if the task it queues will actually run
            # before the next sweep. Nothing recorded that a task had been queued,
            # so an enrollment whose send was still sitting in the broker was due
            # again five minutes later and dispatched again — and again. A worker
            # half an hour behind was handed six copies of every task in its
            # backlog, each paying for contact research and an LLM composition
            # before the send claim refused it. The lease expires so a task lost
            # with its broker is re-dispatched rather than stranded forever.
            #
            # Time-based only, never explicitly cleared. Every path out of the
            # send task either moves ``next_send_at`` well past the lease (a
            # send schedules the next step days out, a soft skip defers an hour,
            # a failure backs off) or ends the enrollment, so the lease costs
            # nothing on the paths that matter and one expiry on the few it
            # doesn't.
            from app.config import get_settings as _get_settings

            dispatch_lease = timedelta(
                minutes=max(getattr(_get_settings(), "dispatch_lease_minutes", 15), 6)
            )
            for enrollment in enrollments:
                leased_at = enrollment.dispatched_at
                if leased_at is not None:
                    if leased_at.tzinfo is None:
                        leased_at = leased_at.replace(tzinfo=timezone.utc)
                    if now - leased_at < dispatch_lease:
                        skipped_in_flight += 1
                        logger.debug(
                            "Enrollment %s already dispatched at %s — not queuing again",
                            enrollment.id, leased_at.isoformat(),
                        )
                        continue

                # A step with no row / no channel is an email step: that is the
                # column's default and what every pre-multi-channel step is.
                is_email_step = step_channels.get(
                    (enrollment.sequence_id, enrollment.current_step), "email"
                ) in (None, "email")
                # Fast path: if next_send_at is set, use it directly
                if enrollment.next_send_at is not None:
                    next_at = enrollment.next_send_at
                    if next_at.tzinfo is None:
                        next_at = next_at.replace(tzinfo=timezone.utc)
                    if now < next_at:
                        continue  # Not yet due
                    due_at = next_at
                else:
                    # Legacy path: check delay from last event
                    last_event_result = await db.execute(
                        select(OutreachEvent)
                        .where(
                            and_(
                                OutreachEvent.enrollment_id == enrollment.id,
                                OutreachEvent.event_type == "sent",
                            )
                        )
                        .order_by(OutreachEvent.occurred_at.desc())
                        .limit(1)
                    )
                    last_event = last_event_result.scalar_one_or_none()

                    # Load current step to check delay
                    step_result = await db.execute(
                        select(SequenceStep).where(
                            and_(
                                SequenceStep.sequence_id == enrollment.sequence_id,
                                SequenceStep.step_number == enrollment.current_step,
                            )
                        )
                    )
                    step = step_result.scalar_one_or_none()

                    if not step:
                        enrollment.status = "completed"
                        enrollment.completed_at = now
                        continue

                    # Check delay. Anchored on the last send, or on the
                    # enrollment for a first touch. Resolved through
                    # ``services.step_delay`` with the enrollment's adaptive
                    # factor, so this sweep and the send task it dispatches
                    # cannot disagree about whether a step is due — they used
                    # different arithmetic before, and an enrollment whose factor
                    # had extended its wait would be picked up here and then
                    # deferred right back by the task.
                    from app.services import step_delay

                    anchor = last_event.occurred_at if last_event else enrollment.enrolled_at
                    due_now, due_date = step_delay.is_due(
                        step,
                        anchor,
                        now=now,
                        factor=enrollment.adaptive_delay_factor,
                        enrollment_id=enrollment.id,
                    )
                    if not due_now:
                        continue
                    due_at = due_date

                # --- Anti-spam checks ---
                # Check daily limit — if hit, stop dispatching *email*, but keep
                # walking the queue.
                #
                # This used to ``break``. The daily cap counts messages put on
                # the wire, and a call task has not put one there, so abandoning
                # the sweep on a spent email quota leaves every held manual step
                # unresolved until the quota resets — a cadence stalled by a
                # limit that does not apply to it. The answer is latched instead
                # of re-queried, so a capped day still costs exactly one check.
                if is_email_step:
                    if email_quota_spent is None:
                        can_send, reason = await limiter.can_send_today(db)
                        if not can_send:
                            email_quota_spent = reason
                            logger.warning(
                                "Daily send limit reached: %s — no further email "
                                "dispatch this sweep", reason,
                            )
                    if email_quota_spent is not None:
                        skipped_quota += 1
                        continue

                # Load contact to check per-contact limits + timezone window
                contact_result = await db.execute(
                    select(Contact).where(Contact.id == enrollment.contact_id)
                )
                contact = contact_result.scalar_one_or_none()

                # Hard suppression: a contact suppressed via an unsubscribe reply
                # (Feature 1), one who used the unsubscribe link, or an address on
                # the org do-not-contact list (Feature 2), is never emailed again.
                # Same three-way check ``send_sequence_step`` makes; see there.
                #
                # **Stopped here, not just skipped.** This block used to
                # ``continue`` on the stated grounds that "send_sequence_step
                # re-checks and stops the enrollment properly" — but skipping is
                # precisely what stops that task from ever running, so nothing
                # closed the enrollment. It stayed ``active`` forever: re-read,
                # re-ranked and re-suppression-checked on every five-minute
                # sweep, for every contact who has ever opted out, and shown as
                # a live enrollment on a campaign that can never send it another
                # message. An opt-out is terminal for the enrollment, and the
                # sweep is the only thing that will ever look at this row again,
                # so it has to be the one to end it — with the same status and
                # stamp the send task would have written.
                from app.services import suppression as _suppression

                if contact and (
                    getattr(contact, "suppressed", False)
                    or getattr(contact, "unsubscribed", False)
                    or await _suppression.is_suppressed(db, contact.email)
                ):
                    # ``unsubscribed``, never ``bounced``: a compliance stop is
                    # not a delivery failure and must not move sender metrics.
                    enrollment.status = "unsubscribed"
                    enrollment.completed_at = now
                    stopped_suppressed += 1
                    logger.info(
                        "Enrollment %s stopped: contact is suppressed", enrollment.id
                    )
                    continue

                # Timezone-aware send window: use the contact's own timezone when
                # set, else the global default. Skip (don't drop) if it's outside
                # their business-hours window right now.
                #
                # Applied to manual steps too: the window is business hours in
                # the contact's own timezone, and "don't ring them at 3 AM" is at
                # least as true of a phone call as of an email.
                tz = getattr(contact, "timezone", None) if contact else None
                in_window, window_reason = limiter.within_send_window(tz)
                if not in_window:
                    skipped_window += 1
                    logger.debug("Enrollment %s outside send window: %s", enrollment.id, window_reason)
                    continue

                # Skip enrollments whose recipient domain is already known-bad.
                # Email only — a dead mailbox says nothing about a phone number.
                if is_email_step and contact and contact.email_status == "invalid":
                    logger.info("Skipping enrollment %s: invalid recipient domain", enrollment.id)
                    continue

                if is_email_step and contact and contact.email:
                    can_send_contact, contact_reason = await limiter.can_send_to_contact(db, contact.email)
                    if not can_send_contact:
                        logger.info("Skipping enrollment %s: %s", enrollment.id, contact_reason)
                        continue

                # --- Smart Send-Time Optimization (Feature 6) ---
                # The step is due and we're inside the (broad) business-hours
                # window. Ask the optimizer whether to HOLD it for the recipient's
                # personal best moment instead of firing now. It assigns the A/B
                # arm (optimized vs control) once, resolves/caches the contact's
                # timezone, and stamps ``scheduled_send_at``; the delay is capped
                # (max_smart_delay) so it can never stall a sequence, and the
                # warm-up/danger-pause guards above always take precedence.
                #
                # Email only. The optimizer picks the moment a message is most
                # likely to be *read*, which is a fact about a mailbox — holding
                # a call task for the recipient's best email-open hour optimizes
                # the wrong thing, and holding a step that is waiting to expire
                # just delays the expiry.
                if is_email_step and contact is not None:
                    optimizer = get_send_time_optimizer()
                    if await optimizer.should_defer(db, enrollment, contact, now, due_at):
                        scheduled_smart += 1
                        logger.debug(
                            "Enrollment %s held for optimal window (scheduled %s)",
                            enrollment.id, enrollment.scheduled_send_at,
                        )
                        continue

                # Take the lease *before* the task exists, and commit it there
                # and then. A stamp written now and committed at the end of the
                # sweep would leave the row locked for the whole loop, against a
                # worker whose first act is to UPDATE that same row to claim the
                # step — and would be lost entirely if the sweep died mid-list,
                # which is exactly when the duplicate dispatch it prevents is
                # most likely.
                enrollment.dispatched_at = datetime.now(timezone.utc)
                await db.commit()

                # Dispatch the send
                send_sequence_step.delay(str(enrollment.id))
                dispatched += 1

            await db.commit()
            return {
                "active_enrollments": len(enrollments),
                "dispatched": dispatched,
                "skipped_outside_window": skipped_window,
                "held_for_optimal_window": scheduled_smart,
                "skipped_daily_quota": skipped_quota,
                "skipped_in_flight": skipped_in_flight,
                "stopped_suppressed": stopped_suppressed,
            }

        return await for_each_org(_body, label="process_outreach_queue")

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("process_outreach_queue failed: %s", str(exc))
        return {"status": "error", "error": str(exc)}


async def expire_stale_review_drafts(
    db: AsyncSession,
    *,
    older_than_hours: int = 24,
    now: datetime | None = None,
    dispatch: bool = True,
) -> dict:
    """Release ``pending`` AI drafts older than the cutoff and un-park them.

    A ``pending_review`` draft freezes its enrollment: the outreach queue only
    dispatches ``active`` enrollments, so an unreviewed draft stalls the prospect
    forever. This sweep is the backstop for the human reviewer never showing up.

    Crucially it does **not** send the unreviewed AI copy — that is the exact
    thing the review gate exists to prevent. It mirrors a *rejection*: the draft
    is marked ``rejected`` (which makes :func:`ai_compose.compose_step` fall back
    to the static template the human already wrote for the step) and the
    enrollment is un-parked back to ``active`` so the next send goes out on the
    safe copy. A missed review therefore costs a plainer email, never a frozen
    enrollment or an unreviewed cold send.

    Returns ``{"expired": n, "resumed": m}``. Callable directly (tested against a
    plain session); the Celery task wraps it per-org.
    """
    from app.models.ai_email_draft import AiEmailDraft

    now = now or datetime.now(timezone.utc)
    if older_than_hours <= 0:
        return {"expired": 0, "resumed": 0}
    cutoff = now - timedelta(hours=older_than_hours)

    stale = list(
        (
            await db.execute(
                select(AiEmailDraft).where(
                    AiEmailDraft.status == "pending",
                    AiEmailDraft.created_at < cutoff,
                )
            )
        ).scalars().all()
    )
    if not stale:
        return {"expired": 0, "resumed": 0}

    resumed = 0
    to_dispatch: list[str] = []
    for draft in stale:
        draft.status = "rejected"
        draft.reviewed_at = now
        # Distinguish the auto-release from a human rejection in the audit trail
        # while staying within the draft-status enum (compose_step reads the
        # 'rejected' status, not the note).
        note = f"auto-expired after {older_than_hours}h with no review — released to static template"
        draft.review_notes = (
            f"{draft.review_notes}\n{note}" if draft.review_notes else note
        )

        enrollment = (
            await db.execute(
                select(OutreachEnrollment).where(
                    OutreachEnrollment.id == draft.enrollment_id
                )
            )
        ).scalar_one_or_none()
        if enrollment is not None and enrollment.status == "pending_review":
            enrollment.status = "active"
            enrollment.next_send_at = now
            resumed += 1
            to_dispatch.append(str(enrollment.id))

    await db.commit()

    if dispatch and to_dispatch:
        try:
            for eid in to_dispatch:
                send_sequence_step.delay(eid)
        except Exception as exc:  # pragma: no cover - broker-dependent
            logger.warning("Could not dispatch resumed enrollment(s): %s", exc)

    logger.info(
        "Auto-expired %d stale review draft(s), resumed %d enrollment(s)",
        len(stale), resumed,
    )
    return {"expired": len(stale), "resumed": resumed}


@celery_app.task(name="app.tasks.outreach.auto_expire_review_drafts")
def auto_expire_review_drafts():
    """Periodic task: release ``pending_review`` drafts left too long unreviewed.

    See :func:`expire_stale_review_drafts` for the semantics — a stalled draft is
    dropped back to the static template and its enrollment un-parked, so a missed
    review never freezes a prospect. Runs per-org so releases stay tenant-scoped.
    """
    from app.config import get_settings as _get_settings

    hours = _get_settings().ai_compose_review_expiry_hours
    if hours <= 0:
        logger.info("Review auto-expiry disabled (ai_compose_review_expiry_hours=0)")
        return {"status": "disabled"}

    async def _run():
        async def _body(db):
            return await expire_stale_review_drafts(db, older_than_hours=hours)

        return await for_each_org(_body, label="auto_expire_review_drafts")

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("auto_expire_review_drafts failed: %s", str(exc))
        return {"status": "error", "error": str(exc)}


@celery_app.task(name="app.tasks.outreach.resume_out_of_office")
def resume_out_of_office():
    """Bring back sequences paused by an out-of-office autoresponder.

    The other half of :func:`app.services.out_of_office.pause_for_ooo`. Without
    it, an OOO pause is a one-way door: the queue only dispatches ``active``, so
    every contact who ever sent an autoreply would drop out of outreach for good.

    Hourly is the right cadence — return dates are day-granular, so being up to
    an hour late costs nothing, and the sweep is one indexed query per org.
    """

    async def _run():
        from app.services import out_of_office

        async def _body(db):
            summary = await out_of_office.resume_due(db)
            await db.commit()
            return summary

        return await for_each_org(_body, label="resume_out_of_office")

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("resume_out_of_office failed: %s", str(exc))
        return {"status": "error", "error": str(exc)}


@celery_app.task(name="app.tasks.outreach.check_email_replies")
def check_email_replies():
    """Periodic dispatcher: fan out one reply-processing task per mailbox.

    Reading a mailbox is synchronous Gmail I/O; doing every connected account
    serially inside one beat tick blocked the event loop for the whole sweep and
    let one slow/oversized mailbox starve the rest. Instead this task only
    *enumerates* connected accounts (per org, so the scoping is correct) and
    dispatches ``process_account_replies`` for each, so mailboxes are read
    concurrently across workers and a failure in one is isolated to its own task.
    """
    logger.info("Dispatching per-account smart-reply tasks")

    async def _run():
        from app.services.smart_reply import get_smart_reply_service

        service = get_smart_reply_service()

        # Per-org: Gmail accounts are org-scoped, so enumerate within each org's
        # context and hand the id (plus its org) to the per-account task.
        async def _body(db):
            from app.org_context import get_current_org_id

            org_id = get_current_org_id()
            account_ids = await service.connected_account_ids(db)
            for account_id in account_ids:
                process_account_replies.delay(str(account_id), str(org_id) if org_id else None)
            return {"dispatched": len(account_ids), "org_id": str(org_id) if org_id else None}

        return await for_each_org(_body, label="check_email_replies")

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("check_email_replies failed: %s", str(exc))
        return {"status": "error", "error": str(exc)}


@celery_app.task(
    name="app.tasks.outreach.process_account_replies", bind=True, max_retries=2
)
def process_account_replies(self, gmail_account_id: str, org_id: Optional[str] = None):
    """Process incoming replies for ONE connected Gmail account.

    Dispatched per-account by :func:`check_email_replies`. The org context is
    lost over the Celery wire, so it is rebound here — from the passed ``org_id``
    when present, else resolved from the account row (unfiltered) — before any
    org-scoped read/write. All Gmail I/O inside the service is offloaded off the
    event loop, so this task never blocks the loop while reading the mailbox.
    """
    logger.info("Processing replies for Gmail account %s", gmail_account_id)

    async def _run():
        from app.models.gmail_account import GmailAccount
        from app.services.smart_reply import get_smart_reply_service

        resolved_org = org_id
        if resolved_org is None:
            async with celery_session() as db0:
                resolved_org = (
                    await db0.execute(
                        select(GmailAccount.org_id)
                        .where(GmailAccount.id == UUID(gmail_account_id))
                        .execution_options(skip_org_filter=True)
                    )
                ).scalar_one_or_none()

        async with org_scope(UUID(str(resolved_org)) if resolved_org else None):
            async with celery_session() as db:
                service = get_smart_reply_service()
                results = await service.process_account_by_id(db, UUID(gmail_account_id))
                for r in results:
                    logger.info(
                        "Reply from %s (%s) -- intent: %s, auto-sent: %s",
                        r.get("contact_name"),
                        r.get("from_email"),
                        r.get("analysis", {}).get("intent", "unknown"),
                        r.get("auto_sent", False),
                    )
                return {"account_id": gmail_account_id, "processed": len(results)}

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("process_account_replies(%s) failed: %s", gmail_account_id, str(exc))
        return {"status": "error", "error": str(exc), "account_id": gmail_account_id}


@celery_app.task(name="app.tasks.outreach.sync_email_replies")
def sync_email_replies():
    """Periodic task: sync email replies from Gmail to local DB."""

    async def _run():
        from app.services.email_history import EmailHistoryService

        service = EmailHistoryService()

        # Per-org: connected accounts and their history are org-scoped.
        async def _body(db):
            result = await service.sync_all_accounts(db)
            return result

        return await for_each_org(_body, label="sync_email_replies")

    try:
        return _run_async(_run())
    except Exception as exc:
        logger.error("sync_email_replies failed: %s", str(exc))
        return {"status": "error", "error": str(exc)}
