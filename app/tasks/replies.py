"""Celery task + routing logic for Dialtone reply classification (Feature 1).

``classify_new_replies`` runs every 5 minutes on the ``enrichment`` queue. It
**claims** every ``email_replies`` row that hasn't been classified yet — a
conditional UPDATE stamping ``classified_at``, so a pass that overruns the Beat
interval cannot have its batch re-actioned by the next one — then runs the
keyword pre-filter → :meth:`AIService.classify_reply`, applies the confidence
gate, and dispatches the per-category auto-action:

* ``interested``     → flag for the priority inbox + notify the operator.
* ``not_interested`` → complete the contact's active enrollments.
* ``out_of_office``  → pause the contact's sequences until the parsed return
  date (or +7d), resumed automatically by
  ``app.tasks.outreach.resume_out_of_office``.
* ``wrong_person``   → mark the contact for list cleanup + stash any referral.
* ``unsubscribe``    → suppress the contact everywhere + write a consent record.
* ``bounce``         → record the failed recipient in the bounce pipeline (hard
  bounce → suppressed), so a DSN that lands as a reply is handled like any other.
* ``unknown``        → no auto-action (surfaced for manual review).

The heavy lifting lives in plain ``async`` helpers so it is unit-testable
without Celery; the task is a thin ``_run_async`` wrapper.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.tasks.org_runner import for_each_org
from app.models.bounce_record import BounceRecord
from app.models.company import Company
from app.models.consent import ConsentRecord
from app.models.contact import Contact
from app.models.email_reply import EmailReply
from app.models.email_template import EmailSend
from app.models.outreach import OutreachEnrollment
from app.services import bounce_handler
from app.services import meeting_booking_agent
from app.services.ai_service import get_ai_service
from app.services.notifications import get_notification_service
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# Auto-action confidence gate. A category from the LLM is only auto-actioned at
# or above this confidence; below it the reply is treated as ``unknown`` and
# left for manual review. Keyword hits bypass the gate (they are high-precision
# by construction, and unsubscribe must never be gated).
AUTO_ACTION_THRESHOLD = 0.75

# Default OOO hold when no return date is parseable. Re-exported from the
# service that owns the pause so the two can't drift apart.
from app.services.out_of_office import OOO_DEFAULT_PAUSE_DAYS as OOO_DEFAULT_REQUEUE_DAYS  # noqa: E402,F401

# Enrollment statuses that a reply can still act on (a completed/bounced/
# unsubscribed enrollment is already terminal).
#
# Taken from ``opt_out`` rather than spelled out here. The literal this replaces
# was ``("active", "paused")`` and omitted ``pending_review`` — an enrollment
# parked on an AI draft awaiting a human. That is not a resting state:
# ``expire_stale_review_drafts`` un-parks it back to ``active`` and dispatches a
# send. So a reply saying "unsubscribe" stopped the enrollments it could see and
# the review sweep revived the one it couldn't.
from app.services.opt_out import ACTIONABLE_STATUSES as _ACTIONABLE_STATUSES  # noqa: E402


def _run_async(coro):
    """Run async code in Celery's sync context with a fresh event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def apply_confidence_gate(
    category: str,
    confidence: float,
    source: str,
    threshold: float = AUTO_ACTION_THRESHOLD,
) -> str:
    """Return the *effective* category after the confidence gate.

    Keyword-sourced results (``source == 'keyword'``) are trusted outright —
    they are high-precision and, for unsubscribe, compliance-critical. An
    AI-sourced category below ``threshold`` degrades to ``unknown`` so a
    low-confidence guess doesn't trigger an auto-action.
    """
    if category == "unknown":
        return "unknown"
    if source == "keyword":
        return category
    if confidence >= threshold:
        return category
    return "unknown"


async def _enrollments_for_contact(
    db: AsyncSession, contact_id, statuses=_ACTIONABLE_STATUSES
) -> list[OutreachEnrollment]:
    """Load a contact's enrollments in the given (actionable) statuses."""
    if contact_id is None:
        return []
    result = await db.execute(
        select(OutreachEnrollment).where(
            OutreachEnrollment.contact_id == contact_id,
            OutreachEnrollment.status.in_(statuses),
        )
    )
    return list(result.scalars().all())


async def _contact_by_email(db: AsyncSession, email: Optional[str]) -> Optional[Contact]:
    """The contact holding ``email``, matched the way the rest of the codebase does.

    Case-insensitively, and taking the earliest row rather than asserting one.

    ``contacts.email`` is not normalised on write — the scraper, the CSV import
    and the quick-add all store whatever they were given — which is why ten other
    call sites compare ``func.lower(Contact.email)``. The reply handler compared
    the column raw. ``from_email`` is lower-cased on ingest, so the comparison
    was between a lower-cased header and a column that is only *sometimes* lower
    case, and a contact stored as ``John.Smith@Acme.com`` matched nothing.

    On the routing path that costs a reply some context. On the branch above it,
    it is the compliance failure the opt-out work was done to remove: the reply
    is classified ``unsubscribe``, no contact is found, ``suppress_contact``
    never runs, no flag is set, no consent record is written, **no address goes
    on the suppression list**, and the enrollment keeps its schedule. The person
    who answered our cold email with the word "unsubscribe" receives the next
    four steps. Nothing errors and the reply is logged as handled.

    Ordered by ``(created_at, id)`` — a duplicated address is expected here and
    ``created_at`` is a server default, so rows written in one transaction tie
    without the second key and repeat runs would pick different rows.
    """
    if not email:
        return None
    return (
        await db.execute(
            select(Contact)
            .where(func.lower(Contact.email) == email.strip().lower())
            .order_by(Contact.created_at.asc(), Contact.id.asc())
            .limit(1)
        )
    ).scalars().first()


async def _contact_via_thread(db: AsyncSession, reply: EmailReply) -> Optional[Contact]:
    """The person *we mailed*, for a reply whose ``From:`` we hold no row for.

    Address matching is the only way this handler has ever resolved a reply, and
    ``email_history`` admits a message to the inbox on a **thread** match — rule
    1 of ``_is_outreach_reply``, and the one that catches essentially every
    genuine reply. Those two facts do not line up. A prospect who answers from
    an alias, a personal address, a shared ``team@`` box, or a mailbox that
    rewrites the envelope sender is a reply we deliberately accepted and a
    ``from_email`` that matches no ``contacts`` row, so ``contact_id`` is NULL
    and every address lookup after it returns None.

    On the ``unsubscribe`` branch that is the same silent compliance failure
    case-folding was fixed for, reached by a different route: no flag, no
    consent record, no suppression entry, **and the enrollment keeps its
    schedule** — so the remaining steps go out to the address we originally
    mailed, which is exactly the address the person replying is asking us to
    stop using. Nothing raises; the reply is logged as handled and the operator
    notification says they were suppressed.

    The thread is the strongest link we have, and the ingest already stored both
    ends of it: ``email_send_id`` (matched by ``gmail_thread_id`` when the reply
    was written) and ``gmail_thread_id`` itself, which the sequence engine
    stamps on the enrollment. Either resolves to the contact the outreach was
    addressed to.

    **A forwarded thread can resolve to someone who did not write the reply.**
    That is accepted deliberately, on the reasoning ``opt_out`` already records:
    the cost of being wrong here is one prospect an org stops mailing, against a
    person who asked us to stop and kept receiving mail. Those are not
    comparable. ``wrong_person`` is the category for "you have the wrong
    person", and it does not suppress.
    """
    contact_id = None

    if reply.email_send_id is not None:
        row = (
            await db.execute(
                select(EmailSend.contact_id, EmailSend.to_email)
                .where(EmailSend.id == reply.email_send_id)
                .limit(1)
            )
        ).first()
        if row is not None:
            contact_id = row[0]
            if contact_id is None and row[1]:
                by_address = await _contact_by_email(db, row[1])
                if by_address is not None:
                    return by_address

    if contact_id is None and reply.gmail_thread_id:
        # The sequence engine writes no ``EmailSend`` row (see "Sent mail lives
        # in two tables"), so the enrollment is the only carrier of the thread id
        # for the channel with the most volume. Most recent enrollment on the
        # thread, with the id breaking a tie so repeat passes agree.
        contact_id = (
            await db.execute(
                select(OutreachEnrollment.contact_id)
                .where(OutreachEnrollment.gmail_thread_id == reply.gmail_thread_id)
                .order_by(
                    OutreachEnrollment.enrolled_at.desc(),
                    OutreachEnrollment.id.desc(),
                )
                .limit(1)
            )
        ).scalars().first()

        if contact_id is None:
            # …and the non-sequence side of the same thread, for replies whose
            # ``email_send_id`` was never filled — ``POST /replies`` and the
            # smart-reply webhook write a thread id and nothing else.
            contact_id = (
                await db.execute(
                    select(EmailSend.contact_id)
                    .where(
                        EmailSend.gmail_thread_id == reply.gmail_thread_id,
                        EmailSend.contact_id.isnot(None),
                    )
                    .order_by(EmailSend.sent_at.asc(), EmailSend.id.asc())
                    .limit(1)
                )
            ).scalars().first()

    if contact_id is None:
        return None
    return (
        await db.execute(select(Contact).where(Contact.id == contact_id).limit(1))
    ).scalars().first()


async def resolve_reply_contact(
    db: AsyncSession, reply: EmailReply
) -> Optional[Contact]:
    """Who this reply is about: the ingest's guess, then the address, then the thread.

    One resolver, because the auto-action is reached from two places and both
    have to answer this the same way. The Celery classifier does it for a fresh
    reply; ``POST /dialtone/inbox/{id}/reclassify`` does it when an operator
    corrects a category and ``route_auto_action`` is re-run. The router resolved
    ``reply.contact_id`` alone — the column the ingest fills by ``From:`` address
    — so the *manual* fix for a mis-routed reply landed on exactly the replies
    that had no such row, and an operator marking one "unsubscribe" by hand
    stopped no enrollment either.

    Order matters and is cheapest-and-strongest first: an id the ingest already
    resolved is never second-guessed (see ``_contact_via_thread`` on why a shared
    thread must not be able to redirect a named reply).
    """
    if reply.contact_id is not None:
        contact = (
            await db.execute(select(Contact).where(Contact.id == reply.contact_id))
        ).scalars().first()
        if contact is not None:
            return contact
    if reply.from_email:
        contact = await _contact_by_email(db, reply.from_email)
        if contact is not None:
            return contact
    return await _contact_via_thread(db, reply)


async def suppress_contact(db: AsyncSession, contact: Contact, reason: str) -> int:
    """Compliance-critical hard suppression for an unsubscribe.

    Sets the single-source-of-truth ``suppressed`` flag (and the legacy
    ``unsubscribed`` flag), writes a revoked consent record for the audit trail,
    and marks every actionable enrollment ``unsubscribed`` so the queue stops
    dispatching immediately. Returns the number of enrollments stopped.

    Also puts the *address* on the org suppression list. The flags live on a
    contact row, and rows are created by the scraper: someone who replied
    "unsubscribe" today and is rediscovered next month through another source
    arrives as a fresh row with ``suppressed=False``, and the sequence engine
    mails a person who has already told us to stop. The list is keyed by address
    and outlives every row, so it is the half that actually holds.
    """
    contact.suppressed = True
    contact.unsubscribed = True

    from app.services import suppression as _suppression

    # Flushes, doesn't commit — it joins this caller's transaction, so the list
    # entry and the stopped enrollments land together or not at all.
    await _suppression.add_entry(
        db, contact.email, reason="unsubscribe", source="auto"
    )

    now = datetime.now(timezone.utc)
    db.add(
        ConsentRecord(
            contact_id=contact.id,
            consent_type="none",
            basis=reason[:2000] if reason else "unsubscribe reply",
            granted_at=None,
            revoked_at=now,
        )
    )

    enrollments = await _enrollments_for_contact(db, contact.id)
    for enr in enrollments:
        enr.status = "unsubscribed"
        enr.completed_at = now

    # Book it against the campaign that prompted it, so the unsubscribe rate on
    # every campaign surface is something other than a structural zero. See
    # ``opt_out.attribute_to_campaign``. A reply saying "unsubscribe" and a click
    # on the footer link are the same event to a recipient and must count the
    # same; only the bounce and complaint paths stay out, because those have
    # their own event types and folding them in would report a dead mailbox as a
    # person who objected.
    from app.services.opt_out import attribute_to_campaign

    await attribute_to_campaign(db, [contact], now=now, reason="unsubscribe")
    return len(enrollments)


async def route_auto_action(
    db: AsyncSession,
    reply: EmailReply,
    contact: Optional[Contact],
    company: Optional[Company],
    effective_category: str,
    result: dict,
    notification_service=None,
) -> str:
    """Perform the per-category auto-action; return the ``handled_action`` code.

    Idempotent side effects only — the caller has already ensured this reply
    hasn't been classified before. Returns one of ``requeued`` | ``unsubscribed``
    | ``sequence_removed`` | ``flagged_cleanup`` | ``none``.
    """
    now = datetime.now(timezone.utc)
    contact_name = contact.full_name if contact else (reply.from_email or "a contact")
    company_name = company.name if company else "their company"

    # ``reply.contact_id`` is what the Gmail sync could resolve at ingest, and it
    # resolves by ``From:`` address alone — so it is NULL for every reply that
    # reached the inbox on a thread match from an address we hold no row for.
    # The caller has already done the wider resolution (address fold, then
    # thread); using the column instead of the result is how three branches below
    # acted on nothing for exactly the replies that ask us to stop.
    contact_id = contact.id if contact is not None else reply.contact_id

    if effective_category == "interested":
        # Route to the priority inbox and tell the operator. Do NOT auto-advance
        # or complete the sequence — a human takes it from here.
        reply.priority_flag = True
        notif = notification_service or get_notification_service()
        try:
            await notif.notify_reply(contact_name, company_name, reply.from_email or "")
        except Exception as exc:  # pragma: no cover - notification is best-effort
            logger.info("interested notify failed for reply %s: %s", reply.id, exc)
        # Hand off to the Meeting Booking Agent: if the org has an active agent
        # and the reply reads as wanting to meet, it proposes calendar slots in
        # this same thread. Best-effort and fully self-contained — a booking
        # failure must never disrupt reply routing, so it's guarded here.
        try:
            await meeting_booking_agent.on_interested_reply(
                db, reply=reply, contact=contact, company=company
            )
        except Exception as exc:  # noqa: BLE001 — booking is best-effort
            logger.info("meeting-booking handoff failed for reply %s: %s", reply.id, exc)
        return "none"

    if effective_category == "unsubscribe":
        # Compliance-critical: suppress everywhere + consent log. Also notify the
        # operator as an audit line.
        target = contact
        if target is None and reply.from_email:
            target = await _contact_by_email(db, reply.from_email)
        stopped = 0
        if target is not None:
            stopped = await suppress_contact(
                db, target, reason=f"Unsubscribe reply: {reply.subject or ''}"
            )

        # The address that actually asked us to stop goes on the list, whatever
        # we happen to hold for it. ``suppress_contact`` lists ``contact.email``
        # — the address we *mailed* — and those are only the same string when the
        # prospect replied from the mailbox we wrote to. Reply from an alias, a
        # personal address or a shared inbox and the two diverge, and the half of
        # this that survives a re-scrape recorded the wrong one, or (with no
        # contact row at all) recorded nothing while the notification below
        # claimed a suppression had happened.
        #
        # Unconditional and address-keyed, for the same reason erasure's entry is:
        # the list is what binds the send gate for a row that does not exist yet.
        # Ingest has already refused DSNs, autoresponders and our own mail before
        # a row reaches here, so ``from_email`` is a person.
        from app.services import suppression as _suppression

        listed = False
        if reply.from_email:
            _, list_status = await _suppression.add_entry(
                db, reply.from_email, reason="unsubscribe", source="auto"
            )
            listed = list_status in ("added", "duplicate")

        notif = notification_service or get_notification_service()
        try:
            outcome = (
                f"They have been suppressed across all sequences "
                f"({stopped} enrollment(s) stopped)."
                if (target is not None or listed)
                else "No suppression could be recorded — review this reply by hand."
            )
            await notif._send_notification(
                subject=f"[Dialtone] Unsubscribe — {contact_name}",
                body=(
                    f"{contact_name} ({reply.from_email}) requested to unsubscribe. "
                    f"{outcome}"
                ),
            )
        except Exception as exc:  # pragma: no cover - best-effort audit line
            logger.info("unsubscribe audit notify failed for reply %s: %s", reply.id, exc)
        return "unsubscribed"

    if effective_category == "out_of_office":
        # Genuinely pause, don't just push the date out. Leaving the enrollment
        # ``active`` with a future ``next_send_at`` reads as a live sequence on
        # every dashboard and count, and anything that recomputes that date (an
        # adaptive-delay pass, a manual re-queue, a kill-switch resume) silently
        # undoes the hold and mails someone on parental leave. ``pause_reason``
        # marks it as ours to un-pause; ``resume_out_of_office`` brings it back.
        from app.services import out_of_office

        summary = await out_of_office.pause_for_ooo(
            db, contact_id, return_date=result.get("return_date"), now=now
        )
        reply.classification_meta = {
            **(reply.classification_meta or {}),
            "ooo_paused": summary["paused"],
            "requeued_until": summary["resume_at"],
        }
        return "requeued"

    if effective_category == "bounce":
        # A delivery-failure notice for a dead address. Route it into the shared
        # bounce pipeline so the *failed recipient* (not the mail daemon that
        # sent the DSN) is recorded and suppressed on a hard bounce. The address
        # to suppress is the contact we resolved, else the original send's
        # recipient — never ``reply.from_email`` (that's mailer-daemon@…).
        target_email = contact.email if contact is not None else None
        if not target_email and reply.email_send_id is not None:
            send = (
                await db.execute(
                    select(EmailSend).where(EmailSend.id == reply.email_send_id)
                )
            ).scalars().first()
            if send is not None:
                target_email = send.to_email
        if not target_email:
            # Couldn't tie the bounce to a recipient — surface for manual review
            # rather than guess.
            return "none"

        # Two pipelines read the same mailbox and this DSN can be in both.
        # ``bounce_ingestion`` sweeps for delivery-failure notices and ledgers
        # each one in ``BounceRecord`` keyed on its Gmail message id, precisely
        # because ``record_bounce`` is not idempotent — it increments
        # ``Contact.bounce_count``. This path had no such ledger, so a DSN that
        # also landed in ``email_replies`` was counted once by each pipeline.
        #
        # The direction that costs is soft bounces. A hard bounce suppresses on
        # the first one and a second count changes nothing; a soft bounce
        # suppresses at ``soft_bounce_suppress_threshold``, so double-counting
        # reaches the threshold in half the bounces and takes a live contact out
        # of play. The contact is deliverable, the address is fine, and the only
        # thing that went wrong is that two sweeps read one message.
        #
        # ``EmailReply.gmail_message_id`` is the same id the sweep keys on, so
        # sharing the ledger makes the two paths dedup against each other rather
        # than each against itself.
        message_id = reply.gmail_message_id
        if message_id:
            already = (
                await db.execute(
                    select(BounceRecord.id)
                    .where(BounceRecord.gmail_message_id == message_id)
                    .limit(1)
                )
            ).first()
            if already is not None:
                logger.info(
                    "Reply %s is a DSN the bounce sweep already recorded (%s) — "
                    "not counting it twice", reply.id, message_id,
                )
                return "bounced"

        summary = await bounce_handler.record_bounce(
            db,
            target_email,
            diagnostic=f"{reply.subject or ''} {(reply.body_text or reply.snippet or '')[:500]}",
        )
        if message_id:
            # Ledgered after the fact rather than before, unlike the sweep: the
            # sweep parses the DSN itself and can write the row before handling
            # it, while here ``record_bounce`` is what determines the kind. The
            # window this leaves is one pass wide and closes on the next.
            db.add(
                BounceRecord(
                    gmail_message_id=message_id,
                    recipient_email=target_email,
                    bounce_kind=summary.get("kind") or "soft",
                    diagnostic=(reply.subject or "")[:2000] or None,
                    result={"source": "reply_classification", "reply_id": str(reply.id)},
                )
            )
            await db.flush()
        reply.classification_meta = {
            **(reply.classification_meta or {}),
            "bounce": {
                "recipient": target_email,
                "kind": summary.get("kind"),
                "suppressed": summary.get("suppressed"),
            },
        }
        return "bounced"

    if effective_category == "not_interested":
        # Remove from active sequences (complete), no notification.
        enrollments = await _enrollments_for_contact(db, contact_id)
        for enr in enrollments:
            enr.status = "completed"
            enr.completed_at = now
        return "sequence_removed"

    if effective_category == "wrong_person":
        # Flag for list cleanup; stash any referral for a Feature-4-style handoff.
        if contact is not None:
            contact.status = "wrong_person"
        referral = result.get("referral") or {}
        if referral.get("name") or referral.get("email"):
            reply.classification_meta = {
                **(reply.classification_meta or {}),
                "referral": {"name": referral.get("name"), "email": referral.get("email")},
            }
        enrollments = await _enrollments_for_contact(db, contact_id)
        for enr in enrollments:
            enr.status = "completed"
            enr.completed_at = now
        return "flagged_cleanup"

    # unknown → surfaced for manual review, no side effects.
    return "none"


async def classify_reply_record(
    db: AsyncSession,
    reply: EmailReply,
    ai_service=None,
    notification_service=None,
) -> dict:
    """Classify one reply, persist the result, and dispatch its auto-action.

    Writes ``reply_classification`` / ``classification_confidence`` /
    ``priority_flag`` / ``classified_at`` / ``classification_meta`` /
    ``handled_action`` on the reply. Returns a small summary dict.
    """
    ai_service = ai_service or get_ai_service()

    # Resolve contact/company for context + routing: the ingest's id, then the
    # address, then the thread. The inbox admits replies on a thread match, so
    # "we accepted this as a reply to our outreach" and "we hold a contact row
    # for the sender" are not the same condition, and the gap between them is
    # where an opt-out went missing.
    contact = await resolve_reply_contact(db, reply)

    company = None
    if reply.company_id is not None:
        company = (
            await db.execute(select(Company).where(Company.id == reply.company_id))
        ).scalars().first()
    elif contact is not None:
        company = (
            await db.execute(select(Company).where(Company.id == contact.company_id))
        ).scalars().first()

    contact_name = contact.full_name if contact else ""
    company_name = company.name if company else ""

    result = await ai_service.classify_reply(
        subject=reply.subject,
        body_text=reply.body_text or reply.snippet,
        original_subject=reply.subject or "",
        contact_name=contact_name,
        company_name=company_name,
        from_email=reply.from_email,
    )

    raw_category = result.get("category", "unknown")
    confidence = float(result.get("confidence", 0.0) or 0.0)
    source = result.get("source", "ai")
    effective_category = apply_confidence_gate(raw_category, confidence, source)

    # Persist classification fields.
    reply.reply_classification = effective_category
    reply.classification_confidence = confidence
    reply.classified_at = datetime.now(timezone.utc)
    reply.classification_meta = {
        **(reply.classification_meta or {}),
        "source": source,
        "model": result.get("model"),
        "reason": result.get("reason"),
        "raw_category": raw_category,
        "unsubscribe_explicit": result.get("unsubscribe_explicit", False),
        "return_date": result.get("return_date"),
        "referral": result.get("referral") or {"name": None, "email": None},
        # Jev's label and probabilities whenever it was asked — including when
        # the LLM made the final call — so the two can be compared per reply.
        "jev": result.get("jev"),
    }

    handled = await route_auto_action(
        db, reply, contact, company, effective_category, result,
        notification_service=notification_service,
    )
    reply.handled_action = handled

    logger.info(
        "Classified reply %s from %s → %s (conf=%.2f, source=%s) → action=%s",
        reply.id, reply.from_email, effective_category, confidence, source, handled,
    )
    return {
        "reply_id": str(reply.id),
        "category": effective_category,
        "raw_category": raw_category,
        "confidence": confidence,
        "source": source,
        "handled_action": handled,
    }


async def classify_new_replies_async(
    db: AsyncSession,
    ai_service=None,
    notification_service=None,
    limit: int = 200,
) -> dict:
    """Claim and classify every not-yet-processed reply.

    Idempotent **across overlapping runs**, which the ``IS NULL`` filter alone
    was not.

    The filter reads as the guard and is one only within a single pass. The
    classification a pass writes is not visible to anyone else until the task
    commits, and the task commits once, after the whole batch. A batch is up to
    ``limit`` replies and each costs an LLM round trip, so a pass routinely runs
    for minutes — against a Beat cadence of five. The docstring anticipated that
    overlap and drew the wrong conclusion from it: a second run starting mid-pass
    re-selects the very same uncommitted rows, sees them as unclassified, and
    runs every auto-action a second time.

    Which auto-actions those are is the point. A duplicate ``unsubscribe``
    re-suppresses (harmless). A duplicate ``out_of_office`` re-pauses and pushes
    the return date out again. A duplicate ``interested`` notifies the operator
    twice. A duplicate ``bounce`` calls ``record_bounce`` again, and that one is
    explicitly *not* idempotent — it increments ``Contact.bounce_count``, so two
    passes over one soft bounce count two, and the third such coincidence
    suppresses a live contact for no reason but how the sweeps overlapped.

    So the batch is **claimed** before it is worked: one conditional UPDATE
    stamping ``classified_at`` on rows that don't have it yet, returning the ids
    it actually took. This is the ``send_claim`` mechanism and it works for the
    same reason — the row lock. A concurrent pass issuing the identical UPDATE
    blocks until this transaction commits, then re-evaluates its ``IS NULL``
    predicate against the committed value and claims nothing. No extra commit,
    no new column, no broker coordination.

    ``classified_at`` is the column stamped, because it is the one with no
    meaning other than "a pass has taken this one" — ``reply_classification`` is
    read as a *category* by four other surfaces, and parking a sentinel there
    would surface as a category to all of them.

    The *selection* still requires both to be unset. They are written together
    here, but ``PATCH /replies/{id}/classify`` sets the category alone when an
    operator corrects a reply, so a row can carry a human's classification and
    no stamp. Selecting on the stamp alone would hand those back to the AI and
    overwrite the correction with a guess.
    """
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(EmailReply.id)
        .where(
            EmailReply.classified_at.is_(None),
            EmailReply.reply_classification.is_(None),
        )
        .order_by(EmailReply.received_at.asc())
        .limit(limit)
    )
    candidate_ids = list(result.scalars().all())

    replies: list[EmailReply] = []
    if candidate_ids:
        claimed_ids = list(
            (
                await db.execute(
                    sa_update(EmailReply)
                    .where(
                        EmailReply.id.in_(candidate_ids),
                        EmailReply.classified_at.is_(None),
                        EmailReply.reply_classification.is_(None),
                    )
                    .values(classified_at=now)
                    .returning(EmailReply.id)
                    .execution_options(synchronize_session=False)
                )
            ).scalars().all()
        )
        if claimed_ids:
            replies = list(
                (
                    await db.execute(
                        select(EmailReply)
                        .where(EmailReply.id.in_(claimed_ids))
                        .order_by(EmailReply.received_at.asc())
                    )
                ).scalars().all()
            )
        skipped = len(candidate_ids) - len(claimed_ids)
        if skipped:
            logger.info(
                "Reply classification: %d of %d candidates already claimed by a "
                "concurrent pass", skipped, len(candidate_ids),
            )

    summaries = []
    for reply in replies:
        try:
            summaries.append(
                await classify_reply_record(
                    db, reply, ai_service=ai_service, notification_service=notification_service
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad reply must not stop the batch
            logger.error("Failed to classify reply %s: %s", reply.id, exc)

    by_category: dict[str, int] = {}
    for s in summaries:
        by_category[s["category"]] = by_category.get(s["category"], 0) + 1

    return {"processed": len(summaries), "by_category": by_category}


@celery_app.task(name="app.tasks.replies.classify_new_replies")
def classify_new_replies():
    """Periodic task: classify newly-arrived replies and route them (every 5m)."""
    logger.info("Classifying new email replies")

    async def _run():
        # Per-org: replies, contacts and suppression lists are org-scoped.
        async def _body(db):
            report = await classify_new_replies_async(db)
            await db.commit()
            return report

        return await for_each_org(_body, label="classify_new_replies")

    try:
        return _run_async(_run())
    except Exception as exc:  # noqa: BLE001
        logger.error("classify_new_replies failed: %s", str(exc))
        return {"status": "error", "error": str(exc)}
