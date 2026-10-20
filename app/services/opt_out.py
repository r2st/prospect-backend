"""One address asked us to stop. Make that stick, everywhere, permanently.

The problem this exists for
---------------------------
There were three ways to opt out of Dialtone mail, and they did three different
things:

* **Reply "UNSUBSCRIBE"** (``tasks.replies.suppress_contact``) — set
  ``suppressed`` *and* ``unsubscribed``, wrote a revoked ``ConsentRecord``, and
  marked every live enrollment ``unsubscribed``. Correct.
* **The bulk compliance route** (``routers.contacts``) — the same. Correct.
* **The unsubscribe link in the footer of every email we send**, and the RFC 8058
  one-click header Gmail and Yahoo require — set ``contact.unsubscribed`` and
  nothing else.

The send path gates on ``contact.suppressed`` and on the org suppression list.
It has never read ``unsubscribed``. So the *only* opt-out a recipient can
actually reach — the link we put in front of them, and the one-click button the
mailbox provider renders on our behalf — was the only one that did not stop the
mail. Click it after step 1 of a five-step sequence and steps 2 through 5 still
arrive, on schedule, from the sender you just told to stop.

``unsubscribe`` was already a legal value in ``SUPPRESSION_REASONS`` and nothing
in the codebase had ever written one.

Two more faults in the same handler, both live:

* It matched a single contact with ``scalar_one_or_none()``. ``contacts.email``
  has no unique constraint and is not unique across orgs, so an address held by
  two rows raised ``MultipleResultsFound``, which the caller turned into a 400.
  The human saw "That link didn't work"; the mailbox provider treating a 400 as
  a broken one-click affordance saw the same. Nothing was recorded either way,
  and the duplicate rows that caused it are exactly what a re-scraped prospect
  looks like.
* It matched ``Contact.email == email`` exactly. Addresses are handled
  case-insensitively everywhere else in this codebase (``normalize_email``).

Why the suppression list, and not just the flags
------------------------------------------------
Flags live on a *row*. The scraper creates rows. Someone who opts out today and
is rediscovered next month through a different source arrives as a new contact
with ``suppressed=False``, and the sequence engine mails a person who has
already told us to stop — the same violation, one import later.

``SuppressionEntry`` is keyed by ``(org, email)`` and outlives every contact row,
so it is the durable half. The flags are the fast half, read on the row the send
path already has in hand. This module always writes both, which is why it exists
rather than each caller assembling its own half of the operation.

Which org
---------
An opt-out is addressed to the sender, not to the platform, so it is scoped to
the org whose mail prompted it. New unsubscribe tokens carry that org id.

Tokens already sitting in inboxes do not — they are a bare address, they stay
valid for a year, and we cannot ask the recipient which of our customers they
meant. For those, ``org_id=None`` applies the opt-out to every org holding that
address. That is deliberately the broad reading: the cost of being wrong is a
prospect one org stops mailing, against a prospect who asked to stop and kept
receiving mail. Those are not comparable.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import NamedTuple, Optional

from itsdangerous import URLSafeTimedSerializer
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.consent import ConsentRecord
from app.models.contact import Contact
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.models.suppression import SUPPRESSION_REASONS, SUPPRESSION_SOURCES, SuppressionEntry
from app.org_context import APPREND_ORG_ID, get_current_org_id
from app.services.suppression import normalize_email

logger = logging.getLogger(__name__)

# Random per-process fallback, used only when ``secret_key`` is unconfigured.
# Tokens signed with it die at restart, which is the correct failure for a
# missing secret — a hardcoded one is a forgeable opt-out for every address.
_FALLBACK_KEY = secrets.token_urlsafe(32)


def unsubscribe_serializer() -> URLSafeTimedSerializer:
    """The one serializer for unsubscribe tokens, minting and verifying alike.

    There were two, and they disagreed. The sender
    (``email_dispatch._make_unsubscribe_url``) fell back to a hardcoded
    ``"landline-unsubscribe-secret"`` when ``secret_key`` was unset; the verifier
    (``routers.tracking``) fell back to a random per-process key. On a box with
    no ``secret_key`` configured, every unsubscribe link in every email we sent
    was therefore signed with a key the endpoint would never accept — a
    guaranteed "That link didn't work" on the one control a recipient has, and a
    400 to every mailbox provider trying one-click on our behalf.

    One factory, so the two ends cannot drift apart again. The random fallback is
    the surviving one: a hardcoded signing key lets anyone forge an opt-out for
    any address, and losing tokens on restart is much the smaller failure.
    """
    key = get_settings().secret_key
    if not key:
        logger.warning(
            "secret_key not set — unsubscribe tokens will use a random per-process key"
        )
        key = _FALLBACK_KEY
    return URLSafeTimedSerializer(key)


def make_unsubscribe_token(email: str, org_id: Optional[uuid.UUID] = None) -> str:
    """Mint the token that goes in the footer link and the List-Unsubscribe header.

    Carries the sending org so the opt-out lands on *that* sender's suppression
    list rather than being applied across every org that happens to hold the
    address. Falls back to the ambient org context, which is bound for the whole
    of a send.
    """
    org = org_id if org_id is not None else get_current_org_id()
    payload: dict[str, str] = {"e": email}
    if org is not None:
        payload["o"] = str(org)
    return unsubscribe_serializer().dumps(payload, salt="unsubscribe")


# The endpoint that honours these tokens. It is an API route, not a frontend
# one: `routers.tracking.unsubscribe` renders its own confirmation page, and the
# RFC 8058 one-click POST has to land on the same origin.
# Derived from ``tracking_base_url`` so it tracks the deployment rather than
# being a hardcoded constant that drifts when the domain changes.


def _unsubscribe_endpoint() -> str:
    base = get_settings().tracking_base_url.rstrip("/")
    return f"{base}/api/v1/track/unsubscribe"


def make_unsubscribe_url(email: str, org_id: Optional[uuid.UUID] = None) -> str:
    """The full link that goes in a footer and in the List-Unsubscribe header.

    One builder, for the same reason there is one serializer. The SES sender
    (``email_sender._build_unsubscribe_footer``) pointed its own copy at
    ``https://landline.apprendtechnologies.com/unsubscribe`` — a *frontend* path
    that does not exist and never has, so that footer offered a 404 as the
    recipient's only way out. Signed with the hardcoded fallback secret, too, so
    even against the right endpoint it would not have verified.
    """
    return f"{_unsubscribe_endpoint()}?token={make_unsubscribe_token(email, org_id)}"


def unsubscribe_headers(
    list_unsubscribe_url: Optional[str], sender_email: Optional[str] = None
) -> dict[str, str]:
    """The RFC 2369 + RFC 8058 headers, for every transport that sends outreach.

    One builder, for the same reason there is one serializer and one URL: the
    header pair is a Gmail/Yahoo bulk-sender requirement (June 2024), and a
    transport that omits it is not slightly less compliant than the others —
    it loses the native "Unsubscribe" affordance in the recipient's client and
    gets rated as a bulk sender who doesn't offer one.

    There were two copies (``gmail_sender`` and ``smtp_sender``) and a third
    transport that had neither. Returns ``{}`` when there is no URL, so a caller
    can splat it unconditionally.
    """
    if not list_unsubscribe_url:
        return {}
    methods = []
    mailto = get_settings().unsubscribe_mailto or sender_email
    if mailto:
        methods.append(f"<mailto:{mailto}?subject=unsubscribe>")
    methods.append(f"<{list_unsubscribe_url}>")
    return {
        "List-Unsubscribe": ", ".join(methods),
        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
    }


# Enrollment states a do-not-contact decision has to interrupt. Everything else
# is already finished (``completed``, ``bounced``, ``unsubscribed``) and
# rewriting it would only destroy the record of how it ended.
#
# **Every** do-not-contact writer uses this set, not a copy of it. Five paths
# take an address out of play — the unsubscribe link, a reply saying
# "unsubscribe", a spam complaint, a hard bounce, an erasure request — and until
# now four of them carried their own narrower literal:
#
# * ``complaint_handler`` and ``tasks.replies`` stopped ``("active", "paused")``,
#   missing ``pending_review``;
# * ``bounce_handler`` stopped ``status == "active"`` alone, missing both;
# * ``gdpr.erase_subject`` stopped nothing at all.
#
# ``pending_review`` is the one that matters, because it is not a resting state.
# ``expire_stale_review_drafts`` un-parks a stale draft's enrollment back to
# ``active``, stamps it due now, and dispatches ``send_sequence_step`` — so an
# enrollment left parked by a complaint or a bounce is revived hours later and
# put back on the send path. The compliance floor in ``check_send_allowed``
# refuses the message, which is why this was survivable; what it costs is a
# revived enrollment for someone who complained, a full research-and-compose
# cycle spent to be refused, and a campaign that reports live outreach to a
# contact it can never mail.
#
# This is the lesson ``reply_matching.REPLY_STOPPABLE_STATUSES`` already
# records — "status too narrow" — arrived at a second time, from compliance
# rather than from replies.
ACTIONABLE_STATUSES = ("active", "paused", "pending_review")


async def stoppable_enrollments(
    db: AsyncSession, contact_ids, *, skip_org_filter: bool = False
) -> list[OutreachEnrollment]:
    """Every enrollment a do-not-contact decision has to interrupt.

    Takes one contact id or an iterable of them; returns the rows in
    :data:`ACTIONABLE_STATUSES`. Deliberately returns rows rather than writing
    the status itself, because the callers legitimately disagree about what to
    write: a compliance stop is ``unsubscribed`` and a hard bounce is
    ``bounced``, and ``bounce_handler`` additionally books a funnel event per
    enrollment. What they must *not* disagree about is which enrollments they
    are looking at, which is the whole of what this shares.

    ``skip_org_filter`` is for the unsubscribe-link handler, which runs on a
    token rather than a logged-in caller and so has no org bound.
    """
    if isinstance(contact_ids, (str, uuid.UUID)) or not hasattr(contact_ids, "__iter__"):
        contact_ids = [contact_ids]
    ids = [cid for cid in contact_ids if cid is not None]
    if not ids:
        return []

    stmt = select(OutreachEnrollment).where(
        OutreachEnrollment.contact_id.in_(ids),
        OutreachEnrollment.status.in_(ACTIONABLE_STATUSES),
    )
    if skip_org_filter:
        stmt = stmt.execution_options(skip_org_filter=True)
    return list((await db.execute(stmt)).scalars().all())


class OptOutResult(NamedTuple):
    """What the opt-out actually changed.

    ``recorded`` is the honest answer to "is this address now safe from us?" and
    is true whenever a suppression entry exists — including the case where no
    contact row matched at all, which is precisely when the flags alone would
    have recorded nothing and the confirmation page would have lied.
    """

    email: str
    contacts: int
    enrollments_stopped: int
    orgs: tuple[uuid.UUID, ...]

    @property
    def recorded(self) -> bool:
        return bool(self.orgs)


async def record(
    db: AsyncSession,
    email: Optional[str],
    *,
    org_id: Optional[uuid.UUID] = None,
    reason: str = "unsubscribe",
    source: str = "auto",
    basis: Optional[str] = None,
) -> Optional[OptOutResult]:
    """Stop all mail to ``email``. Commits. Idempotent.

    ``org_id`` scopes the opt-out to one sender; ``None`` applies it to every org
    holding the address (see the module docstring on legacy tokens).

    Returns ``None`` only when ``email`` is not email-shaped — the one case where
    there is nothing a caller could sensibly suppress. Otherwise the address ends
    up on at least one suppression list, so a caller can tell a recipient the
    truth without checking whether we happened to have a row for them.
    """
    normalized = normalize_email(email)
    if normalized is None:
        return None
    if reason not in SUPPRESSION_REASONS:
        reason = "unsubscribe"
    if source not in SUPPRESSION_SOURCES:
        source = "auto"
    now = datetime.now(timezone.utc)

    # ``skip_org_filter`` throughout, with the org applied explicitly instead.
    # This runs from an unauthenticated endpoint where no org is bound, and the
    # global filter's no-context behaviour is to *not* filter — relying on that
    # by accident is how a scoping change turns a compliance stop into a no-op.
    contact_q = (
        select(Contact)
        .where(func.lower(Contact.email) == normalized)
        .execution_options(skip_org_filter=True)
    )
    if org_id is not None:
        contact_q = contact_q.where(Contact.org_id == org_id)
    contacts = list((await db.execute(contact_q)).scalars().all())

    # Every row for this address, not the one row a `scalar_one_or_none` was
    # willing to accept. A person the scraper found twice is one person, and the
    # duplicate used to raise instead of unsubscribing either copy.
    for contact in contacts:
        contact.suppressed = True
        contact.unsubscribed = True
        db.add(
            ConsentRecord(
                # Explicit: the insert stamp reads the ambient org, which is
                # unset here and would file every record under the first-party
                # org regardless of whose contact it is.
                org_id=contact.org_id,
                contact_id=contact.id,
                consent_type="none",
                basis=(basis or "unsubscribe link")[:2000],
                granted_at=None,
                revoked_at=now,
            )
        )

    enrollments_stopped = 0
    if contacts:
        enrollments = await stoppable_enrollments(
            db, [c.id for c in contacts], skip_org_filter=True
        )
        for enr in enrollments:
            # ``unsubscribed``, never ``bounced``. A compliance stop is not a
            # delivery failure and must not move sender reputation metrics.
            enr.status = "unsubscribed"
            enr.completed_at = now
            enrollments_stopped += 1

        await attribute_to_campaign(db, contacts, now=now, reason=reason)

    await db.commit()

    # The durable half, committed separately so a lost race on the unique
    # constraint cannot roll back the flags and the stopped enrollments above.
    targets = {c.org_id for c in contacts}
    if org_id is not None:
        targets.add(org_id)
    if not targets:
        # Nobody holds this address — an opt-out from a legacy token whose
        # contact has since been deleted, or a bare address. Suppress in the
        # first-party org anyway: the point of the list is to survive the row.
        targets.add(APPREND_ORG_ID)

    written: list[uuid.UUID] = []
    for target in sorted(targets, key=str):
        if await _ensure_entry(db, target, normalized, reason=reason, source=source):
            written.append(target)

    result = OptOutResult(
        email=normalized,
        contacts=len(contacts),
        enrollments_stopped=enrollments_stopped,
        orgs=tuple(written),
    )
    logger.info(
        "Opt-out recorded for %s: %d contact(s), %d enrollment(s) stopped, %d org list(s)",
        normalized, result.contacts, result.enrollments_stopped, len(result.orgs),
    )
    return result


# Suppression reasons that mean *this person asked us to stop*, and so belong on
# a campaign's unsubscribe rate. The rest of ``SUPPRESSION_REASONS`` — bounce,
# complaint, erasure, manual — do not.
_OPT_OUT_REASONS = frozenset({"unsubscribe"})


async def attribute_to_campaign(
    db: AsyncSession,
    contacts: list[Contact],
    *,
    now: datetime,
    reason: str = "unsubscribe",
) -> int:
    """Record the opt-out against the campaign whose mail prompted it.

    Flushes; the caller commits. Returns the number of events written.

    Every campaign reporting surface — the campaign table, the operator
    dashboard, the bounce/funnel summary — sources ``unsubscribed`` from
    ``OutreachEvent``, and **nothing in the product had ever written one**. The
    only writer was the external ``POST /outreach/events`` webhook, which almost
    nobody calls. So the opt-out flow worked perfectly and reported itself as
    never having happened: ``unsubscribe_rate`` was structurally 0.0% on every
    campaign, forever.

    That is the worst direction for this particular number to be wrong in. Reply
    and bounce rates are read to judge a campaign; the unsubscribe rate is read
    to judge whether we are *annoying people* — the leading indicator of the spam
    complaints that put a sending domain on a blocklist. A flat zero reads as
    "nobody minds", which is the reading that keeps a bad campaign running.

    Attribution is deliberately **one event per org per opt-out**, on the
    enrollment that most recently put mail in front of this address:

    * Not one per stopped enrollment. A contact in three live campaigns would
      book three opt-outs against one human, and the portfolio totals sum the
      campaign rows — so the platform would report more unsubscribes than people
      who unsubscribed.
    * Not attributed at all when nothing was ever sent. A campaign with no sends
      is already excluded from the table (and its rate would divide by zero);
      an opt-out it cannot have caused is noise.

    Which campaign actually prompted the click is not knowable — the RFC 8058
    token carries an address and an org, not an enrollment, and the tokens
    already sitting in inboxes could not carry more even if we changed it today.
    The most recent send is the honest guess, and it is the one an operator
    reading the row would make anyway.

    Only a recipient's own opt-out is booked, hence :data:`_OPT_OUT_REASONS`.
    A bounce and a complaint have their own event types and their own rails; a
    ``manual`` entry is an operator's decision about an address, not a person
    objecting. Folding any of them in here would put a dead mailbox on the chart
    an operator reads as "how many people we annoyed".
    """
    if not contacts or reason not in _OPT_OUT_REASONS:
        return 0

    by_org: dict[uuid.UUID, list[uuid.UUID]] = {}
    for contact in contacts:
        by_org.setdefault(contact.org_id, []).append(contact.id)

    written = 0
    for target_org, contact_ids in by_org.items():
        last_send = (
            await db.execute(
                select(
                    OutreachEvent.enrollment_id,
                    OutreachEvent.step_number,
                )
                .join(
                    OutreachEnrollment,
                    OutreachEvent.enrollment_id == OutreachEnrollment.id,
                )
                .where(
                    OutreachEnrollment.contact_id.in_(contact_ids),
                    OutreachEvent.event_type == "sent",
                )
                # ``occurred_at`` alone ties for steps written in one
                # transaction; the id breaks it so repeat runs agree.
                .order_by(OutreachEvent.occurred_at.desc(), OutreachEvent.id.desc())
                .limit(1)
                .execution_options(skip_org_filter=True)
            )
        ).first()
        if last_send is None:
            continue
        enrollment_id, step_number = last_send

        # Idempotent: ``record`` is called again by every one-click retry and by
        # a human clicking the link after their provider already POSTed it.
        #
        # Asked of the **contact**, not of the enrollment this pass happens to
        # have chosen. "One event per org per opt-out" is the rule stated above,
        # and a per-enrollment check only enforces it while this function is the
        # sole writer of ``unsubscribed`` events. It is not: ``POST
        # /outreach/events`` records an ESP's unsubscribe against the enrollment
        # the ESP names, which need not be the one holding the most recent send.
        # Scoped per org already — ``contact_ids`` comes from the ``by_org``
        # grouping — so one tenant's booking cannot suppress another's.
        already = (
            await db.execute(
                select(OutreachEvent.id)
                .join(
                    OutreachEnrollment,
                    OutreachEvent.enrollment_id == OutreachEnrollment.id,
                )
                .where(
                    OutreachEnrollment.contact_id.in_(contact_ids),
                    OutreachEvent.event_type == "unsubscribed",
                )
                .limit(1)
                .execution_options(skip_org_filter=True)
            )
        ).scalars().first()
        if already is not None:
            continue

        event = OutreachEvent(
            # Explicit: this runs from an unauthenticated endpoint with no
            # org bound, where the insert stamp falls back to the
            # first-party org and would file a stranger's opt-out under ours.
            org_id=target_org,
            enrollment_id=enrollment_id,
            step_number=step_number,
            event_type="unsubscribed",
            occurred_at=now,
            event_metadata={"source": "opt_out", "reason": reason},
        )
        # The check above is a SELECT, and this is the INSERT — which is not one
        # statement, and this function runs from public endpoints served
        # concurrently: a mailbox provider POSTing one-click twice in parallel,
        # or a human clicking the footer link while that POST is in flight. Both
        # passes read no event and both wrote one. ``uq_outreach_events_outcome_
        # once`` (migration 118) now refuses the second, so losing the race has
        # to leave the session usable — hence the savepoint, the same shape
        # ``_ensure_entry`` uses a few lines down for the suppression row and
        # for the same reason. Losing means the opt-out is already booked, which
        # is the whole requirement.
        try:
            async with db.begin_nested():
                db.add(event)
                await db.flush()
        except IntegrityError:
            logger.debug(
                "Opt-out event for enrollment %s step %s was written by a "
                "concurrent pass", enrollment_id, step_number,
            )
            continue
        written += 1

    return written


async def _ensure_entry(
    db: AsyncSession,
    org_id: uuid.UUID,
    normalized: str,
    *,
    reason: str,
    source: str,
) -> bool:
    """Put ``normalized`` on ``org_id``'s suppression list. True if it is there now.

    Idempotent against ``uq_suppression_org_email`` from both directions: the
    pre-check handles the ordinary repeat, and the ``IntegrityError`` handles the
    one that matters — a mailbox provider POSTing one-click twice in parallel, or
    a click landing while the provider's POST is in flight. Losing that race
    still leaves the address suppressed, which is the whole requirement, so it is
    reported as success rather than raised.
    """
    existing = (
        await db.execute(
            select(SuppressionEntry.id)
            .where(
                SuppressionEntry.org_id == org_id,
                SuppressionEntry.email == normalized,
            )
            .limit(1)
            .execution_options(skip_org_filter=True)
        )
    ).first()
    if existing is not None:
        return True

    db.add(
        SuppressionEntry(org_id=org_id, email=normalized, reason=reason, source=source)
    )
    try:
        await db.commit()
        return True
    except IntegrityError:
        await db.rollback()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        # The flags and the stopped enrollments are already committed, so mail
        # has stopped; what is lost is only the protection against a future
        # re-import. Worth a loud log, not worth failing the opt-out.
        logger.error(
            "Could not add %s to org %s suppression list: %s", normalized, org_id, exc
        )
        await db.rollback()
        return False


# --------------------------------------------------------------------------
# Reading the state back, and undoing *this* opt-out — not any other.
# --------------------------------------------------------------------------


class SubscriptionStatus(NamedTuple):
    """What the unsubscribe page shows for one signed token.

    ``can_resubscribe`` is false whenever the address is on a list for a reason
    other than ``unsubscribe``. A bounce, a complaint, an operator's manual
    entry and above all an erasure are not the recipient's to lift from a
    footer link — the page offers re-subscribe only where the one thing in the
    way is the opt-out that same link records.
    """

    email: str
    unsubscribed: bool
    can_resubscribe: bool


def _scoped(stmt, column, org_id: Optional[uuid.UUID]):
    # Same reading as ``record``: no org on the token (legacy shape) means every
    # org holding the address. Explicit, with the global filter skipped, because
    # these run from a public endpoint with no org bound.
    stmt = stmt.execution_options(skip_org_filter=True)
    return stmt.where(column == org_id) if org_id is not None else stmt


async def status(
    db: AsyncSession, email: Optional[str], *, org_id: Optional[uuid.UUID] = None
) -> Optional[SubscriptionStatus]:
    """Is ``email`` opted out of ``org_id``'s mail? ``None`` if not an address.

    Asks the same two halves ``record`` writes: the list entries and the contact
    flags. Either one is enough to read as unsubscribed — rows carrying a flag
    with no list entry are in the database, and showing such a person an
    "Unsubscribe" button for mail that has already stopped would be a lie in
    the other direction.
    """
    normalized = normalize_email(email)
    if normalized is None:
        return None

    reasons = set(
        (
            await db.execute(
                _scoped(
                    select(SuppressionEntry.reason).where(
                        SuppressionEntry.email == normalized
                    ),
                    SuppressionEntry.org_id,
                    org_id,
                )
            )
        ).scalars().all()
    )
    flagged = (
        await db.execute(
            _scoped(
                select(Contact.id)
                .where(
                    func.lower(Contact.email) == normalized,
                    (Contact.suppressed.is_(True)) | (Contact.unsubscribed.is_(True)),
                )
                .limit(1),
                Contact.org_id,
                org_id,
            )
        )
    ).first() is not None

    return SubscriptionStatus(
        email=normalized,
        unsubscribed=bool(reasons) or flagged,
        can_resubscribe=not (reasons - _OPT_OUT_REASONS),
    )


async def resubscribe(
    db: AsyncSession, email: Optional[str], *, org_id: Optional[uuid.UUID] = None
) -> Optional[SubscriptionStatus]:
    """Undo an opt-out the recipient made with this same link. Commits.

    Lifts only what :func:`record` writes for ``reason="unsubscribe"``: those
    list entries, and the contact flags. Refuses (changes nothing, returns a
    status with ``can_resubscribe=False``) when any other reason is on the list
    — see :class:`SubscriptionStatus`. A contact's ``suppressed`` flag is the
    same flag a bounce or a complaint sets, which is why the refusal is decided
    before anything is touched rather than per row.

    Stopped enrollments stay stopped. Re-subscribing says "you may mail me",
    not "resume the cadence from step 3"; restarting outreach is an operator's
    call, made from the product with the context to make it.
    """
    current = await status(db, email, org_id=org_id)
    if current is None or not current.can_resubscribe:
        return current
    normalized = current.email
    now = datetime.now(timezone.utc)

    entries = (
        await db.execute(
            _scoped(
                select(SuppressionEntry).where(
                    SuppressionEntry.email == normalized,
                    SuppressionEntry.reason.in_(_OPT_OUT_REASONS),
                ),
                SuppressionEntry.org_id,
                org_id,
            )
        )
    ).scalars().all()
    for entry in entries:
        await db.delete(entry)

    contacts = (
        await db.execute(
            _scoped(
                select(Contact).where(func.lower(Contact.email) == normalized),
                Contact.org_id,
                org_id,
            )
        )
    ).scalars().all()
    for contact in contacts:
        was_opted_out = contact.suppressed or contact.unsubscribed
        contact.suppressed = False
        contact.unsubscribed = False
        if was_opted_out:
            db.add(
                ConsentRecord(
                    org_id=contact.org_id,
                    contact_id=contact.id,
                    consent_type="express",
                    basis="re-subscribed via unsubscribe link",
                    granted_at=now,
                    revoked_at=None,
                )
            )

    await db.commit()
    logger.info(
        "Re-subscribed %s: %d list entr(ies) lifted, %d contact(s) cleared",
        normalized, len(entries), len(contacts),
    )
    return SubscriptionStatus(email=normalized, unsubscribed=False, can_resubscribe=True)
