"""Unified outreach email dispatch: Gmail-as-user, with SES fallback.

Supports multiple sending accounts with round-robin rotation. Picks sender:

    1. An explicitly requested connected Gmail account (``sender_account_id``)
    2. Round-robin rotation across all ``sending_enabled`` + connected accounts
    3. Amazon SES (legacy fallback, used when no Gmail account is connected)

The CAN-SPAM unsubscribe footer is appended here so every transport sends the
same compliant body.
"""

import asyncio
import html as html_mod
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from sqlalchemy import select, func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession


from app.config import get_settings
from app.models.gmail_account import GmailAccount
from app.models.smtp_account import SmtpAccount
from app.models.email_signature import EmailSignature
from app.services.email_sender import EmailSender
from app.services.gmail_sender import GmailSender
from app.services.smtp_sender import SmtpSender
from app.services.template_attachments import OutgoingAttachment

logger = logging.getLogger(__name__)


def _html_to_plain_text(html_body: str) -> str:
    """Convert HTML email body to clean plain text.

    Strips tags, converts common HTML elements to readable text equivalents,
    and collapses excessive whitespace.  Used to generate the ``text/plain``
    MIME alternative so that the plain-text part never contains raw HTML tags
    — a well-known spam signal.
    """
    text = html_body

    # Block-level breaks
    text = re.sub(r'<br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<hr[^>]*>', '\n---\n', text, flags=re.IGNORECASE)

    # Links: keep the text and URL
    text = re.sub(
        r'<a\s[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        r'\2 (\1)',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Strip all remaining tags
    text = re.sub(r'<[^>]+>', '', text)

    # Decode HTML entities
    text = html_mod.unescape(text)

    # Collapse runs of whitespace (but preserve intentional double-newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _is_html(text: str) -> bool:
    """Check if text contains block-level HTML tags (is already formatted HTML)."""
    return bool(re.search(r'<(p|div|ul|ol|h[1-6]|table|br|hr)\b', text, re.IGNORECASE))


def _plain_text_to_html(text: str) -> str:
    """Convert plain text with newlines to simple HTML with <br> tags.

    This mimics how Gmail composes emails — each newline becomes a <br>,
    producing a natural email look without marketing-style paragraph spacing.
    """
    # Double newlines → paragraph break (two <br>s)
    # Single newlines → line break (one <br>)
    return text.replace('\n', '<br>\n')


def _inject_inline_styles(html: str) -> str:
    """Add inline styles to common HTML tags for email client compatibility."""
    replacements = [
        (r'<p(?=[\s>])', '<p style="margin:0 0 10px 0"'),
        (r'<ul(?=[\s>])', '<ul style="margin:0 0 10px 0;padding-left:24px"'),
        (r'<ol(?=[\s>])', '<ol style="margin:0 0 10px 0;padding-left:24px"'),
        (r'<li(?=[\s>])', '<li style="margin:0 0 4px 0"'),
        (r'<strong(?=[\s>])', '<strong style="font-weight:bold"'),
        (r'<b(?=[\s>])', '<b style="font-weight:bold"'),
        (r'<em(?=[\s>])', '<em style="font-style:italic"'),
        (r'<i(?=[\s>])', '<i style="font-style:italic"'),
        (r'<a(?=[\s>])', '<a style="color:#1a73e8;text-decoration:underline"'),
        (r'<h2(?=[\s>])', '<h2 style="font-size:18px;font-weight:bold;margin:16px 0 8px 0"'),
        (r'<h3(?=[\s>])', '<h3 style="font-size:16px;font-weight:bold;margin:14px 0 6px 0"'),
        (r'<hr(?=[\s>/])', '<hr style="border:none;border-top:1px solid #dadce0;margin:16px 0"'),
        (r'<blockquote(?=[\s>])', '<blockquote style="border-left:3px solid #dadce0;padding-left:12px;margin:0 0 10px 0;color:#5f6368"'),
    ]
    for pattern, replacement in replacements:
        # Leave a tag that already carries a style alone: injecting a second
        # ``style=`` produced ``<p style="…" style="…">`` — invalid HTML (only
        # the first attribute counts), and malformed markup is itself a
        # spam-filter signal. Every footer paragraph and link hit it.
        html = re.sub(pattern + r"(?![^>]*\bstyle\s*=)", replacement, html)
    return html


def wrap_html_email(body_html: str) -> str:
    """Wrap email content in a simple HTML document that looks like a normal Gmail email.

    No marketing-style card layout — just clean, simple formatting that matches
    how Gmail composes and sends emails.
    """
    # Convert plain text to HTML if needed
    if not _is_html(body_html):
        body_html = _plain_text_to_html(body_html)

    styled = _inject_inline_styles(body_html)
    return (
        '<!DOCTYPE html>'
        '<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '</head>'
        '<body style="margin:0;padding:0">'
        '<div style="font-family:Arial,Helvetica,sans-serif;font-size:14px;'
        'line-height:1.5;color:#222222">'
        f'{styled}'
        '</div>'
        '</body></html>'
    )


def _make_sender(account):
    """Return the transport sender object for a Gmail or SMTP account."""
    if isinstance(account, SmtpAccount):
        return SmtpSender(account)
    return GmailSender(account)


def _transport_for(account) -> str:
    return "smtp" if isinstance(account, SmtpAccount) else "gmail"


def _rotation_key(account):
    """Sort key for round-robin: never-used first, then least-recently-used.

    Avoids comparing ``None`` directly (Python 3 forbids it) by sorting on a
    has-been-used flag first, then falling back to ``connected_at``.

    Both timestamps are normalised to UTC first. They are
    ``TIMESTAMP(timezone=True)`` columns, so a value read back from the database
    and a value just assigned in Python are both aware — but a pool can mix a
    row loaded by one driver with an instance stamped by
    ``record_account_send``, and ``min()`` over a naive/aware mixture raises
    ``TypeError`` rather than sorting badly. That would take down *sender
    selection* for every send, which is a large failure for a comparison.
    """
    def _utc(value):
        if value is None:
            return None
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    last_used = _utc(account.last_used_at)
    connected = _utc(account.connected_at)
    return (0 if last_used is None else 1, last_used or connected, connected)


@dataclass
class SenderInfo:
    """Describes who outreach mail will be sent as."""

    transport: str  # "gmail" | "smtp" | "ses"
    from_email: str
    display_name: Optional[str] = None
    account_id: Optional[uuid.UUID] = None
    connected: bool = False


async def get_active_account(db: AsyncSession):
    """The primary sender across all account types (Gmail preferred, then SMTP).

    Takes the **oldest** connected mailbox of the preferred transport, rather
    than asserting there is only one.

    Both queries used ``scalar_one_or_none``, which reads as the careful option
    and is in fact an assertion that at most one row matches — raising
    ``MultipleResultsFound`` the moment a second does. Nothing promises that
    here: ``gmail_accounts`` is unique on ``email`` and ``google_sub`` and on
    nothing else, and :func:`get_sending_accounts` directly below returns a
    *list* of connected accounts because multi-mailbox sending is the design.
    Rotation, per-account warm-up ramps and per-account sender health all exist
    to spread volume over several. So the second mailbox a tenant connected
    turned this into a 500.

    It survived because the outreach path reaches this only as a fallback, after
    :func:`get_next_rotation_account` finds nothing — which needs connected
    accounts with ``sending_enabled=False``. ``welcome_email`` and
    ``notifications`` have no rotation step and call this outright, so for them
    a second mailbox was an unconditional failure.

    Oldest-first (``connected_at``, then ``id`` to break a tie) because this
    answers "who is the established primary sender", and because an arbitrary
    winner is its own bug: the caller is choosing an outreach *identity*, and
    alternating between mailboxes at random is exactly what the rotation and
    warm-up machinery exists to do deliberately instead.
    """
    for model in (GmailAccount, SmtpAccount):
        account = (
            await db.execute(
                select(model)
                .where(
                    model.is_active == True,  # noqa: E712
                    model.status == "connected",
                )
                .order_by(model.connected_at.asc(), model.id.asc())
                .limit(1)
            )
        ).scalars().first()
        if account is not None:
            return account
    return None


async def get_account(db: AsyncSession, account_id: uuid.UUID):
    """Look up a sender by id across both Gmail and SMTP accounts."""
    gmail = (
        await db.execute(select(GmailAccount).where(GmailAccount.id == account_id))
    ).scalar_one_or_none()
    if gmail is not None:
        return gmail
    return (
        await db.execute(select(SmtpAccount).where(SmtpAccount.id == account_id))
    ).scalar_one_or_none()


async def get_sending_accounts(db: AsyncSession) -> list:
    """Return all sending-enabled, connected senders (Gmail + SMTP)."""
    gmail = (
        await db.execute(
            select(GmailAccount).where(
                GmailAccount.sending_enabled == True,  # noqa: E712
                GmailAccount.status == "connected",
            )
        )
    ).scalars().all()
    smtp = (
        await db.execute(
            select(SmtpAccount).where(
                SmtpAccount.sending_enabled == True,  # noqa: E712
                SmtpAccount.status == "connected",
            )
        )
    ).scalars().all()
    return list(gmail) + list(smtp)


async def _filter_by_sender_health(db: AsyncSession, accounts: list) -> list:
    """Drop accounts that shouldn't send right now: deliverability health + warm-up.

    Excludes accounts whose reputation is ``critical`` and accounts that have
    already hit their **effective** daily cap — the lower of the reputation-
    throttled cap (feature 3.3) and the per-account **warm-up** ramp (based on
    ``connected_at`` age). A brand-new mailbox is therefore held to its warm-up
    volume even before any SenderHealth assessment exists. Fail-open: accounts
    with non-numeric send counts (mocks) or no data are never excluded.
    """
    from app.models.sender_health import SenderHealth  # local import avoids a cycle
    from app.services import warmup as warmup_svc
    from app.services.send_limiter import get_send_limiter

    if not accounts:
        return accounts
    ids = [a.id for a in accounts]
    rows = (
        await db.execute(select(SenderHealth).where(SenderHealth.account_id.in_(ids)))
    ).scalars().all()
    health = {(r.account_type, r.account_id): r for r in rows}

    base_limit = get_settings().max_emails_per_day
    limiter = get_send_limiter()
    keep = []
    for a in accounts:
        h = health.get((_transport_for(a), a.id))
        if h is not None and h.status == "critical":
            continue
        throttle = h.throttle_factor if h is not None else 1.0
        effective = max(1, round(base_limit * throttle))
        # Persisted warm-up plan (Feature 1) is authoritative; else config ramp.
        warmup = await warmup_svc.plan_daily_limit(db, a)
        if warmup is None:
            warmup = limiter.account_warmup_limit(a)
        if warmup is not None:
            effective = min(effective, warmup)
        try:
            # ``warmup.used_today`` and not the raw column: ``daily_send_count``
            # is only *today's* count when ``daily_count_reset_at`` is today, and
            # nothing rolls it over except a send. Reading it raw meant a mailbox
            # that reached its cap yesterday was still at its cap this morning,
            # so rotation excluded it, so it never sent, so nothing ever reset
            # it — a mailbox frozen out for as long as any sibling stayed under
            # cap. Silent, and precisely inverted: the warm-up fleet exists to
            # spread volume, and this concentrated a whole day of it on whichever
            # mailbox happened not to finish yesterday.
            if warmup_svc.used_today(a) >= effective:
                continue
        except (TypeError, ValueError):
            # Non-numeric send count (e.g. a bare mock) → don't exclude.
            pass
        keep.append(a)
    return keep


async def get_next_rotation_account(db: AsyncSession):
    """Pick the least-recently-used sending-enabled sender for round-robin.

    Considers every sending-enabled, connected account across Gmail and SMTP so
    volume is spread evenly regardless of transport. Brand-new accounts (never
    used) are picked first. Accounts flagged unhealthy by the deliverability
    optimizer (feature 3.3) are skipped when healthier accounts are available.
    """
    accounts = await get_sending_accounts(db)
    if not accounts:
        return None
    # Prefer healthy accounts; never halt sending entirely if all are throttled.
    pool = await _filter_by_sender_health(db, accounts) or accounts
    # Prefer business-domain mailboxes over personal ones (gmail.com, outlook.com
    # …). Cold outreach from a free-mail address is one of the strongest spam
    # signals there is, and in rotation it drags down every message it carries.
    # A personal mailbox still sends when it is the only one left.
    business = [a for a in pool if not is_personal_mailbox(a.email)]
    if business:
        pool = business
    else:
        logger.warning(
            "Only personal mailboxes are sending-enabled (%s); cold outreach from "
            "free-mail addresses is likely to land in spam",
            ", ".join(sorted(a.email for a in pool)),
        )
    return min(pool, key=_rotation_key)


def is_personal_mailbox(email: Optional[str]) -> bool:
    """Is this a free-mail (consumer) address rather than a business domain?"""
    from app.services.crm_import import FREE_EMAIL_DOMAINS

    domain = (email or "").rsplit("@", 1)[-1].strip().lower()
    return domain in FREE_EMAIL_DOMAINS


PERSONAL_MAILBOX_WARNING = (
    "Personal mailbox: cold outreach from a free-mail address (gmail.com, "
    "outlook.com …) is a top spam trigger. Rotation uses your business-domain "
    "mailboxes first; send outreach from a domain you own."
)


async def resolve_sender_info(
    db: AsyncSession, sender_account_id: Optional[uuid.UUID] = None
) -> SenderInfo:
    """Describe the sender that would be used for an outreach send right now."""
    account = None
    if sender_account_id is not None:
        account = await get_account(db, sender_account_id)

    # Try rotation (least recently used sending-enabled account)
    if account is None:
        account = await get_next_rotation_account(db)

    # Fallback to active account
    if account is None:
        account = await get_active_account(db)

    if account is not None and account.status == "connected":
        return SenderInfo(
            transport=_transport_for(account),
            from_email=account.email,
            display_name=account.display_name,
            account_id=account.id,
            connected=True,
        )
    # No Gmail/SMTP account connected yet. Report the intended primary sender
    # (apprend.tech) so the deliverability + test-send endpoints check the domain
    # outreach will actually send from once OAuth is connected. Fall back to the
    # SES identity only if no primary sender is configured.
    settings = get_settings()
    return SenderInfo(
        transport="ses",
        from_email=settings.primary_sender_email or settings.ses_sender_email,
        connected=False,
    )


@dataclass(frozen=True)
class SenderIdentity:
    """Who the mail is from, for the footer's CAN-SPAM / CASL lines.

    Both laws require the sender's name and a valid physical postal address on
    every commercial email, and the sender is the *organization* using
    Prospect — not DoAide. So it comes from the org's own settings
    (``Organization.branding``, edited on Organization Settings):

    * ``company_name``    — falls back to the org's name;
    * ``company_address`` — the postal address (left out when not set, with a
      warning logged: the footer is then not CAN-SPAM complete);
    * ``company_website`` — optional, its own line under the address.

    A link to the sender's *own* website aligns with the From domain, unlike the
    old "Powered by DoAide Prospect" link to a third-party domain.
    """

    name: str
    postal_address: str = ""
    website: str = ""

    def line(self) -> str:
        return f"{self.name} · {self.postal_address}" if self.postal_address else self.name

    @property
    def website_url(self) -> str:
        site = self.website.strip()
        if site and not re.match(r"^https?://", site, re.IGNORECASE):
            site = f"https://{site}"
        return site


def default_sender_identity() -> SenderIdentity:
    """The deployment's identity, for a send with no organization bound."""
    settings = get_settings()
    return SenderIdentity(settings.company_name, settings.company_postal_address)


def sender_identity_for_org(org) -> SenderIdentity:
    """The footer identity from an org's branding settings."""
    branding = org.branding or {}

    def _get(key: str) -> str:
        value = branding.get(key)
        return value.strip() if isinstance(value, str) else ""

    identity = SenderIdentity(
        name=_get("company_name") or org.name,
        postal_address=_get("company_address"),
        website=_get("company_website"),
    )
    if not identity.postal_address:
        logger.warning(
            "Org %s has no company address set; email footers omit the postal "
            "address CAN-SPAM requires. Set it on Organization Settings.",
            getattr(org, "slug", org.id),
        )
    return identity


async def resolve_sender_identity(db: AsyncSession) -> SenderIdentity:
    """The current org's footer identity.

    Never raises: a lookup failure must not stop the mail.
    """
    try:
        from app.models.tenant import Organization
        from app.org_context import get_current_org_id

        org_id = get_current_org_id()
        if org_id is not None:
            org = await db.get(Organization, org_id)
            if org is not None:
                return sender_identity_for_org(org)
    except Exception:  # pragma: no cover - defensive
        logger.exception("Sender identity lookup failed; using the default")
    return default_sender_identity()


def _make_unsubscribe_url(contact_email: str) -> str:
    """Generate the signed unsubscribe URL for a contact.

    The token comes from :mod:`app.services.opt_out`, which owns both minting and
    verification. This used to sign with ``secret_key or
    "landline-unsubscribe-secret"`` while the endpoint verified with
    ``secret_key`` or a *random per-process* key — so on any box without a
    configured ``secret_key``, every unsubscribe link we sent was signed with a
    key the endpoint would reject. It also carries the sending org now, so the
    opt-out is scoped to the sender the recipient is actually replying to.
    """
    from app.services.opt_out import make_unsubscribe_url

    return make_unsubscribe_url(contact_email)


def _unsubscribe_footer_text(
    contact_email: str, identity: Optional[SenderIdentity] = None
) -> str:
    """Plain-text unsubscribe footer for the text/plain part.

    Unsubscribe plus the sender's name, postal address and website (see
    :class:`SenderIdentity`), nothing else. No "Powered by" link: a link to a
    domain other than the sender's, in every message, is a cross-domain signal
    filters score against cold mail.
    """
    identity = identity or default_sender_identity()
    unsubscribe_url = _make_unsubscribe_url(contact_email)
    return (
        "\n\n—\n"
        f"Don't want these emails? Reply UNSUBSCRIBE or click: "
        f"{unsubscribe_url}\n"
        f"{identity.line()}"
        + (f"\n{identity.website_url}" if identity.website_url else "")
    )


def _unsubscribe_footer_html(
    contact_email: str, identity: Optional[SenderIdentity] = None
) -> str:
    """HTML unsubscribe footer with proper inline styles.

    Same lines as :func:`_unsubscribe_footer_text`, so the HTML and plain-text
    parts say the same thing.
    """
    identity = identity or default_sender_identity()
    unsubscribe_url = html_mod.escape(_make_unsubscribe_url(contact_email), quote=True)
    return (
        '<p style="margin:24px 0 0 0;font-size:12px;color:#6b7280">—<br>'
        'Don\'t want these emails? Reply UNSUBSCRIBE or click: '
        f'<a href="{unsubscribe_url}" style="color:#6b7280">unsubscribe</a><br>'
        f'{html_mod.escape(identity.line())}'
        + (
            f'<br><a href="{html_mod.escape(identity.website_url, quote=True)}" '
            f'style="color:#6b7280">{html_mod.escape(identity.website_url)}</a>'
            if identity.website_url else ""
        )
        + '</p>'
    )


async def check_send_allowed(
    db: AsyncSession,
    to_email: str,
    *,
    bypass_kill_switch: bool = False,
    check_kill_switch: bool = True,
) -> Optional[dict]:
    """The two compliance floors, in one place. ``None`` means "allowed".

    Returns the same ``{"status": ..., "reason": ...}`` shape
    :func:`send_outreach_email` returns, so a caller that is *not* going through
    it can hand the result straight back and report the block identically.

    Two floors, and the order matters:

    * the **global sending kill switch** — an operator's emergency stop, and the
      reason it exists is to cover paths their own module doesn't know about.
      ``bypass_kill_switch`` exists for internal callers that must still work
      while paused; ``check_kill_switch=False`` is for mail that is not outreach
      at all (a connectivity test to your own mailbox), where the pause has
      nothing to say.
    * the **do-not-contact decision** — a compliance gate, not an operational
      one. An address under one must never receive *any* outbound mail, which is
      why it is enforced even for kill-switch-bypass callers and why no flag
      turns it off. Suppression is stricter than the operational pause.

      That decision lives in **three** places, not one: the org suppression list
      and the two per-contact flags. This gate read only the list, while
      ``send_sequence_step`` reads all three and explains in its own comment why
      the flags cannot be dropped — rows with ``unsubscribed=True`` and no list
      entry are in the database now, written by handlers that predate
      ``opt_out.record``. Since every non-sequence send path is gated *here* and
      nowhere else, those opt-outs stopped the cadence and stopped nothing else:
      the smart-reply auto-send, a meeting proposal, a reminder, a human's inbox
      reply and a per-account test send all cleared a gate that could not see
      them. ``suppression.address_blocking_reason`` is the three-clause test.

    This exists as a function rather than as a block inside
    :func:`send_outreach_email` because ``send_outreach_email`` is *not*, in
    fact, the only way mail leaves this system. Several paths hold a
    ``GmailSender``/``SmtpSender`` directly — the meeting-booking agent, the
    smart-reply auto-send, the inbox reply composer, the per-account test send.
    Each of those is a send the recipient cannot tell apart from any other, so
    each has to clear the same floors; giving them a shared gate to call is what
    makes "every outbound path" a checkable claim rather than a hope. See
    ``tests/test_send_gate_coverage.py``, which enforces it mechanically.
    """
    if check_kill_switch and not bypass_kill_switch:
        from app.services import system_settings

        if not await system_settings.is_sending_enabled(db):
            logger.warning(
                "Send to %s blocked: global sending kill switch is engaged", to_email
            )
            return {
                "status": "halted",
                "reason": "Global sending kill switch is engaged — all outbound email is paused.",
                "transport": "none",
                "sender": None,
            }

    from app.services import suppression

    reason = await suppression.address_blocking_reason(db, to_email)
    if reason is not None:
        logger.info("Send to %s blocked: %s", to_email, reason)
        return {
            "status": "suppressed",
            "reason": f"Recipient must not be contacted: {reason.lower()}.",
            "transport": "none",
            "sender": None,
        }

    return None


async def record_account_send(db: AsyncSession, account, *, sent: bool) -> None:
    """Book one wire attempt against the mailbox it went out from.

    ``daily_send_count`` is what the per-account warm-up ramp is enforced
    against (``warmup.check_send_allowed`` reads it through ``used_today``) and
    what rotation uses to skip a mailbox that has had enough for today. It was
    incremented in exactly one place — inside :func:`send_outreach_email` — so
    the paths holding a transport directly ramped a mailbox's *reputation* while
    contributing nothing to the number that is supposed to be governing it. A
    day-1 mailbox on a 5/day plan sent its 5 sequence steps, then as many
    meeting proposals, auto-replies, inbox replies and "tests" as the day
    produced, and every warm-up check still read 5/5.

    The counter rolls over at the **sending day** boundary
    (:mod:`app.services.send_day` — the sender's local midnight) so it reflects
    *today* rather than growing without bound, and only a send that actually
    landed increments it: an attempt that never reached the wire cost the
    mailbox nothing.

    It rolled on the UTC boundary, which is not the same instant and not outside
    the sending day. The send window is evaluated in the *contact's* timezone,
    so with the shipped Toronto defaults a contact in Los Angeles is still
    inside their 08:00–18:00 window at 00:00 UTC — 17:00 for them. The ramp
    therefore reset an hour before that mailbox stopped sending, and a day-1
    plan on a 5/day limit put ten on the wire while every warm-up check read
    ``5/5``. The one number governing a cold domain's volume, doubled by the
    clock. ``warmup.used_today`` reads the same boundary; they have to agree or
    the counter and the gate disagree about which sends are today's.

    **The increment is a SQL expression, not a Python one.** ``count = count + 1``
    on the loaded instance is a read-modify-write across a window that spans the
    whole send: the attribute was read when the row was loaded, before recipient
    validation, composition and the SMTP round trip. Two workers sending from the
    same mailbox at once both load 3, both write 4, and the mailbox has sent 5.
    Every concurrent send after the first is free — silently, in the direction
    the ramp exists to prevent, on the one counter that is supposed to be
    governing a cold domain's volume. The same shape as the org daily cap, which
    is why that one is a Redis reservation rather than a row count.

    ``daily_send_count + 1`` evaluated by the database under the row lock the
    UPDATE takes is exact however many workers ask at once. The rollover moves
    into the same statement — a ``CASE`` on ``daily_count_reset_at`` — so a
    worker that reads a stale reset stamp cannot zero a counter another worker
    has already rolled over and incremented today.

    **Everything here runs after the message is already on the wire.** That is
    what makes the shape of the failure matter more than the odds of it: an
    exception raised in this function rolls back the send's whole bookkeeping
    transaction and Celery retries the step, which is the duplicate cold email
    ``send_claim`` exists to prevent. The statement above needs a *mapped* class
    to name a table and a column; an account object that isn't one — a stub, a
    test double, some future transport that reports sends without a row — has no
    counter to keep, and raising ``AttributeError`` at it would convert a
    delivered message into a retry over accounting we never had anywhere to put.
    So a non-mapped account is a no-op, logged, not an exception.
    """
    from sqlalchemy import case, inspect as sa_inspect, update as sa_update
    from sqlalchemy.exc import NoInspectionAvailable

    from app.services import send_day

    now = datetime.now(timezone.utc)
    day_start = send_day.day_start(now)

    # Rolled over already today? Then keep the running count; otherwise this
    # statement is the rollover. Decided in SQL against the row's *current*
    # value, so it does not depend on how stale this instance is.
    OrmAccount = type(account)
    try:
        sa_inspect(OrmAccount)
    except NoInspectionAvailable:
        logger.debug(
            "record_account_send: %r is not a mapped account — no counter to keep",
            OrmAccount.__name__,
        )
        return

    fresh = OrmAccount.daily_count_reset_at.isnot(None) & (
        OrmAccount.daily_count_reset_at >= day_start
    )
    base = case((fresh, OrmAccount.daily_send_count), else_=0)

    await db.execute(
        sa_update(OrmAccount)
        .where(OrmAccount.id == account.id)
        .values(
            last_used_at=now,
            daily_send_count=(base + 1) if sent else base,
            daily_count_reset_at=case((fresh, OrmAccount.daily_count_reset_at), else_=now),
        )
        .execution_options(synchronize_session=False)
    )
    # The instance in hand is now behind the row. Callers read these attributes
    # straight back — rotation's cap check and the warm-up gate both do — so
    # reload them from the values the database just computed rather than leaving
    # a stale count in the identity map for the rest of the request.
    await db.refresh(account, ["daily_send_count", "daily_count_reset_at", "last_used_at"])

    if sent:
        # Auto-start the warm-up ramp on the first real send so the daily cap
        # ratchets up from the true first-send date (not a stale config).
        from app.services import system_settings

        await system_settings.ensure_warmup_started(db)


@dataclass
class VolumeSlot:
    """A claim on today's sending volume, settled once the send resolves.

    Two shapes, and which one a path gets is a product decision rather than a
    technical one:

    * **refusable** — autonomous cold volume (the sequence queue, the meeting
      agent, a smart-reply auto-send). The warm-up ramp and the daily cap are
      checked *before* the wire and the send is declined when either is spent.
    * **non-refusable** (``refusable=False``) — a human hitting Reply, an
      operator testing a mailbox. Blocking those on a cold-outreach cap would be
      a worse failure than the cap prevents, so they are never declined — but
      they are counted, because the volume left the same mailbox and the next
      cold send has to see it.
    """

    account: object
    daily_reserved: bool = False
    count_daily_after: bool = False
    blocked: Optional[dict] = None


async def claim_send_volume(
    db: AsyncSession,
    account,
    *,
    refusable: bool = True,
    daily_slot_held: bool = False,
) -> VolumeSlot:
    """Claim volume for one send. ``slot.blocked`` non-None means "don't send".

    ``daily_slot_held`` is for the one caller that reserves its own slot before
    it gets here — the sequence queue, which has to claim ahead of the send
    claim and the pacing window so it can release all three coherently. Every
    other caller gets the reservation by default, because the alternative is the
    arrangement this replaced: a cap enforced in one path and absent from five,
    which reads exactly like a cap that works.

    ``account`` may be ``None`` — that is the SES fallback, used when the org
    has no mailbox connected at all. There is no per-account ramp to check
    (there is no account), but the org's daily cap is not a property of the
    transport and still applies: skipping it there would leave the one
    configuration with *no* per-account limits as the one with no limits.
    """
    slot = VolumeSlot(account=account)

    if not refusable:
        slot.count_daily_after = True
        return slot

    transport = _transport_for(account) if account is not None else "ses"
    sender_email = account.email if account is not None else None

    if account is not None:
        from app.services import warmup

        allowed, warmup_reason = await warmup.check_send_allowed(db, account)
        if not allowed:
            logger.info("Send held by warm-up on %s: %s", sender_email, warmup_reason)
            slot.blocked = {
                "status": "skipped",
                "reason": warmup_reason,
                "transport": transport,
                "sender": sender_email,
            }
            return slot

    if not daily_slot_held:
        from app.services.send_limiter import get_send_limiter

        reserved, reason = await get_send_limiter().reserve_daily_send(db)
        if not reserved:
            logger.info("Send held by the daily cap on %s: %s", sender_email, reason)
            slot.blocked = {
                "status": "skipped",
                "reason": reason,
                "daily_limit_reached": True,
                "transport": transport,
                "sender": sender_email,
            }
            return slot
        slot.daily_reserved = True

    return slot


async def settle_send_volume(db: AsyncSession, slot: VolumeSlot, *, sent: bool) -> None:
    """Close out a claim against the day: count what landed, hand back what didn't.

    Call it on every outcome including an exception out of the transport. An
    unreleased reservation is not merely untidy — it is allowance the day never
    gets back, so a flapping SMTP host would quietly eat a mailbox's quota
    without sending anything.

    This settles the *day's* counter only. The mailbox's own counter is
    :func:`record_account_send`, kept separate because the two disagree about
    what an unsent attempt means: the day gets its slot back, and the mailbox is
    still stamped as the one we last tried, which is what rotation reads.
    """
    from app.services.send_limiter import get_send_limiter

    if sent:
        if slot.count_daily_after:
            await get_send_limiter().consume_daily_send(db)
    elif slot.daily_reserved:
        await get_send_limiter().release_daily_send()


async def send_outreach_email(
    db: AsyncSession,
    to_email: str,
    subject: str,
    body_html: str,
    body_text: Optional[str] = None,
    reply_to: Optional[str] = None,
    sender_account_id: Optional[uuid.UUID] = None,
    include_unsubscribe: bool = True,
    signature_id: Optional[uuid.UUID] = None,
    thread_id: Optional[str] = None,
    in_reply_to: Optional[str] = None,
    references: Optional[str] = None,
    bypass_kill_switch: bool = False,
    skip_spam_scan: bool = False,
    daily_slot_reserved: bool = False,
    attachments: Optional[Sequence[OutgoingAttachment]] = None,
    plain_text: bool = False,
) -> dict:
    """Send one outreach email via the resolved sender.

    Returns a dict with at least ``status`` ("sent" | "skipped" | "halted" |
    "error") plus ``transport`` and ``sender`` so callers can record how it went.

    The two compliance floors — the **global sending kill switch** and the
    **do-not-contact list** — are checked first, via :func:`check_send_allowed`.
    Every *outreach* path funnels through here, so they cover sequence steps, the
    outreach queue and manual sends alike; ``bypass_kill_switch`` exists only for
    internal callers that must still work while paused (none in normal
    operation), and does not lift suppression.

    They live in :func:`check_send_allowed` rather than inline because this
    function is not the only way mail leaves the system — the meeting-booking
    agent, the smart-reply auto-send, the inbox composer and the per-account test
    send each hold a transport directly and call that gate themselves.

    The **volume** floors — the per-account warm-up ramp and the org's daily cap
    — are claimed here too, via :func:`claim_send_volume`. The daily cap used to
    be enforced only in the sequence queue, which made it a cap on *sequence*
    volume wearing the name of a cap on sending: a template blast through this
    function walked past it one contact at a time. ``daily_slot_reserved=True``
    is for the queue, which claims its own slot earlier so it can release it
    alongside the send claim and the pacing window.

    ``plain_text=True`` sends a single ``text/plain`` body and no HTML part.
    ``body_html`` is still what is scanned and what the signature and footer are
    appended to; it is flattened to text instead of wrapped in a document.

    ``attachments`` are sent as-is on every transport. They are passed to the
    transport only when present, so a sender that predates them (or a test
    double of one) is called exactly as before for a plain email.
    """
    blocked = await check_send_allowed(
        db, to_email, bypass_kill_switch=bypass_kill_switch
    )
    if blocked is not None:
        return blocked

    # --- Pre-send spam gate ---------------------------------------------------
    # The backstop for every path that doesn't scan its own copy first (manual
    # compose, template blasts, smart-reply auto-sends). The sequence path scans
    # the body *before* tracking is baked in — a rewrapped link set would read as
    # "too many links" here — so it passes ``skip_spam_scan=True`` rather than
    # being scored twice on copy it already cleared.
    if not skip_spam_scan:
        from app.services import spam_guard

        allowed, verdict = spam_guard.scan_for_send(
            subject, body_html, has_unsubscribe=include_unsubscribe
        )
        if not allowed:
            logger.error("Send to %s blocked by spam gate: %s", to_email, verdict.reason())
            return {
                "status": "skipped",
                "reason": verdict.reason(),
                "spam_blocked": True,
                "spam_scan": verdict.as_metadata(),
                "transport": "none",
                "sender": None,
            }

    account = None
    if sender_account_id is not None:
        account = await get_account(db, sender_account_id)
        if account is None or account.status != "connected":
            account = None

    # Use rotation: pick least-recently-used sending-enabled account
    if account is None:
        account = await get_next_rotation_account(db)

    # Last resort: the primary active account (even if sending_enabled is False)
    if account is None:
        account = await get_active_account(db)

    # --- Volume gates: per-account warm-up ramp, then the org's daily cap. ----
    # Both run on the *resolved* account regardless of how it was chosen
    # (explicit id, rotation, or fallback). Graduated / no plan → no warm-up
    # constraint. Skipped entirely for internal bypass callers, who are not
    # outreach and have nothing to ramp.
    # ``account`` may be None here — the SES fallback. It has no ramp, and the
    # org's daily cap still applies to it.
    slot = VolumeSlot(account=account)
    if not bypass_kill_switch:
        slot = await claim_send_volume(
            db, account, daily_slot_held=daily_slot_reserved
        )
        if slot.blocked is not None:
            return slot.blocked

    # Append email signature (specific or default)
    signature = None
    if signature_id is not None:
        result = await db.execute(
            select(EmailSignature).where(EmailSignature.id == signature_id)
        )
        signature = result.scalar_one_or_none()
    if signature is None:
        result = await db.execute(
            select(EmailSignature).where(EmailSignature.is_default == True)  # noqa: E712
        )
        signature = result.scalar_one_or_none()
    if signature is not None and signature.content_html:
        body_html = body_html + "<br><br>" + signature.content_html

    # Auto-generate clean plain text from HTML (never send HTML in text/plain).
    # This must happen *before* the HTML footer is appended: converting after
    # turned the HTML footer into text and then appended the text footer too,
    # so every text/plain part carried the unsubscribe block twice.
    body_text = _html_to_plain_text(body_html)

    # Build the unsubscribe URL once — used for both footers and the headers
    unsubscribe_url = None
    if include_unsubscribe:
        unsubscribe_url = _make_unsubscribe_url(to_email)
        identity = await resolve_sender_identity(db)
        body_html = body_html + _unsubscribe_footer_html(to_email, identity)
        body_text = body_text + _unsubscribe_footer_text(to_email, identity)

    # Wrap in a complete HTML email document with inline styles — or, for a
    # plain-text send, drop the HTML part altogether.
    body_html = "" if plain_text else wrap_html_email(body_html)

    extra = {"attachments": list(attachments)} if attachments else {}

    if account is not None:
        sender = _make_sender(account)
        try:
            result = await asyncio.to_thread(
                sender.send_email,
                to_email=to_email,
                subject=subject,
                body_html=body_html,
                body_text=body_text,
                reply_to=reply_to,
                thread_id=thread_id,
                in_reply_to=in_reply_to,
                references=references,
                list_unsubscribe_url=unsubscribe_url,
                **extra,
            )
        except Exception:
            # The transport blew up, so nothing went out and the day's allowance
            # must not stay spent — a flapping SMTP host would otherwise eat the
            # cap without sending a single message.
            await settle_send_volume(db, slot, sent=False)
            raise
        # Capture the sender email and account id for thread tracking
        result["from_email"] = account.email
        result["sender_account_id"] = str(account.id)
        landed = result.get("status") == "sent"
        await settle_send_volume(db, slot, sent=landed)
        await record_account_send(db, account, sent=landed)
        if result.get("status") == "error":
            account.status = "error"
        return result

    # Fallback: Amazon SES. No account, so no ramp and nothing to book a
    # per-mailbox count against — but the daily slot claimed above is real and
    # has to be settled either way.
    #
    # ``include_unsubscribe=False`` because the footer is already in ``body_html``
    # above; ``list_unsubscribe_url`` is the *header* pair, which this branch
    # could not set at all until ``EmailSender`` grew a raw-MIME path. The same
    # URL the Gmail and SMTP branches send — a fallback transport sharing the
    # domain's reputation must not be the one that looks like a bulk sender with
    # no one-click unsubscribe.
    ses = EmailSender()
    try:
        result = await asyncio.to_thread(
            ses.send_email,
            to_email=to_email,
            subject=subject,
            body_html=body_html,
            body_text=body_text,
            reply_to=reply_to,
            include_unsubscribe=False,
            list_unsubscribe_url=unsubscribe_url,
            **extra,
        )
    except Exception:
        await settle_send_volume(db, slot, sent=False)
        raise
    result.setdefault("transport", "ses")
    result.setdefault("sender", ses.sender)
    await settle_send_volume(db, slot, sent=result.get("status") == "sent")
    if result.get("status") == "sent":
        from app.services import system_settings
        await system_settings.ensure_warmup_started(db)
    return result
