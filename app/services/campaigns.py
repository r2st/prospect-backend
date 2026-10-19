"""Autopilot Outreach: ICP → discovery → drafts → paced sends, per campaign.

The pipeline a campaign runs, driven every 15 minutes by
``app.tasks.campaigns.run_campaign_scheduler`` (one org at a time, via
``for_each_org``):

1. **Reconcile** — sent prospects that bounced (a hard/blocked
   ``BounceRecord``) or replied (an ``EmailReply`` on the thread or from the
   address) are marked so.
2. **Bounce guard** — more than 5% of today's campaign sends bounced → the
   campaign pauses itself (``pause_reason='bounce_rate'``).
3. **Warm-up** — ``warmup_day`` advances once per local day *on which the
   campaign actually sent*, so a paused week doesn't jump the ramp. The day's
   campaign cap is ``min(daily_send_limit, 5 × warmup_day)``.
4. **Discovery**, once per local day and only while the queue is short: the
   first run queues the contacts already in the campaign's documents; later
   runs search for ICP-matching companies (grounded Gemini, else the org's
   Groq/OpenRouter fallback) and queue contacts with an email.
5. **Drafting** — a personalised email per queued prospect from the org's LLM
   chain; auto-approved unless ``review_before_send``.
6. **Scheduling** — inside the send window, approved prospects are handed to
   ``send_campaign_prospect`` with 120–300s random gaps between them.

What this module deliberately does **not** own, because the platform already
does and a second copy would drift:

* **Compliance and volume.** Every send goes through
  ``email_dispatch.send_outreach_email`` — kill switch, do-not-contact list,
  spam gate, the mailbox's warm-up ramp and the org's daily cap. The campaign's
  ramp is an extra ceiling, never a replacement.
* **Unsubscribe.** The dispatcher adds the footer and the RFC 8058
  ``List-Unsubscribe`` headers pointing at ``GET/POST /api/v1/unsubscribe``,
  which records the opt-out on the org's durable suppression list. A
  campaign-specific unsubscribe would be a second list the gate can't see.
* **The send record.** Each send writes an ``EmailSend`` row, so the inbox,
  reply matching and the daily cap's seed all see campaign mail.

"Never the same address twice" is checked when queueing (any earlier campaign
prospect, any earlier send from any path), re-checked before sending, and
finally enforced by a partial unique index on mailed campaign prospects.
"""

from __future__ import annotations

import asyncio
import html
import logging
import random
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Awaitable, Callable, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.models.bounce_record import BounceRecord
from app.models.campaign import Campaign, CampaignDailyLog, CampaignProspect
from app.models.company import Company
from app.models.contact import Contact
from app.models.discovery_document import DiscoveryDocument
from app.models.email_reply import EmailReply
from app.models.email_template import WIRE_STATUSES, EmailSend, EmailTemplate
from app.models.gmail_account import GmailAccount
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.models.tenant import Organization
from app.models.user import User
from app.services import org_gemini, suppression

logger = logging.getLogger(__name__)

TICK = timedelta(minutes=15)
# Approved emails go out as soon as they're approved, at any hour. A burst of
# approvals is spaced a random 60–120s apart so the mailbox doesn't look like a
# bot to the receiving providers' spam filters.
GAP_SECONDS = (60, 120)
WARMUP_STEP = 5
MAX_DAILY_LIMIT = 100
BOUNCE_PAUSE_RATE = 0.05
DRAFTS_PER_TICK = 5
# Per run. More than the old 10 now that a web search can supply real ones;
# bounded by Hunter's credits (a find and a verify each per company).
DISCOVERY_COMPANIES = 15
# A prospect handed to a delayed send task holds this lease; past it, a task
# that never ran (worker restart, broker loss) doesn't strand the prospect.
SCHEDULE_LEASE = timedelta(hours=1)
PENDING = ("queued", "draft_created", "approved")
# Prospects that count as "mailed today" for the campaign's own cap.
WIRE = ("sent", "replied", "bounced")
ATTACHABLE_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

ScheduleFn = Callable[[uuid.UUID, uuid.UUID, int], Any]


class CampaignError(Exception):
    """An operator-facing reason a campaign action can't proceed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Time: the campaign's own window, in the campaign's timezone
# ---------------------------------------------------------------------------
def parse_hhmm(value: str) -> time:
    m = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", (value or "").strip())
    if not m:
        raise ValueError("Use 24-hour HH:MM, e.g. 09:00")
    return time(int(m.group(1)), int(m.group(2)))


def tz_of(campaign: Campaign) -> ZoneInfo:
    try:
        return ZoneInfo(campaign.timezone or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def _aware(now: Optional[datetime]) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def local_today(campaign: Campaign, now: Optional[datetime] = None) -> date:
    return _aware(now).astimezone(tz_of(campaign)).date()


def day_start_utc(campaign: Campaign, now: Optional[datetime] = None) -> datetime:
    tz = tz_of(campaign)
    local = _aware(now).astimezone(tz)
    return datetime.combine(local.date(), time(0), tzinfo=tz).astimezone(timezone.utc)


def window_bounds(campaign: Campaign, now: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """Today's send window in UTC."""
    tz = tz_of(campaign)
    local = _aware(now).astimezone(tz)
    start = datetime.combine(local.date(), parse_hhmm(campaign.send_window_start), tzinfo=tz)
    end = datetime.combine(local.date(), parse_hhmm(campaign.send_window_end), tzinfo=tz)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def window_open(campaign: Campaign, now: Optional[datetime] = None) -> bool:
    now = _aware(now)
    if not campaign.send_on_weekends and now.astimezone(tz_of(campaign)).weekday() >= 5:
        return False
    start, end = window_bounds(campaign, now)
    return start <= now < end


def next_window_open(campaign: Campaign, now: Optional[datetime] = None) -> Optional[datetime]:
    """When sending can next happen, in UTC: ``now`` if the window is open."""
    now = _aware(now)
    if window_open(campaign, now):
        return now
    tz = tz_of(campaign)
    today = now.astimezone(tz).date()
    for offset in range(8):
        day = today + timedelta(days=offset)
        if not campaign.send_on_weekends and day.weekday() >= 5:
            continue
        start = datetime.combine(day, parse_hhmm(campaign.send_window_start), tzinfo=tz).astimezone(timezone.utc)
        if start > now:
            return start
    return None


def send_cap(campaign: Campaign) -> int:
    """How many approved emails may go out today.

    Only spam protection holds an approved email back, never the time of day:
    the warm-up ramp, capped by the campaign's daily limit. A new mailbox that
    jumps straight to full volume gets its mail filtered as spam.
    """
    return ramp_cap(campaign)


def ramp_cap(campaign: Campaign) -> int:
    """Today's campaign ceiling: 5 on day 1, +5 a day, up to the campaign's limit."""
    return max(0, min(campaign.daily_send_limit, WARMUP_STEP * max(1, campaign.warmup_day)))


# ---------------------------------------------------------------------------
# Daily log
# ---------------------------------------------------------------------------
async def daily_log(db: AsyncSession, campaign: Campaign, day: date) -> CampaignDailyLog:
    row = (
        await db.execute(
            select(CampaignDailyLog)
            .where(CampaignDailyLog.campaign_id == campaign.id, CampaignDailyLog.date == day)
            .limit(1)
        )
    ).scalars().first()
    if row is None:
        row = CampaignDailyLog(
            campaign_id=campaign.id, date=day,
            prospects_found=0, emails_sent=0, emails_bounced=0, errors=[], discovery_runs=0,
        )
        db.add(row)
        await db.flush()
    return row


async def log_error(
    db: AsyncSession, campaign: Campaign, message: str, now: Optional[datetime] = None
) -> None:
    row = await daily_log(db, campaign, local_today(campaign, now))
    row.errors = [*(row.errors or []), {"at": _aware(now).isoformat(), "error": message[:500]}][-50:]
    logger.warning("Campaign %s: %s", campaign.id, message)


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------
class CompanySize(BaseModel):
    min: Optional[int] = None
    max: Optional[int] = None


class Icp(BaseModel):
    """The ideal-customer profile the AI derives from the campaign's documents."""

    summary: str = ""
    offering: str = ""
    value_props: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    target_locations: list[str] = Field(default_factory=list)
    company_size: CompanySize = Field(default_factory=CompanySize)
    buyer_titles: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    search_query: str = ""

    @field_validator(
        "value_props", "target_industries", "target_locations", "buyer_titles", "keywords",
        mode="before",
    )
    @classmethod
    def _list(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip()]
        return v if isinstance(v, list) else []

    @field_validator("company_size", mode="before")
    @classmethod
    def _size(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else {}


ICP_SYSTEM = (
    "You are a B2B go-to-market strategist. From the documents a company uploaded, "
    "work out what it sells and who should buy it. Be specific and practical. "
    "Answer with a single JSON object and nothing else."
)

DOC_TEXT_LIMIT = 12000


async def _document_text(db: AsyncSession, doc: DiscoveryDocument) -> str:
    """Readable text for one document: the file if it's still held, else what was extracted."""
    import asyncio

    from app.services import document_extract

    parts: list[str] = []
    # ``content`` is deferred; touching ``doc.content`` would lazy-load it, which
    # an async session cannot do (MissingGreenlet). Select the column explicitly.
    content = (
        await db.execute(select(DiscoveryDocument.content).where(DiscoveryDocument.id == doc.id))
    ).scalar_one_or_none()
    if content:
        try:
            parsed = await asyncio.to_thread(document_extract.parse_document, doc.file_type, content)
            if parsed.kind == "table":
                parts.append(" | ".join(parsed.headers))
                parts.extend(" | ".join(r) for r in parsed.rows[:40])
            else:
                parts.extend(text for _ref, text in parsed.chunks)
        except Exception as exc:  # noqa: BLE001 — fall back to the extraction
            logger.info("Campaign doc %s unreadable (%s); using extraction", doc.id, exc)
    if not parts:
        for cand in ((doc.extracted or {}).get("candidates") or [])[:60]:
            bits = [cand.get("name"), cand.get("industry"), cand.get("city"), cand.get("country")]
            parts.append(", ".join(b for b in bits if b))
    text = "\n".join(p for p in parts if p).strip()
    return f"### {doc.filename}\n{text[:DOC_TEXT_LIMIT]}" if text else ""


def icp_prompt(doc_texts: list[str]) -> str:
    body = "\n\n".join(doc_texts)[: DOC_TEXT_LIMIT * 2]
    return (
        "Documents:\n" + body + "\n\n"
        "If these documents describe the sender's own product or service, profile its "
        "ideal buyers. If they are a list of companies or leads, profile what those "
        "companies have in common so we can find more like them.\n\n"
        "Return exactly this JSON shape:\n"
        '{"summary": str (2-3 sentences: what the sender offers and to whom), '
        '"offering": str (one line), "value_props": [str], '
        '"target_industries": [str], "target_locations": [str], '
        '"company_size": {"min": int|null, "max": int|null}, '
        '"buyer_titles": [str], "keywords": [str], '
        '"search_query": str (one sentence describing the companies to find, e.g. '
        '"Mid-size injection molding manufacturers in Ohio that supply automotive OEMs")}'
    )


async def build_icp(
    db: AsyncSession, org_id: uuid.UUID, docs: list[DiscoveryDocument]
) -> tuple[Icp, str]:
    """``(icp, document_summary)``. Raises ``org_gemini.GeminiKeyMissing`` / ``GeminiError``."""
    texts = [t for t in [await _document_text(db, d) for d in docs] if t]
    if not texts:
        raise CampaignError("documents_empty", "These documents have no readable text to build a profile from.")
    icp = await org_gemini.generate_json(
        db, org_id, system=ICP_SYSTEM, prompt=icp_prompt(texts), schema=Icp, max_tokens=4096,
    )
    return icp, (icp.summary or icp.offering or "")[:4000]


# ---------------------------------------------------------------------------
# Queueing prospects
# ---------------------------------------------------------------------------
async def previously_mailed(db: AsyncSession, emails: Iterable[str]) -> set[str]:
    """Addresses (lower-case) this org has already put mail on the wire to, by any path."""
    emails = {e.lower() for e in emails if e}
    if not emails:
        return set()
    direct = (
        await db.execute(
            select(func.lower(EmailSend.to_email)).where(
                func.lower(EmailSend.to_email).in_(emails),
                EmailSend.status.in_(WIRE_STATUSES),
            )
        )
    ).scalars().all()
    sequence = (
        await db.execute(
            select(func.lower(Contact.email))
            .join(OutreachEnrollment, OutreachEnrollment.contact_id == Contact.id)
            .join(OutreachEvent, OutreachEvent.enrollment_id == OutreachEnrollment.id)
            .where(func.lower(Contact.email).in_(emails), OutreachEvent.event_type == "sent")
        )
    ).scalars().all()
    campaign = (
        await db.execute(
            select(CampaignProspect.to_email).where(
                CampaignProspect.to_email.in_(emails),
                CampaignProspect.status.in_(("sending", *WIRE)),
            )
        )
    ).scalars().all()
    return {e for e in [*direct, *sequence, *campaign] if e}


def canonical_address(email: Optional[str]) -> Optional[str]:
    """The mailbox an address reaches, for "is this us?" comparisons.

    Strips a ``+tag`` everywhere, and Gmail's ignored dots: the E2E campaign's
    test contact ``suman.inster18@gmail.com`` *was* the sending mailbox
    ``sumaninster18@gmail.com``, and a plain string compare said otherwise.
    """
    if not email or "@" not in email:
        return None
    local, _, domain = email.strip().lower().rpartition("@")
    local = local.split("+", 1)[0]
    if domain == "googlemail.com":
        domain = "gmail.com"
    if domain == "gmail.com":
        local = local.replace(".", "")
    return f"{local}@{domain}" if local and domain else None


async def own_addresses(db: AsyncSession) -> tuple[set[str], set[str]]:
    """``(domains, mailboxes)`` that belong to this org: never prospects.

    A document an operator uploads usually carries their own signature — a
    partner brief names the author — and nothing stopped that address being
    queued and mailed by the org's own campaign. Sources: every connected
    sending account (Gmail and SMTP) and every user of the org. A work domain
    excludes the whole domain (a colleague is not a prospect); a free-mail
    host can't — everyone at gmail.com is not one company — so for those only
    the exact mailbox is excluded, canonicalised.
    """
    from app.dependencies import scope_users_to_current_org
    from app.models.smtp_account import SmtpAccount
    from app.services.crm_import import FREE_EMAIL_DOMAINS

    emails = [
        *(await db.execute(select(GmailAccount.email))).scalars().all(),
        *(await db.execute(select(SmtpAccount.email))).scalars().all(),
        *(await db.execute(scope_users_to_current_org(select(User.email)))).scalars().all(),
    ]
    domains: set[str] = set()
    mailboxes: set[str] = set()
    for e in emails:
        canon = canonical_address(e)
        if not canon:
            continue
        domain = canon.rpartition("@")[2]
        if domain in FREE_EMAIL_DOMAINS or domain == "googlemail.com":
            mailboxes.add(canon)
        else:
            domains.add(domain)
    return domains, mailboxes


def is_own_address(email: Optional[str], own: tuple[set[str], set[str]]) -> bool:
    """Whether ``email`` is one of :func:`own_addresses`' ``(domains, mailboxes)``."""
    canon = canonical_address(email)
    if not canon:
        return False
    domains, mailboxes = own
    return canon in mailboxes or canon.rpartition("@")[2] in domains


async def queue_contacts(
    db: AsyncSession, campaign: Campaign, contacts: list[Contact], *, source: str
) -> int:
    """Queue contacts as prospects. Returns how many were newly queued.

    Silently passes over anyone without an email, on the do-not-contact list or
    flagged, already a prospect of *any* campaign, already mailed by any path,
    or belonging to this org itself (:func:`own_addresses`).
    """
    queued, _ = await queue_contacts_detailed(db, campaign, contacts, source=source)
    return queued


# Why queue_contacts_detailed passed a contact over — shown to the operator by
# the add-to-campaign endpoints, so "nothing happened" always has a reason.
SKIP_REASONS = {
    "no_email": "No email address",
    "blocked": "On the do-not-contact list or unsubscribed",
    "own_org": "Your own organization's address",
    "already_prospect": "Already a prospect in a campaign",
    "already_mailed": "Already emailed — never sent twice",
    "duplicate": "Listed twice",
}


async def queue_contacts_detailed(
    db: AsyncSession, campaign: Campaign, contacts: list[Contact], *, source: str
) -> tuple[int, dict[uuid.UUID, str]]:
    """:func:`queue_contacts`, also returning ``{contact_id: SKIP_REASONS key}``."""
    own = await own_addresses(db)
    skipped: dict[uuid.UUID, str] = {}
    candidates: dict[str, Contact] = {}
    for c in contacts:
        email = (c.email or "").strip().lower()
        if not email:
            skipped[c.id] = "no_email"
        elif suppression.blocking_flag(c) is not None:
            skipped[c.id] = "blocked"
        elif email in candidates:
            skipped[c.id] = "duplicate"
        elif is_own_address(email, own):
            logger.info("Campaign %s: not queueing %s — it is this org's own address", campaign.id, email)
            skipped[c.id] = "own_org"
        else:
            candidates[email] = c
    if not candidates:
        return 0, skipped
    emails = set(candidates)
    blocked = await suppression.suppressed_addresses(db, emails)
    mailed = await previously_mailed(db, emails)
    prospects = set(
        (
            await db.execute(
                select(CampaignProspect.to_email).where(CampaignProspect.to_email.in_(emails))
            )
        ).scalars().all()
    )
    queued = 0
    for email, contact in candidates.items():
        if email in blocked:
            skipped[contact.id] = "blocked"
            continue
        if email in mailed:
            skipped[contact.id] = "already_mailed"
            continue
        if email in prospects:
            skipped[contact.id] = "already_prospect"
            continue
        db.add(CampaignProspect(
            campaign_id=campaign.id,
            company_id=contact.company_id,
            contact_id=contact.id,
            to_email=email,
            status="queued",
            source=source,
        ))
        queued += 1
    if queued:
        await db.flush()
    return queued, skipped


async def contact_block_reason(
    db: AsyncSession, contact: Contact, *, exclude_prospect_id: Optional[uuid.UUID] = None
) -> Optional[str]:
    """Why ``contact`` can't be a prospect (a :data:`SKIP_REASONS` key), or None.

    The same rules :func:`queue_contacts_detailed` applies when a contact is
    first queued, for a prospect whose contact was edited in place — a changed
    address has to clear do-not-contact, own-org and "never twice" again.
    """
    email = (contact.email or "").strip().lower()
    if not email:
        return "no_email"
    if suppression.blocking_flag(contact) is not None or email in await suppression.suppressed_addresses(db, {email}):
        return "blocked"
    if is_own_address(email, await own_addresses(db)):
        return "own_org"
    if email in await previously_mailed(db, [email]):
        return "already_mailed"
    stmt = select(CampaignProspect.id).where(CampaignProspect.to_email == email)
    if exclude_prospect_id is not None:
        stmt = stmt.where(CampaignProspect.id != exclude_prospect_id)
    if (await db.execute(stmt.limit(1))).first() is not None:
        return "already_prospect"
    return None


UNSENT = ("queued", "draft_created", "approved", "no_email", "skipped", "failed")


def reset_for_redraft(p: CampaignProspect) -> None:
    """Back to ``queued`` with no draft: the next pass verifies and drafts it again."""
    p.status = "queued"
    p.email_subject = p.email_body = None
    p.template_id = None
    p.approved_at = p.scheduled_for = None
    p.skip_reason = p.error = None


async def redraft(db: AsyncSession, campaign: Campaign) -> int:
    """Discard the unsent, unapproved drafts so they are written again.

    For after the campaign's template (or its mode) changes: drafts written
    before the change don't follow it. Approved emails are left alone — the
    operator signed off on that copy — and nothing sent is ever touched.
    """
    rows = (
        await db.execute(
            select(CampaignProspect).where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status == "draft_created",
            )
        )
    ).scalars().all()
    for p in rows:
        reset_for_redraft(p)
    await db.flush()
    return len(rows)


async def add_contacts(
    db: AsyncSession, campaign: Campaign, contacts: list[Contact], *, source: str
) -> dict:
    """Put chosen contacts into ``campaign`` — from the contact database or a manual add.

    A contact with no email but a name and a company domain is first given one
    the way discovery does (MX check, Hunter finder, guess); every address then
    goes through Hunter verification on the campaign's next pass, and only a
    deliverable one is ever sent. Returns ``{"queued", "skipped": [{contact_id,
    email, reason, label}]}``.
    """
    targets: list[tuple[Company, Contact]] = []
    for c in contacts:
        if c.email or not plausible_person_name(c.first_name, c.last_name):
            continue
        company = await db.get(Company, c.company_id) if c.company_id else None
        if company is not None and company.domain:
            targets.append((company, c))
    if targets:
        await _address_contacts(db, targets)
    queued, skipped = await queue_contacts_detailed(db, campaign, contacts, source=source)
    by_id = {c.id: c for c in contacts}
    return {
        "queued": queued,
        "skipped": [
            {"contact_id": str(cid), "email": by_id[cid].email, "reason": r, "label": SKIP_REASONS[r]}
            for cid, r in skipped.items()
        ],
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
async def _seed_from_documents(db: AsyncSession, campaign: Campaign) -> int:
    """Queue the people the campaign's documents name.

    Contacts the document gives an address for are queued as they are. A
    company the document names people at but gives no address for gets one of
    those people — the first plausible name — run through the same
    finder → guess → verify path as search results. These are the operator's
    own leads, written down by them, and they used to be dropped for want of
    an email.
    """
    from app.services import discovery_documents

    ids = [uuid.UUID(str(i)) for i in campaign.document_ids or []]
    if not ids:
        return 0
    docs = (
        await db.execute(select(DiscoveryDocument).where(DiscoveryDocument.id.in_(ids)))
    ).scalars().all()
    contact_ids: set[str] = set()
    named: list[tuple[uuid.UUID, str, Optional[str]]] = []
    for doc in docs:
        cands = [
            c for c in ((doc.extracted or {}).get("candidates") or [])
            if any(p.get("email") or p.get("first_name") for p in c.get("contacts", []))
        ]
        if not cands:
            continue
        result = await discovery_documents.import_candidates(db, doc, [c["id"] for c in cands])
        contact_ids |= set(result.get("contact_ids") or [])
        wanted = {c["id"] for c in cands}
        # import_candidates stamps imported_company_id on the stored candidates.
        for cand in (doc.extracted or {}).get("candidates") or []:
            people = cand.get("contacts", [])
            if cand["id"] not in wanted or not cand.get("imported_company_id"):
                continue
            if any(p.get("email") for p in people):
                continue  # the document already gives this company an address
            person = next(
                (p for p in people if plausible_person_name(p.get("first_name"), p.get("last_name"))),
                None,
            )
            if person is not None:
                named.append((uuid.UUID(cand["imported_company_id"]),
                              person["first_name"], person.get("last_name")))

    contacts = list(
        (
            await db.execute(
                select(Contact).where(Contact.id.in_([uuid.UUID(c) for c in contact_ids]))
            )
        ).scalars().all()
    ) if contact_ids else []

    targets: list[tuple[Company, Contact]] = []
    for company_id, first, last in named:
        company = await db.get(Company, company_id)
        if company is None or not company.domain:
            continue
        # The person the document named, not whoever is oldest at a company the
        # import matched to an existing row.
        stmt = select(Contact).where(
            Contact.company_id == company_id,
            func.lower(Contact.first_name) == first.strip().lower(),
        )
        if last:
            stmt = stmt.where(func.lower(Contact.last_name) == last.strip().lower())
        contact = (
            await db.execute(stmt.order_by(Contact.created_at, Contact.id).limit(1))
        ).scalars().first()
        if contact is None:
            continue
        if contact.email:
            contacts.append(contact)
        else:
            targets.append((company, contact))
    if targets:
        _, addressed = await _address_contacts(db, targets)
        contacts.extend(addressed)
    return await queue_contacts(db, campaign, contacts, source="document")


def guess_email(first_name: Optional[str], domain: Optional[str]) -> Optional[str]:
    """``firstname@domain`` — the commonest B2B pattern — or None if either half is unusable.

    Discovery LLMs name people reliably and their addresses almost never, so a
    contact arrives with a name and no email. The guess is unverified: it is left
    with ``email_status`` NULL so the send path's MX/verification checks still run.
    """
    if not first_name or not domain:
        return None
    d = domain.strip().lower()
    d = re.sub(r"^[a-z]+://", "", d).split("/", 1)[0].split(":", 1)[0]
    d = d.removeprefix("www.").strip(".")
    if "." not in d or not re.fullmatch(r"[a-z0-9.-]+", d):
        return None
    tokens = first_name.strip().split()
    if not tokens:
        return None
    ascii_name = unicodedata.normalize("NFKD", tokens[0]).encode("ascii", "ignore").decode()
    local = re.sub(r"[^a-z0-9]", "", ascii_name.lower())
    if not local:
        return None
    return f"{local}@{d}"


# Words an LLM puts in a name field when it has no name: "John K. (Placeholder
# for CTO)" cost a Hunter search and produced john@veeva.com.
_PLACEHOLDER_WORDS = {
    "placeholder", "unknown", "tbd", "n/a", "na", "none", "null", "example",
    "contact", "name", "firstname", "lastname", "redacted", "anonymous",
    "ceo", "cto", "cfo", "coo", "cio", "ciso", "vp", "head", "director",
    "manager", "founder", "owner", "team", "sales", "info", "admin", "support",
}
_NOT_A_NAME = re.compile(r"[()\[\]{}<>@#/\\|_:;!?*=+0-9]")


def plausible_person_name(first_name: Optional[str], last_name: Optional[str]) -> bool:
    """Whether a discovered contact's name could be a real person's.

    A discovered "contact" is only worth a Hunter search, or a guessed address,
    if it names someone. Rejects brackets/digits/emails, placeholder and
    job-title words, and the textbook dummies.
    """
    first = (first_name or "").strip()
    if len(first.strip(".")) < 2:
        return False
    full = f"{first} {(last_name or '').strip()}".strip()
    if _NOT_A_NAME.search(full):
        return False
    words = {w.strip(".,'-").lower() for w in full.split()}
    if words & _PLACEHOLDER_WORDS:
        return False
    return full.lower() not in {"john doe", "jane doe", "john smith"}


# A prospect is only ever scheduled once its contact's address carries this
# verdict. Pattern guesses and finder results alike start unverified.
VERIFIED = "deliverable"
VERIFY_PER_TICK = 5


def _stamp_origin(contact: Contact, origin: str) -> None:
    meta = dict(contact.email_verification_meta or {})
    meta["email_origin"] = origin
    contact.email_verification_meta = meta


async def _resolve_missing_emails(
    db: AsyncSession, company_ids: list[uuid.UUID], *, grounded: bool = True
) -> dict[str, int]:
    """Give one named contact per address-less company an address, then verify it.

    Discovery LLMs name people reliably and their addresses almost never. Only
    one contact per company is tried, because only one per company is ever
    queued and Hunter's free tier is a few dozen finds a month. ``grounded`` is
    False when the search answered from training data (no web access); see
    :func:`_address_contacts`.
    """
    stats = {"found": 0, "guessed": 0, "verified": 0, "domain_searched": 0, "no_mx": 0}
    companies = (
        await db.execute(
            select(Company).where(Company.id.in_(company_ids), Company.domain.isnot(None))
        )
    ).scalars().all()
    if not companies:
        return stats
    has_email = set(
        (
            await db.execute(
                select(Contact.company_id).where(
                    Contact.company_id.in_([c.id for c in companies]),
                    Contact.email.isnot(None), Contact.email != "",
                )
            )
        ).scalars().all()
    )
    targets: list[tuple[Company, Contact]] = []
    nameless: list[Company] = []
    for company in companies:
        if company.id in has_email:
            continue
        named = (
            await db.execute(
                select(Contact)
                .where(Contact.company_id == company.id, Contact.first_name.isnot(None))
                .order_by(Contact.created_at, Contact.id)
            )
        ).scalars().all()
        contact = next(
            (c for c in named if plausible_person_name(c.first_name, c.last_name)), None
        )
        if contact is not None:
            targets.append((company, contact))
        else:
            if named:
                logger.info(
                    "Campaign discovery: %s has no plausible contact name (%r) — trying domain search",
                    company.domain, [f"{c.first_name} {c.last_name or ''}".strip() for c in named][:3],
                )
            nameless.append(company)
    if not targets and not nameless:
        return stats
    from app.services import hunter_verification

    hunter = hunter_verification.get_hunter_service()
    budget = await hunter.make_budget(db) if hunter.enabled else None
    if targets:
        stats, _ = await _address_contacts(db, targets, grounded=grounded, budget=budget)
    if nameless and hunter.enabled:
        found = await _people_by_domain(db, nameless, hunter=hunter, budget=budget)
        stats["domain_searched"] += found["searched"]
        stats["found"] += found["found"]
        stats["verified"] += found["verified"]
        stats["no_mx"] += found["no_mx"]
    return stats


async def _people_by_domain(
    db: AsyncSession, companies: list[Company], *, hunter, budget
) -> dict[str, int]:
    """One Hunter domain search per company that has a domain and nobody named.

    Web results name companies far more often than people — of the first web
    run's seven companies only one came with a person — and without a person
    the finder has nothing to search for, so the company was recorded as
    ``no_email`` and given up on. A domain search (one credit) returns the
    people Hunter knows at that domain; the best *personal* address with a name
    becomes the company's contact and is verified like any other. A generic
    ``info@`` is not taken: it lands in a shared inbox, and catch-all domains
    are exactly where those come back "deliverable".
    """
    from app.services import email_validation

    out = {"searched": 0, "found": 0, "verified": 0, "no_mx": 0}
    for company in companies:
        if await asyncio.to_thread(email_validation.check_mx, company.domain) == "invalid":
            out["no_mx"] += 1
            continue
        out["searched"] += 1
        result = await hunter.domain_search(db, company.domain, budget=budget)
        person = next(
            (
                e for e in result.get("emails") or []
                if e.get("type") == "personal" and e.get("email")
                and plausible_person_name(e.get("first_name"), e.get("last_name"))
            ),
            None,
        )
        if person is None:
            continue
        email = person["email"].strip().lower()
        if (await db.execute(
            select(Contact.id).where(func.lower(Contact.email) == email).limit(1)
        )).first() is not None:
            continue  # someone already holds this address; contacts dedup by email
        contact = Contact(
            company_id=company.id, first_name=person.get("first_name"),
            last_name=person.get("last_name"), title=person.get("position"),
            email=email, source="hunter_domain_search",
        )
        db.add(contact)
        await db.flush()
        out["found"] += 1
        verdict = await hunter.verify_contact(db, contact, budget=budget)
        if verdict.get("result") == VERIFIED:
            out["verified"] += 1
        _stamp_origin(contact, "hunter_domain_search")
    await db.flush()
    if out["searched"]:
        logger.info("Campaign discovery domain search: %s", out)
    return out


async def _address_contacts(
    db: AsyncSession, targets: list[tuple[Company, Contact]], *, grounded: bool = True,
    budget=None,
) -> tuple[dict[str, int], list[Contact]]:
    """Find, guess or replace an address for each ``(company, contact)``, then verify.

    Returns stats and the contacts that ended up with an address.

    * **MX first.** A domain that can't receive mail is skipped before any
      Hunter call — the finder spends a credit on a dead domain just as
      readily as on a live one, and a guess there is a certain bounce. Only a
      definite ``invalid`` skips: ``unknown`` (DNS down, dnspython missing) says
      nothing about the domain.
    * **No-web results with a first name only get a domain search.** A search
      answered from training data invents its people: the E2E campaign's
      Groq run named "David", "Sarah"… and nine of ten guesses came back
      undeliverable. The finder can't find an invented person, so instead one
      domain-search credit returns the people Hunter knows at that domain, and
      the best one *replaces* the invented contact. Nothing found means no
      address — no ``firstname@`` guess off a made-up first name.
    * Otherwise Hunter's Email Finder, falling back to a ``firstname@domain``
      guess. Either way Hunter's verifier decides whether it is sendable.

    Every Hunter call fails open: no key, no budget or an outage leaves the
    address unverified, which ``schedule_sends`` will not send.
    """
    from app.services import email_validation, hunter_verification

    stats = {"found": 0, "guessed": 0, "verified": 0, "domain_searched": 0, "no_mx": 0}
    addressed: list[Contact] = []
    hunter = hunter_verification.get_hunter_service()
    if budget is None and hunter.enabled:
        budget = await hunter.make_budget(db)
    for company, contact in targets:
        if await asyncio.to_thread(email_validation.check_mx, company.domain) == "invalid":
            stats["no_mx"] += 1
            logger.info("Campaign discovery: %s has no mail server — not searching", company.domain)
            continue
        email, origin = None, "pattern"
        if hunter.enabled and not grounded and not (contact.last_name or "").strip():
            stats["domain_searched"] += 1
            found = await hunter.domain_search(db, company.domain, budget=budget)
            best = (found.get("emails") or [None])[0]
            if not best or not best.get("email") or await _address_taken(db, best["email"], contact.id):
                continue
            email, origin = best["email"], "hunter_domain_search"
            if best.get("first_name"):
                contact.first_name = best["first_name"]
                contact.last_name = best.get("last_name")
                contact.title = best.get("position") or contact.title
        elif hunter.enabled:
            email = await hunter.find_email(
                db, company, contact.first_name, contact.last_name, budget=budget
            )
            if email:
                origin = "hunter_finder"
        if email:
            contact.email = email.strip().lower()
            stats["found"] += 1
        else:
            guess = guess_email(contact.first_name, company.domain)
            if not guess:
                continue
            contact.email = guess
            stats["guessed"] += 1
        if hunter.enabled:
            verdict = await hunter.verify_contact(db, contact, budget=budget)
            if verdict.get("result") == VERIFIED:
                stats["verified"] += 1
        # After verifying: a verdict replaces the meta wholesale.
        _stamp_origin(contact, origin)
        addressed.append(contact)
    await db.flush()
    logger.info("Campaign discovery emails: %s", stats)
    return stats, addressed


async def _address_taken(db: AsyncSession, email: str, contact_id: uuid.UUID) -> bool:
    """Whether another contact already holds this address (contacts dedup by email)."""
    return (
        await db.execute(
            select(Contact.id).where(
                func.lower(Contact.email) == email.strip().lower(), Contact.id != contact_id
            ).limit(1)
        )
    ).first() is not None


async def _record_emailless(
    db: AsyncSession, campaign: Campaign, company_ids: list[uuid.UUID], *, source: str
) -> int:
    """Keep a ``no_email`` row per discovered company we could not address.

    So the dashboard shows what discovery found, not just what it could mail.
    Never sent, never drafted (it is outside ``PENDING``), and replaced by a
    real prospect the day an address turns up.
    """
    if not company_ids:
        return 0
    known = set(
        (
            await db.execute(
                select(CampaignProspect.company_id).where(
                    CampaignProspect.campaign_id == campaign.id,
                    CampaignProspect.company_id.in_(company_ids),
                )
            )
        ).scalars().all()
    )
    added = 0
    for cid in company_ids:
        if cid in known:
            continue
        contact = (
            await db.execute(
                select(Contact).where(Contact.company_id == cid)
                .order_by(Contact.created_at, Contact.id).limit(1)
            )
        ).scalars().first()
        db.add(CampaignProspect(
            campaign_id=campaign.id, company_id=cid,
            contact_id=contact.id if contact else None,
            to_email=None, status="no_email", source=source,
            skip_reason="No email address found for this company yet",
        ))
        added += 1
    if added:
        await db.flush()
    return added


# Excluded from the next search. The model is shown the 40 most recent; the
# whole list filters what comes back.
KNOWN_DOMAINS_LIMIT = 200
# Company.source values a discovery search writes ("tavily+gemini" style web
# results are matched by their "+").
DISCOVERY_SOURCES = ("gemini", "anthropic", "groq", "openrouter")


async def _known_domains(db: AsyncSession) -> list[str]:
    """Domains this org has already found, most recent first (org-scoped read).

    Two sources: companies a campaign holds a prospect row for, and companies
    a discovery search created whether or not a row followed. The second is
    what stopped the loop in production: 23 companies from early runs left no
    prospect row, so they were never excluded and Groq returned the same
    household names every day.
    """
    held = (
        await db.execute(
            select(Company.domain, func.max(CampaignProspect.created_at).label("seen"))
            .join(CampaignProspect, CampaignProspect.company_id == Company.id)
            .where(Company.domain.isnot(None))
            .group_by(Company.domain)
            .order_by(func.max(CampaignProspect.created_at).desc(), Company.domain)
            .limit(KNOWN_DOMAINS_LIMIT)
        )
    ).all()
    discovered = (
        await db.execute(
            select(Company.domain, Company.created_at)
            .where(
                Company.domain.isnot(None),
                or_(Company.source.in_(DISCOVERY_SOURCES), Company.source.like("%+%")),
            )
            .order_by(Company.created_at.desc(), Company.id)
            .limit(KNOWN_DOMAINS_LIMIT)
        )
    ).all()
    return list(dict.fromkeys(d for d, _ in [*held, *discovered]))[:KNOWN_DOMAINS_LIMIT]


async def _known_company_names(db: AsyncSession) -> list[str]:
    """Names of companies discovery or a campaign already holds, most recent first.

    The name half of :func:`_known_domains`: a company found without a domain is
    stored without one, so excluding by domain alone let it come straight back
    (Razorpay, on the first web-search run).
    """
    held = (
        await db.execute(
            select(Company.name)
            .join(CampaignProspect, CampaignProspect.company_id == Company.id)
            .group_by(Company.name)
            .order_by(func.max(CampaignProspect.created_at).desc(), Company.name)
            .limit(KNOWN_DOMAINS_LIMIT)
        )
    ).scalars().all()
    discovered = (
        await db.execute(
            select(Company.name)
            .where(or_(Company.source.in_(DISCOVERY_SOURCES), Company.source.like("%+%")))
            .order_by(Company.created_at.desc(), Company.id)
            .limit(KNOWN_DOMAINS_LIMIT)
        )
    ).scalars().all()
    return list(dict.fromkeys(n for n in [*held, *discovered] if n))[:KNOWN_DOMAINS_LIMIT]


async def _search_companies(db: AsyncSession, campaign: Campaign) -> int:
    from app.services import gemini_discovery
    from app.services.ingestion import get_ingestion_service

    icp = Icp.model_validate(campaign.icp_json or {})
    query = icp.search_query or icp.summary
    # The same ICP asked again returns the same companies — and an ungrounded
    # model returns the same household names every day — all of which
    # queue_contacts then drops as already prospected, so the campaign stalls
    # after day one. Tell the model what it has already given us.
    seen = await _known_domains(db)
    found = await gemini_discovery.search_companies(
        db,
        campaign.org_id,
        query=query,
        industry=", ".join(icp.target_industries[:3]),
        location=", ".join(icp.target_locations[:3]),
        keywords=icp.keywords[:5],
        min_employees=icp.company_size.min,
        max_employees=icp.company_size.max,
        limit=DISCOVERY_COMPANIES,
        exclude_domains=seen,
        exclude_names=await _known_company_names(db),
        offering=icp.offering or icp.summary,
    )
    result = await get_ingestion_service().ingest(db, found.records)
    company_ids = [o.company_id for o in result.outcomes if o.company_id]
    if not company_ids:
        return 0
    company_ids = list(dict.fromkeys(company_ids))
    await _resolve_missing_emails(db, company_ids, grounded=getattr(found, "grounded", True))
    rows = (
        await db.execute(
            select(Contact)
            .where(Contact.company_id.in_(company_ids), Contact.email.isnot(None), Contact.email != "")
            .order_by(Contact.created_at, Contact.id)
        )
    ).scalars().all()
    # One person per company: a cold campaign that mails three people at the
    # same firm on the same day reads as a blast to all three.
    first: dict[uuid.UUID, Contact] = {}
    for c in rows:
        first.setdefault(c.company_id, c)
    # A company recorded as email-less on an earlier day now has an address:
    # drop the placeholder so the real prospect can take its place.
    for stale in (
        await db.execute(
            select(CampaignProspect).where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status == "no_email",
                CampaignProspect.company_id.in_(list(first)),
            )
        )
    ).scalars().all():
        await db.delete(stale)
    await db.flush()
    queued = await queue_contacts(db, campaign, list(first.values()), source=found.provider)
    await _record_emailless(
        db, campaign, [cid for cid in company_ids if cid not in first], source=found.provider
    )
    return queued


async def discover_prospects(
    db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None
) -> int:
    """One day's discovery. Returns prospects queued. Raises the org_gemini errors."""
    seeded = 0
    if campaign.last_discovery_on is None:
        seeded = await _seed_from_documents(db, campaign)
    searched = await _search_companies(db, campaign)
    total = seeded + searched
    campaign.last_discovery_on = local_today(campaign, now)
    log = await daily_log(db, campaign, local_today(campaign, now))
    log.prospects_found = (log.prospects_found or 0) + total
    log.discovery_runs = (log.discovery_runs or 0) + 1
    log.last_discovery_at = _aware(now)
    return total


# Searches the tick runs on its own in a day: the first, then top-ups while the
# campaign has fewer prospects in hand than it can send in a day. A day whose
# search came back mostly undeliverable or email-less used to sit idle until
# tomorrow ("it is not searching more contacts").
AUTO_DISCOVERY_RUNS = 10
DISCOVERY_TOPUP_GAP = timedelta(hours=2)
# Including the operator's "Find more prospects" — each search spends AI quota.
# Kept above AUTO_DISCOVERY_RUNS so the button still works after the tick's own runs.
MANUAL_DISCOVERY_RUNS = 13


async def discovery_due(db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None) -> bool:
    """Whether this tick should search for more companies."""
    now = _aware(now)
    pending = await _pending_count(db, campaign)
    if campaign.last_discovery_on != local_today(campaign, now):
        return pending < 2 * campaign.daily_send_limit
    if pending >= campaign.daily_send_limit:
        return False
    log = await _todays_log(db, campaign, now)
    if log is None or (log.discovery_runs or 0) >= AUTO_DISCOVERY_RUNS:
        return False
    # A day logged before searches were counted ran its search when the log opened.
    last = log.last_discovery_at or log.created_at
    return last is None or now - _aware(last) >= DISCOVERY_TOPUP_GAP


async def _todays_log(db: AsyncSession, campaign: Campaign, now: Optional[datetime]) -> Optional[CampaignDailyLog]:
    return (
        await db.execute(
            select(CampaignDailyLog).where(
                CampaignDailyLog.campaign_id == campaign.id,
                CampaignDailyLog.date == local_today(campaign, now),
            ).limit(1)
        )
    ).scalars().first()


async def searches_today(db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None) -> int:
    log = await _todays_log(db, campaign, now)
    return (log.discovery_runs or 0) if log is not None else 0


async def _pending_count(db: AsyncSession, campaign: Campaign) -> int:
    return (
        await db.execute(
            select(func.count(CampaignProspect.id)).where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status.in_(PENDING),
            )
        )
    ).scalar_one()


# ---------------------------------------------------------------------------
# Drafting
# ---------------------------------------------------------------------------
class DraftEmail(BaseModel):
    subject: str
    body: str


DRAFT_SYSTEM = (
    "You write short, specific, human cold emails for B2B outreach. Plain text, no "
    "markdown, no links, no attachments mentioned, no placeholders like [Name] — "
    "use only the facts given. Never invent facts about the prospect. "
    "Answer with a single JSON object and nothing else."
)

# Used instead of DRAFT_SYSTEM when a template is given. DRAFT_SYSTEM asks for
# "short" emails with no attachments mentioned, which fights a template that is
# long and says "I've attached our brief" — the model sided with the system
# prompt and rewrote the template into a different email.
TEMPLATE_DRAFT_SYSTEM = (
    "You personalise an operator's own email template for one prospect. The template is "
    "the email: keep its subject line, greeting, every paragraph in the same order, its "
    "wording, length, offer, call to action, sign-off and signature block. Change only "
    "what is specific to the prospect: fill the merge fields, and where the template "
    "talks about the prospect's company, make that sentence accurate for them using "
    "only the facts given. Do not add, remove, merge or shorten paragraphs. Plain text, "
    "no markdown, no placeholders like [Name]. Never invent facts about the prospect. "
    "Answer with a single JSON object and nothing else."
)

_PLACEHOLDER = re.compile(r"\[[^\]]{2,40}\]|\{\{|\}\}|<[A-Z][A-Z _]{2,}>")


def draft_prompt(campaign: Campaign, contact: Optional[Contact], company: Optional[Company],
                 sender_name: str, sender_org: str,
                 template: Optional["EmailTemplate"] = None) -> str:
    icp = Icp.model_validate(campaign.icp_json or {})
    who = " ".join(p for p in [getattr(contact, "first_name", None), getattr(contact, "last_name", None)] if p)
    lines = [
        f"Sender: {sender_name or 'the sender'}" + (f" at {sender_org}" if sender_org else ""),
        f"What the sender offers: {campaign.document_summary or icp.summary or icp.offering}",
    ]
    if icp.value_props:
        lines.append("Value: " + "; ".join(icp.value_props[:4]))
    lines.append("")
    lines.append(f"Prospect: {who or 'unknown name'}" + (f", {contact.title}" if contact and contact.title else ""))
    if company is not None:
        place = ", ".join(p for p in [company.city, company.state_province] if p)
        lines.append(
            f"Company: {company.name}"
            + (f" ({company.industry_segment})" if company.industry_segment else "")
            + (f", {place}" if place else "")
            + (f", ~{company.employee_count} employees" if company.employee_count else "")
        )
        if company.domain:
            lines.append(f"Website: {company.domain}")
    if template is not None:
        lines.append("")
        lines.append(
            "Base the email on this template: keep its message, structure, offer and call "
            "to action, and personalise it for this prospect. Fill or drop any {{merge}} "
            "fields; never leave one in the output.\n"
            f"Template subject: {template.subject}\nTemplate body:\n{html_to_text(template.body)}"
        )
        lines.append("")
        lines.append(
            "Return the template as the email: same subject (merge fields filled), same "
            "greeting, same paragraphs in the same order and the same sign-off and signature. "
            "Only adjust the sentences that refer to the prospect or their company. The result "
            "should be about as long as the template, not a shorter rewrite.\n\n"
            "Return: {\"subject\": str, \"body\": str}"
        )
        return "\n".join(lines)
    lines.append("")
    lines.append(
        "Write a first-touch cold email: 60-120 words, a subject under 8 words, open with "
        "something relevant to their company, one clear value point, and end with a "
        "low-pressure question. Greet them by first name if known. Sign off with the "
        f"sender's first name only.\n\nReturn: {{\"subject\": str, \"body\": str}}"
    )
    return "\n".join(lines)


_BLOCK_END = re.compile(r"</(p|div|h[1-6]|li|tr)\s*>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(body: str) -> str:
    """A template's HTML body as the plain text campaign drafts are kept in.

    Drafts are edited and previewed as text and turned back into HTML at send
    (:func:`body_to_html`), so a template body is flattened the same way a
    human would read it: paragraphs to blank lines, ``<br>`` to newlines.
    """
    text = _BR.sub("\n", body or "")
    text = _BLOCK_END.sub("\n\n", text)
    text = html.unescape(_TAG.sub("", text))
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


TEMPLATE_MODES = ("ai", "template", "ai_template")


async def campaign_template(db: AsyncSession, campaign: Campaign) -> Optional[EmailTemplate]:
    """The template drafts use, or None (AI mode, or the template is gone/archived)."""
    if campaign.template_mode == "ai" or campaign.template_id is None:
        return None
    t = await db.get(EmailTemplate, campaign.template_id)
    return None if t is None or t.is_archived else t


def render_template_draft(
    template: EmailTemplate, contact: Optional[Contact], company: Optional[Company]
) -> tuple[str, str]:
    """``(subject, body)`` for one prospect: merge fields filled, no AI.

    The same rendering the template router's preview and compose use (sandboxed
    Jinja, the "there" greeting fallback, subject clean-up), so a template reads
    the same everywhere. An unknown merge field renders empty; anything left
    that still looks like a placeholder sends the draft to review.
    """
    from app.services.email_compose import (
        build_merge_context, clean_body, clean_subject, render, subject_context,
    )

    ctx = build_merge_context(contact, company)
    subject = clean_subject(render(template.subject, subject_context(ctx)))
    body = html_to_text(clean_body(render(template.body, ctx)))
    return subject, body


def generalize_draft(text: str, contact: Optional[Contact], company: Optional[Company]) -> str:
    """Turn one prospect's draft back into a template: their details → merge fields.

    Longest first, so "Gart Solutions" is replaced before a first name that
    happens to appear inside it; whole words only, so "Ana" doesn't eat "Analytics".
    """
    pairs = [
        (getattr(company, "name", None), "{{company}}"),
        (getattr(contact, "title", None), "{{contact_title}}"),
        (" ".join(p for p in [getattr(contact, "first_name", None), getattr(contact, "last_name", None)] if p),
         "{{full_name}}"),
        (getattr(contact, "last_name", None), "{{last_name}}"),
        (getattr(contact, "first_name", None), "{{first_name}}"),
    ]
    out = text or ""
    for value, field in sorted(((v.strip(), f) for v, f in pairs if v and v.strip()),
                               key=lambda vf: -len(vf[0])):
        out = re.sub(rf"(?<!\w){re.escape(value)}(?!\w)", field, out, flags=re.IGNORECASE)
    return out


def body_to_html(body: str) -> str:
    paras = [p.strip() for p in re.split(r"\n\s*\n", body.strip()) if p.strip()]
    return "".join(f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in paras)


async def _sender_identity(db: AsyncSession, campaign: Campaign) -> tuple[str, str]:
    name = ""
    if campaign.user_id:
        user = await db.get(User, campaign.user_id)
        name = (user.name if user else "") or ""
    org = await db.get(Organization, campaign.org_id)
    return name, (org.name if org else "") or ""


async def generate_drafts(
    db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None, *, limit: int = DRAFTS_PER_TICK
) -> int:
    """Draft emails for queued prospects. Stops at the first LLM failure (logged).

    With Hunter configured, only prospects whose address verified deliverable
    are drafted. Ordering verify_prospects ahead of this is not enough on its
    own: each takes a few rows per tick and they select different rows, so the
    first E2E campaign drafted five guessed addresses in the same tick that
    Hunter rejected four others — an AI draft per address that can never send.
    Without a key nothing is ever verified, so drafts are written as before and
    wait for one.
    """
    from app.services import hunter_verification

    stmt = select(CampaignProspect).where(
        CampaignProspect.campaign_id == campaign.id, CampaignProspect.status == "queued"
    )
    if hunter_verification.get_hunter_service().enabled:
        stmt = stmt.join(Contact, Contact.id == CampaignProspect.contact_id).where(
            Contact.email_verification_result == VERIFIED
        )
    queued = (
        await db.execute(
            stmt.order_by(CampaignProspect.created_at, CampaignProspect.id).limit(limit)
        )
    ).scalars().all()
    if not queued:
        return 0
    sender_name, sender_org = await _sender_identity(db, campaign)
    template = await campaign_template(db, campaign)
    drafted = 0
    for p in queued:
        contact = await db.get(Contact, p.contact_id) if p.contact_id else None
        company = await db.get(Company, p.company_id) if p.company_id else None
        if contact is None or not contact.email:
            p.status, p.skip_reason = "skipped", "Contact no longer has an email address"
            continue
        try:
            subject, body = await write_draft(
                db, campaign, contact, company, template, campaign.template_mode,
                sender_name, sender_org,
            )
        except (org_gemini.GeminiKeyMissing, org_gemini.GeminiError) as exc:
            msg = str(exc) or "No AI key is saved in Settings → API Keys"
            await log_error(db, campaign, f"Drafting stopped: {msg}", now)
            break
        if not subject or not body:
            await log_error(db, campaign, f"Empty draft for {p.to_email}; will retry", now)
            continue
        p.email_subject, p.email_body = subject, body
        p.template_id = template.id if template is not None else None
        # A leftover "[Company Name]" must never go out on autopilot.
        needs_review = campaign.review_before_send or bool(_PLACEHOLDER.search(subject + body))
        if needs_review:
            p.status = "draft_created"
        else:
            p.status, p.approved_at = "approved", _aware(now)
        drafted += 1
    await db.flush()
    return drafted


async def write_draft(
    db: AsyncSession, campaign: Campaign, contact: Optional[Contact], company: Optional[Company],
    template: Optional[EmailTemplate], mode: str, sender_name: str, sender_org: str,
) -> tuple[str, str]:
    """``(subject, body)`` for one prospect, from ``template`` used per ``mode``.

    ``template`` mode fills the merge fields with no AI call; otherwise the AI
    writes it, personalising ``template`` when one is given. Raises the
    org_gemini errors for the caller to report.
    """
    if template is not None and mode == "template":
        # The operator's own wording: filled in, not rewritten. No AI call.
        subject, body = render_template_draft(template, contact, company)
        return " ".join(subject.split())[:150], body[:4000]
    use_template = template is not None and mode != "ai"
    draft = await org_gemini.generate_json(
        db, campaign.org_id,
        system=TEMPLATE_DRAFT_SYSTEM if use_template else DRAFT_SYSTEM,
        prompt=draft_prompt(campaign, contact, company, sender_name, sender_org,
                            template=template if use_template else None),
        schema=DraftEmail, max_tokens=2048, temperature=0.3 if use_template else 0.6,
    )
    subject, body = " ".join(draft.subject.split())[:150], draft.body.strip()[:4000]
    if use_template:
        # Guard: a personalised template must still read as the template. If the
        # model collapsed it into a much shorter email, use the merge-filled
        # template itself rather than ship something the operator never wrote.
        base_subject, base_body = render_template_draft(template, contact, company)
        if len(body) < 0.7 * len(base_body):
            logger.warning("AI rewrite of template %s was %d/%d chars; using the template as written",
                           template.id, len(body), len(base_body))
            return " ".join(base_subject.split())[:150], base_body[:4000]
    return subject, body


async def redraft_prospect(
    db: AsyncSession, campaign: Campaign, p: CampaignProspect,
    template: Optional[EmailTemplate], mode: str,
) -> None:
    """Rewrite one unsent draft now, from ``template`` (None = the AI writes it).

    For switching templates on a single draft from its preview; the campaign's
    own template is untouched. The draft stays in review, whatever it was.
    """
    contact = await db.get(Contact, p.contact_id) if p.contact_id else None
    company = await db.get(Company, p.company_id) if p.company_id else None
    sender_name, sender_org = await _sender_identity(db, campaign)
    subject, body = await write_draft(
        db, campaign, contact, company, template, mode if template is not None else "ai",
        sender_name, sender_org,
    )
    if not subject or not body:
        raise ValueError("The draft came back empty.")
    p.email_subject, p.email_body = subject, body
    p.template_id = template.id if template is not None else None
    p.status, p.approved_at, p.scheduled_for, p.error = "draft_created", None, None, None
    await db.flush()


# ---------------------------------------------------------------------------
# Outcomes: bounces and replies
# ---------------------------------------------------------------------------
async def reconcile_outcomes(db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None) -> dict:
    sent = (
        await db.execute(
            select(CampaignProspect).where(
                CampaignProspect.campaign_id == campaign.id, CampaignProspect.status == "sent"
            )
        )
    ).scalars().all()
    if not sent:
        return {"bounced": 0, "replied": 0}
    by_email = {p.to_email: p for p in sent if p.to_email}
    by_thread = {p.gmail_thread_id: p for p in sent if p.gmail_thread_id}

    bounced = replied = 0
    bounces = (
        await db.execute(
            select(func.lower(BounceRecord.recipient_email), BounceRecord.processed_at).where(
                func.lower(BounceRecord.recipient_email).in_(list(by_email)),
                BounceRecord.bounce_kind.in_(("hard", "blocked")),
            )
        )
    ).all()
    for email, at in bounces:
        p = by_email.get(email)
        if p is None or p.status != "sent" or (p.sent_at and at and _aware(at) < _aware(p.sent_at)):
            continue
        p.status, p.bounced_at = "bounced", _aware(now)
        bounced += 1

    conds = []
    if by_email:
        conds.append(func.lower(EmailReply.from_email).in_(list(by_email)))
    if by_thread:
        conds.append(EmailReply.gmail_thread_id.in_(list(by_thread)))
    replies = (
        await db.execute(
            select(EmailReply.from_email, EmailReply.gmail_thread_id, EmailReply.received_at)
            .where(or_(*conds))
        )
    ).all()
    for from_email, thread, at in replies:
        p = by_thread.get(thread) or by_email.get((from_email or "").lower())
        if p is None or p.status != "sent" or (p.sent_at and at and _aware(at) < _aware(p.sent_at)):
            continue
        p.status, p.replied_at = "replied", _aware(at) if at else _aware(now)
        replied += 1

    if bounced:
        log = await daily_log(db, campaign, local_today(campaign, now))
        log.emails_bounced = (log.emails_bounced or 0) + bounced
    await db.flush()
    return {"bounced": bounced, "replied": replied}


async def sent_today(db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None) -> int:
    return (
        await db.execute(
            select(func.count(CampaignProspect.id)).where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status.in_(("sending", *WIRE)),
                CampaignProspect.sent_at >= day_start_utc(campaign, now),
            )
        )
    ).scalar_one()


async def enforce_bounce_rate(db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None) -> bool:
    """Pause the campaign if >5% of today's sends bounced. Returns True when it paused."""
    start = day_start_utc(campaign, now)
    bounced = (
        await db.execute(
            select(func.count(CampaignProspect.id)).where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status == "bounced",
                CampaignProspect.bounced_at >= start,
            )
        )
    ).scalar_one()
    if not bounced:
        return False
    sent = max(await sent_today(db, campaign, now), bounced)
    if bounced / sent <= BOUNCE_PAUSE_RATE:
        return False
    campaign.status, campaign.pause_reason = "paused", "bounce_rate"
    await log_error(
        db, campaign,
        f"Paused automatically: {bounced} of {sent} emails sent today bounced "
        f"(over {int(BOUNCE_PAUSE_RATE * 100)}%). Check the prospect list before resuming.",
        now,
    )
    return True


async def advance_warmup(db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None) -> None:
    today = local_today(campaign, now)
    if campaign.warmup_on is None:
        campaign.warmup_on = today
        return
    if today <= campaign.warmup_on:
        return
    log = (
        await db.execute(
            select(CampaignDailyLog.emails_sent)
            .where(CampaignDailyLog.campaign_id == campaign.id, CampaignDailyLog.date == campaign.warmup_on)
            .limit(1)
        )
    ).scalars().first()
    if log:
        campaign.warmup_day = (campaign.warmup_day or 1) + 1
    campaign.warmup_on = today


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------
async def sending_account(db: AsyncSession, campaign: Campaign):
    """The campaign's mailbox — a ``GmailAccount`` or ``SmtpAccount`` — or None
    when it is gone, disconnected or has sending turned off."""
    from app.models.smtp_account import SmtpAccount

    if campaign.gmail_account_id is not None:
        acct = await db.get(GmailAccount, campaign.gmail_account_id)
    elif campaign.smtp_account_id is not None:
        acct = await db.get(SmtpAccount, campaign.smtp_account_id)
    else:
        return None
    if acct is None or acct.status != "connected" or not acct.sending_enabled:
        return None
    return acct


async def sender_summary(db: AsyncSession, campaign: Campaign) -> Optional[dict]:
    """The campaign's mailbox for display, whether or not it can send right now.

    :func:`sending_account` answers "can we send?" and returns None for a
    disconnected mailbox; the preview needs to say *which* mailbox and *why* it
    can't, so this loads the stored account regardless of its state.
    """
    from app.models.smtp_account import SmtpAccount

    if campaign.gmail_account_id is not None:
        kind, acct = "gmail", await db.get(GmailAccount, campaign.gmail_account_id)
    elif campaign.smtp_account_id is not None:
        kind, acct = "smtp", await db.get(SmtpAccount, campaign.smtp_account_id)
    else:
        return None
    if acct is None:
        return None
    problem = None
    if acct.status != "connected":
        problem = "disconnected"
    elif not acct.sending_enabled:
        problem = "sending_off"
    return {
        "id": str(acct.id), "kind": kind, "email": acct.email,
        "display_name": getattr(acct, "display_name", None),
        "ready": problem is None, "problem": problem,
    }


async def verify_prospects(db: AsyncSession, campaign: Campaign) -> dict[str, int]:
    """Verify this campaign's pending prospects ahead of the platform-wide sweep.

    ``verify_pending_contacts`` would get to them eventually, oldest contact
    first across every org; a campaign waiting behind that backlog sends
    nothing. Then retire prospects whose verdict is in and isn't deliverable:
    they can never be scheduled, and left pending they'd hold the discovery
    gate shut (``_pending_count``).
    """
    from app.services import hunter_verification

    out = {"verified": 0, "rejected": 0}
    hunter = hunter_verification.get_hunter_service()
    if hunter.enabled:
        pairs = (
            await db.execute(
                select(CampaignProspect, Contact)
                .join(Contact, Contact.id == CampaignProspect.contact_id)
                .where(
                    CampaignProspect.campaign_id == campaign.id,
                    CampaignProspect.status.in_(PENDING),
                    Contact.email_verified_at.is_(None),
                )
                .order_by(CampaignProspect.created_at, CampaignProspect.id)
                .limit(VERIFY_PER_TICK)
            )
        ).all()
        if pairs:
            budget = await hunter.make_budget(db)
            for _, contact in pairs:
                origin = (contact.email_verification_meta or {}).get("email_origin")
                await hunter.verify_contact(db, contact, budget=budget)
                if origin:
                    _stamp_origin(contact, origin)
                out["verified"] += 1
    rejected = (
        await db.execute(
            select(CampaignProspect, Contact.email_verification_result)
            .join(Contact, Contact.id == CampaignProspect.contact_id)
            .where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status.in_(PENDING),
                Contact.email_verified_at.isnot(None),
                or_(Contact.email_verification_result.is_(None),
                    Contact.email_verification_result != VERIFIED),
            )
        )
    ).all()
    for p, result in rejected:
        p.status = "skipped"
        p.skip_reason = f"Email not verified deliverable (verification: {result or 'none'})"
        out["rejected"] += 1
    await db.flush()
    return out


async def schedule_sends(
    db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None, *, schedule: ScheduleFn
) -> int:
    """Hand this tick's sends to ``schedule(org_id, prospect_id, countdown_seconds)``."""
    now = _aware(now)
    acct = await sending_account(db, campaign)
    if acct is None:
        logger.info("schedule_sends campaign=%s: no sending account (gmail=%s smtp=%s)",
                    campaign.id, campaign.gmail_account_id, campaign.smtp_account_id)
        campaign.status, campaign.pause_reason = "paused", "gmail_disconnected"
        await log_error(
            db, campaign,
            "Paused automatically: the sending account for this campaign is disconnected "
            "or has sending turned off. Reconnect it, then resume.",
            now,
        )
        return 0
    lease_floor = now - SCHEDULE_LEASE
    in_flight = (
        await db.execute(
            select(func.count(CampaignProspect.id)).where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status == "approved",
                CampaignProspect.scheduled_for.isnot(None),
                CampaignProspect.scheduled_for >= lease_floor,
            )
        )
    ).scalar_one()
    cap, already_sent = send_cap(campaign), await sent_today(db, campaign, now)
    quota = cap - already_sent - in_flight
    if quota <= 0:
        logger.info("schedule_sends campaign=%s: no quota (send_cap=%d sent_today=%d in_flight=%d)",
                    campaign.id, cap, already_sent, in_flight)
        return 0
    # Only verified addresses are sent; an unverified one waits for
    # verify_prospects (or the platform sweep) rather than going on the wire.
    unverified = (
        await db.execute(
            select(func.count(CampaignProspect.id))
            .outerjoin(Contact, Contact.id == CampaignProspect.contact_id)
            .where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status == "approved",
                or_(Contact.id.is_(None), Contact.email_verification_result.is_(None),
                    Contact.email_verification_result != VERIFIED),
            )
        )
    ).scalar_one()
    ready = (
        await db.execute(
            select(CampaignProspect)
            .join(Contact, Contact.id == CampaignProspect.contact_id)
            .where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status == "approved",
                Contact.email_verification_result == VERIFIED,
                or_(CampaignProspect.scheduled_for.is_(None), CampaignProspect.scheduled_for < lease_floor),
            )
            .order_by(CampaignProspect.approved_at, CampaignProspect.id)
            .limit(quota)
        )
    ).scalars().all()
    horizon = now + TICK
    at = max(now, _aware(campaign.next_send_at) if campaign.next_send_at else now)
    logger.info(
        "schedule_sends campaign=%s: account=%s, quota=%d, ready=%d, "
        "unverified_held=%d, now=%s, next_send_at=%s, at=%s, horizon=%s",
        campaign.id, acct is not None, quota, len(ready), unverified,
        now, campaign.next_send_at, at, horizon,
    )
    scheduled = 0
    for p in ready:
        if at >= horizon:
            logger.info("schedule_sends campaign=%s: at=%s is past horizon=%s; stopping",
                        campaign.id, at, horizon)
            break
        p.scheduled_for = at
        await db.flush()
        schedule(campaign.org_id, p.id, max(0, int((at - now).total_seconds())))
        scheduled += 1
        at = at + timedelta(seconds=random.randint(*GAP_SECONDS))
    if scheduled:
        campaign.next_send_at = at
    return scheduled


def _is_transient_hold(result: dict) -> bool:
    """Kill switch, warm-up or daily cap: wait, don't give up (see scheduled_sends)."""
    return not result.get("spam_blocked") and result.get("status") in ("halted", "skipped")


async def _attachments(db: AsyncSession, campaign: Campaign) -> list:
    """Everything a campaign email carries: its documents and its template's files.

    Documents only when ``attach_documents`` is on (the default for a campaign
    created from a PDF or DOCX). A template's attachments travel with it in
    every mode — they are part of what the operator wrote. Deduplicated by
    filename, documents first.
    """
    from app.services.template_attachments import OutgoingAttachment, load_for_send

    out: list = []
    if campaign.attach_documents:
        ids = [uuid.UUID(str(i)) for i in campaign.document_ids or []]
        docs = (
            await db.execute(
                select(DiscoveryDocument)
                .options(undefer(DiscoveryDocument.content))
                .where(
                    DiscoveryDocument.id.in_(ids),
                    DiscoveryDocument.file_type.in_(list(ATTACHABLE_TYPES)),
                )
                .order_by(DiscoveryDocument.created_at, DiscoveryDocument.id)
            )
        ).scalars().all()
        for d in docs:
            if d.content:
                out.append(OutgoingAttachment(
                    filename=d.filename, content_type=ATTACHABLE_TYPES[d.file_type], data=d.content))
            else:
                logger.warning("Campaign %s: document %s has no stored file to attach", campaign.id, d.id)
    if campaign.template_id is not None and campaign.template_mode != "ai":
        out.extend(await load_for_send(db, campaign.template_id))
    seen: set[str] = set()
    return [a for a in out if not (a.filename in seen or seen.add(a.filename))]


async def attachment_names(db: AsyncSession, campaign: Campaign) -> list[str]:
    """Filenames :func:`_attachments` would send, without loading the bytes."""
    from app.models.email_template import EmailTemplateAttachment

    names: list[str] = []
    if campaign.attach_documents:
        ids = [uuid.UUID(str(i)) for i in campaign.document_ids or []]
        names += (
            await db.execute(
                select(DiscoveryDocument.filename).where(
                    DiscoveryDocument.id.in_(ids),
                    DiscoveryDocument.file_type.in_(list(ATTACHABLE_TYPES)),
                    DiscoveryDocument.content_purged_at.is_(None),
                ).order_by(DiscoveryDocument.created_at, DiscoveryDocument.id)
            )
        ).scalars().all()
    if campaign.template_id is not None and campaign.template_mode != "ai":
        names += (
            await db.execute(
                select(EmailTemplateAttachment.filename)
                .where(EmailTemplateAttachment.template_id == campaign.template_id)
                .order_by(EmailTemplateAttachment.created_at, EmailTemplateAttachment.id)
            )
        ).scalars().all()
    return list(dict.fromkeys(names))


@dataclass
class SendOutcome:
    status: str
    detail: Optional[str] = None


async def _release(db: AsyncSession, prospect_id: uuid.UUID, **values) -> None:
    await db.execute(
        update(CampaignProspect)
        .where(CampaignProspect.id == prospect_id, CampaignProspect.status == "sending")
        .values(**values)
    )
    await db.commit()


# A claim older than this with no outcome means the worker died mid-send: the
# task hard limit is 10 minutes, so nothing legitimate is still in flight.
STUCK_SEND_AFTER = timedelta(minutes=30)


async def reap_stuck_sends(db: AsyncSession, now: Optional[datetime] = None) -> dict:
    """Close out prospects stuck in ``sending`` and say so on their campaign.

    ``sending`` is committed before the wire so a crash can never make a
    prospect eligible again — which also means a crash leaves it there forever,
    invisible: not sent, not failed, not retried. After :data:`STUCK_SEND_AFTER`
    this marks it ``failed`` with the same "outcome unknown, not retried" stance
    the send path takes when the transport raises, keeps ``sent_at`` (the claim
    time), and records an error on the campaign's daily log so the operator
    sees it and can check the mailbox's Sent folder before resending.
    """
    now = _aware(now)
    rows = (
        await db.execute(
            select(CampaignProspect)
            .where(
                CampaignProspect.status == "sending",
                CampaignProspect.sent_at < now - STUCK_SEND_AFTER,
            )
            .order_by(CampaignProspect.sent_at, CampaignProspect.id)
        )
    ).scalars().all()
    by_campaign: dict = {}
    for p in rows:
        p.status = "failed"
        p.error = (
            "Send outcome unknown: the worker stopped mid-send and the claim was never "
            "settled. Check the mailbox's Sent folder before resending."
        )
        by_campaign.setdefault(p.campaign_id, []).append(p.to_email or str(p.id))
    for campaign_id, addresses in by_campaign.items():
        campaign = await db.get(Campaign, campaign_id)
        if campaign is not None:
            await log_error(
                db, campaign,
                f"{len(addresses)} send(s) stuck for over "
                f"{int(STUCK_SEND_AFTER.total_seconds() // 60)} min marked failed "
                f"(outcome unknown, not retried): {', '.join(addresses[:10])}",
                now,
            )
        logger.error("Campaign %s: %d stuck send(s) marked failed", campaign_id, len(addresses))
    await db.commit()
    return {"reaped": len(rows), "campaigns": len(by_campaign)}


async def send_prospect(
    db: AsyncSession, prospect_id: uuid.UUID, now: Optional[datetime] = None
) -> SendOutcome:
    """Send one approved prospect's email. Safe to call twice: one wins the claim."""
    from app.services.email_dispatch import send_outreach_email

    now = _aware(now)
    p = await db.get(CampaignProspect, prospect_id)
    if p is None or p.status != "approved":
        return SendOutcome("not_approved")
    campaign = await db.get(Campaign, p.campaign_id)
    if campaign is None or campaign.deleted_at is not None or campaign.status != "active":
        p.scheduled_for = None
        await db.commit()
        return SendOutcome("campaign_inactive")
    if await sent_today(db, campaign, now) >= send_cap(campaign):
        p.scheduled_for = None
        await db.commit()
        return SendOutcome("held", "warm-up or daily send limit reached")
    account = await sending_account(db, campaign)
    if account is None:
        p.scheduled_for = None
        await db.commit()
        return SendOutcome("held", "gmail account unavailable")
    email = (p.to_email or "").lower()
    if not email or not p.email_subject or not p.email_body:
        p.status, p.skip_reason = "skipped", "Nothing to send"
        await db.commit()
        return SendOutcome("skipped", p.skip_reason)
    if email in await previously_mailed(db, [email]):
        p.status, p.skip_reason = "skipped", "Already emailed — never sent twice"
        await db.commit()
        return SendOutcome("skipped", p.skip_reason)
    # queue_contacts already refuses these; this catches a row queued before
    # that check existed (the Partner Brief campaign had the org's own CEO
    # approved and due Monday 09:00), or before a mailbox on that domain was
    # connected. The last gate before the wire, so both the tick and the
    # instant send pass through it.
    if is_own_address(email, await own_addresses(db)):
        p.status, p.skip_reason = "skipped", "This organization's own address — not a prospect"
        await db.commit()
        return SendOutcome("skipped", p.skip_reason)
    # schedule_sends only schedules verified-deliverable addresses; this is the
    # same rule at the last gate, for a verdict that changed after scheduling
    # (a re-verification, a contact edit) — nothing goes on the wire that Hunter
    # hasn't called deliverable.
    contact = await db.get(Contact, p.contact_id) if p.contact_id else None
    if contact is None or contact.email_verification_result != VERIFIED:
        verdict = contact.email_verification_result if contact else None
        p.scheduled_for = None
        if verdict in (None, "unknown"):
            await db.commit()
            return SendOutcome("held", "email not verified yet")
        p.status, p.skip_reason = "skipped", f"Email not verified deliverable (verification: {verdict})"
        await db.commit()
        return SendOutcome("skipped", p.skip_reason)

    # Claim, committed before the wire: whoever flips approved→sending owns the
    # send, and a crash after the transport accepted it leaves the row claimed
    # rather than eligible. The partial unique index refuses the claim outright
    # if another campaign is already mailing this address.
    try:
        claimed = (
            await db.execute(
                update(CampaignProspect)
                .where(CampaignProspect.id == p.id, CampaignProspect.status == "approved")
                .values(status="sending", sent_at=now)
            )
        ).rowcount == 1
        await db.commit()
    except IntegrityError:
        await db.rollback()
        p = await db.get(CampaignProspect, prospect_id)
        p.status, p.skip_reason = "skipped", "Already being emailed by another campaign"
        await db.commit()
        return SendOutcome("skipped", p.skip_reason)
    if not claimed:
        return SendOutcome("not_approved")
    await db.refresh(p)

    try:
        result = await send_outreach_email(
            db,
            to_email=email,
            subject=p.email_subject,
            body_html=body_to_html(p.email_body),
            sender_account_id=account.id,
            attachments=await _attachments(db, campaign) or None,
        )
    except Exception as exc:  # noqa: BLE001
        # The transport may have accepted the message before this raised, so a
        # retry could double-send. Fail it; never put it back in the queue.
        logger.exception("Campaign send %s raised", prospect_id)
        await db.rollback()
        await _release(db, prospect_id, status="failed", error=f"Send outcome unknown, not retried: {exc}"[:500])
        return SendOutcome("failed", str(exc))

    status = result.get("status", "error")
    reason = result.get("reason") or result.get("detail")
    if _is_transient_hold(result):
        await _release(db, prospect_id, status="approved", sent_at=None, scheduled_for=None, error=reason)
        return SendOutcome("held", reason)
    if status == "suppressed":
        await _release(db, prospect_id, status="skipped", sent_at=None, skip_reason=reason or "On the do-not-contact list")
        return SendOutcome("skipped", reason)
    if status != "sent":
        await _release(db, prospect_id, status="failed", sent_at=None, error=reason or status)
        return SendOutcome("failed", reason or status)

    send = EmailSend(
        contact_id=p.contact_id,
        to_email=email,
        subject=p.email_subject[:500],
        body_html=body_to_html(p.email_body),
        status="sent",
        sent_at=now,
        provider_message_id=result.get("message_id"),
        gmail_message_id=result.get("gmail_message_id"),
        gmail_thread_id=result.get("gmail_thread_id"),
        from_email=result.get("from_email") or account.email,
        sender_account_id=account.id,
        user_id=campaign.user_id,
    )
    db.add(send)
    await db.flush()
    p.status = "sent"
    p.email_send_id = send.id
    p.gmail_message_id = result.get("gmail_message_id")
    p.gmail_thread_id = result.get("gmail_thread_id")
    p.scheduled_for = None
    p.error = None
    log = await daily_log(db, campaign, local_today(campaign, now))
    log.emails_sent = (log.emails_sent or 0) + 1
    await db.commit()
    return SendOutcome("sent")


# ---------------------------------------------------------------------------
# The tick
# ---------------------------------------------------------------------------
async def run_campaign(
    db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None, *, schedule: ScheduleFn
) -> dict:
    now = _aware(now)
    out: dict = {"campaign_id": str(campaign.id)}
    out.update(await reconcile_outcomes(db, campaign, now))
    if await enforce_bounce_rate(db, campaign, now):
        await db.commit()
        return {**out, "paused": "bounce_rate"}
    await advance_warmup(db, campaign, now)

    if await discovery_due(db, campaign, now):
        try:
            out["discovered"] = await discover_prospects(db, campaign, now)
        except (org_gemini.GeminiKeyMissing, org_gemini.GeminiError) as exc:
            # Mark the day done anyway: a failing search retried every 15
            # minutes is a quota burner, not a recovery strategy.
            campaign.last_discovery_on = local_today(campaign, now)
            await log_error(db, campaign, f"Discovery failed: {exc or 'no AI key saved'}", now)
    await db.commit()

    # Verify before drafting: an address Hunter rejects is retired here, so no
    # AI draft is written for a prospect who can never be sent.
    out["verification"] = await verify_prospects(db, campaign)
    await db.commit()
    out["drafted"] = await generate_drafts(db, campaign, now)
    await db.commit()
    out["scheduled"] = await schedule_sends(db, campaign, now, schedule=schedule)
    await db.commit()
    return out


async def run_campaign_now(
    db: AsyncSession, campaign_id: uuid.UUID, *, now: Optional[datetime] = None, schedule: ScheduleFn
) -> dict:
    """One campaign's tick, outside the beat — fired when a campaign is created
    so its first search doesn't wait up to 15 minutes. The next beat tick finds
    ``last_discovery_on`` already set and does not search again today."""
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None or campaign.status != "active" or campaign.deleted_at is not None:
        return {"campaign_id": str(campaign_id), "skipped": "not_active"}
    try:
        return await run_campaign(db, campaign, now, schedule=schedule)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Campaign %s first run failed", campaign_id)
        await db.rollback()
        campaign = await db.get(Campaign, campaign_id)
        if campaign is not None:
            await log_error(db, campaign, f"Scheduler error: {exc}", now)
            await db.commit()
        return {"campaign_id": str(campaign_id), "error": str(exc)}


async def send_approved_now(
    db: AsyncSession, campaign_id: uuid.UUID, *, now: Optional[datetime] = None, schedule: ScheduleFn
) -> dict:
    """Put this campaign's approved emails on the wire now, not at the next tick.

    Dispatched when an operator approves a draft or resumes a campaign, so an
    approval doesn't sit for up to 15 minutes. Deliberately only the tail of
    the tick: verification (an approved address Hunter hasn't seen would
    otherwise be held) and ``schedule_sends``. No discovery and no drafting —
    those spend AI and Hunter budget and belong to the beat.

    Same ``schedule_sends`` as the tick, at any hour: no send window, only the
    spam-protection gates — the warm-up ramp and daily limit (``send_cap``),
    verified-only, and the 60–120s gap between sends (``next_send_at``). So
    the first approval goes out now and a burst follows a minute or two apart.
    The
    ``scheduled_for`` lease keeps this and a concurrent tick from scheduling
    the same row twice, and the claim in ``send_prospect`` makes a duplicate
    task harmless anyway. Whatever this can't schedule — limit spent, past the
    horizon — the tick still picks up.
    """
    now = _aware(now)
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None or campaign.status != "active" or campaign.deleted_at is not None:
        return {"campaign_id": str(campaign_id), "skipped": "not_active"}
    try:
        verification = await verify_prospects(db, campaign)
        await db.commit()
        scheduled = await schedule_sends(db, campaign, now, schedule=schedule)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Campaign %s instant send failed", campaign_id)
        await db.rollback()
        return {"campaign_id": str(campaign_id), "error": str(exc)}
    return {"campaign_id": str(campaign_id), "verification": verification, "scheduled": scheduled}


async def process_new_prospects(
    db: AsyncSession, campaign_id: uuid.UUID, *, now: Optional[datetime] = None, schedule: ScheduleFn
) -> dict:
    """Verify, draft and schedule now — dispatched when contacts are added by hand.

    :func:`send_approved_now` plus drafting: an added contact arrives queued,
    so waiting for the tick would leave it unverified and undrafted for up to
    15 minutes. Still no discovery. Every gate is the tick's own.
    """
    now = _aware(now)
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None or campaign.status != "active" or campaign.deleted_at is not None:
        return {"campaign_id": str(campaign_id), "skipped": "not_active"}
    try:
        verification = await verify_prospects(db, campaign)
        await db.commit()
        drafted = await generate_drafts(db, campaign, now)
        await db.commit()
        scheduled = await schedule_sends(db, campaign, now, schedule=schedule)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Campaign %s processing added prospects failed", campaign_id)
        await db.rollback()
        return {"campaign_id": str(campaign_id), "error": str(exc)}
    return {"campaign_id": str(campaign_id), "verification": verification,
            "drafted": drafted, "scheduled": scheduled}


async def discover_now(
    db: AsyncSession, campaign_id: uuid.UUID, *, now: Optional[datetime] = None, schedule: ScheduleFn
) -> dict:
    """Search for more companies now — the operator's "Find more prospects" —
    then verify, draft and schedule what it found, like a tick would."""
    now = _aware(now)
    campaign = await db.get(Campaign, campaign_id)
    if campaign is None or campaign.status != "active" or campaign.deleted_at is not None:
        return {"campaign_id": str(campaign_id), "skipped": "not_active"}
    if await searches_today(db, campaign, now) >= MANUAL_DISCOVERY_RUNS:
        return {"campaign_id": str(campaign_id), "skipped": "daily_limit"}
    out: dict = {"campaign_id": str(campaign_id)}
    try:
        out["discovered"] = await discover_prospects(db, campaign, now)
    except (org_gemini.GeminiKeyMissing, org_gemini.GeminiError) as exc:
        await log_error(db, campaign, f"Discovery failed: {exc or 'no AI key saved'}", now)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Campaign %s discovery on demand failed", campaign_id)
        await db.rollback()
        return {**out, "error": str(exc)}
    await db.commit()
    out["verification"] = await verify_prospects(db, campaign)
    await db.commit()
    out["drafted"] = await generate_drafts(db, campaign, now)
    await db.commit()
    out["scheduled"] = await schedule_sends(db, campaign, now, schedule=schedule)
    await db.commit()
    return out


async def run_campaigns_tick(
    db: AsyncSession, *, now: Optional[datetime] = None, schedule: ScheduleFn
) -> list[dict]:
    """Every active campaign in the bound org, one at a time. One failure never stops the rest."""
    ids = (
        await db.execute(
            select(Campaign.id)
            .where(Campaign.status == "active", Campaign.deleted_at.is_(None))
            .order_by(Campaign.created_at, Campaign.id)
        )
    ).scalars().all()
    results = []
    for cid in ids:
        campaign = await db.get(Campaign, cid)
        if campaign is None:
            continue
        try:
            results.append(await run_campaign(db, campaign, now, schedule=schedule))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Campaign %s tick failed", cid)
            await db.rollback()
            campaign = await db.get(Campaign, cid)
            if campaign is not None:
                await log_error(db, campaign, f"Scheduler error: {exc}", now)
                await db.commit()
            results.append({"campaign_id": str(cid), "error": str(exc)})
    return results
