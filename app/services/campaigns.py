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
from app.models.email_template import WIRE_STATUSES, EmailSend
from app.models.gmail_account import GmailAccount
from app.models.outreach import OutreachEnrollment, OutreachEvent
from app.models.tenant import Organization
from app.models.user import User
from app.services import org_gemini, suppression

logger = logging.getLogger(__name__)

TICK = timedelta(minutes=15)
GAP_SECONDS = (120, 300)
WARMUP_STEP = 5
MAX_DAILY_LIMIT = 100
BOUNCE_PAUSE_RATE = 0.05
DRAFTS_PER_TICK = 5
DISCOVERY_COMPANIES = 10
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
            prospects_found=0, emails_sent=0, emails_bounced=0, errors=[],
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


async def queue_contacts(
    db: AsyncSession, campaign: Campaign, contacts: list[Contact], *, source: str
) -> int:
    """Queue contacts as prospects. Returns how many were newly queued.

    Silently passes over anyone without an email, on the do-not-contact list or
    flagged, already a prospect of *any* campaign, or already mailed by any path.
    """
    candidates: dict[str, Contact] = {}
    for c in contacts:
        email = (c.email or "").strip().lower()
        if not email or suppression.blocking_flag(c) is not None or email in candidates:
            continue
        candidates[email] = c
    if not candidates:
        return 0
    emails = set(candidates)
    excluded = await suppression.suppressed_addresses(db, emails)
    excluded |= await previously_mailed(db, emails)
    excluded |= set(
        (
            await db.execute(
                select(CampaignProspect.to_email).where(CampaignProspect.to_email.in_(emails))
            )
        ).scalars().all()
    )
    queued = 0
    for email, contact in candidates.items():
        if email in excluded:
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
    return queued


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
async def _seed_from_documents(db: AsyncSession, campaign: Campaign) -> int:
    """Queue the contacts (with emails) already found in the campaign's documents."""
    from app.services import discovery_documents

    ids = [uuid.UUID(str(i)) for i in campaign.document_ids or []]
    if not ids:
        return 0
    docs = (
        await db.execute(select(DiscoveryDocument).where(DiscoveryDocument.id.in_(ids)))
    ).scalars().all()
    contact_ids: set[str] = set()
    for doc in docs:
        cands = [
            c["id"] for c in ((doc.extracted or {}).get("candidates") or [])
            if any(p.get("email") for p in c.get("contacts", []))
        ]
        if cands:
            result = await discovery_documents.import_candidates(db, doc, cands)
            contact_ids |= set(result.get("contact_ids") or [])
    if not contact_ids:
        return 0
    contacts = (
        await db.execute(
            select(Contact).where(Contact.id.in_([uuid.UUID(c) for c in contact_ids]))
        )
    ).scalars().all()
    return await queue_contacts(db, campaign, list(contacts), source="document")


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


async def _fill_guessed_emails(db: AsyncSession, company_ids: list[uuid.UUID]) -> int:
    """Give named, address-less contacts at these companies a ``firstname@domain`` guess."""
    rows = (
        await db.execute(
            select(Contact, Company.domain)
            .join(Company, Company.id == Contact.company_id)
            .where(
                Contact.company_id.in_(company_ids),
                or_(Contact.email.is_(None), Contact.email == ""),
                Contact.first_name.isnot(None),
                Company.domain.isnot(None),
            )
        )
    ).all()
    filled = 0
    for contact, domain in rows:
        guess = guess_email(contact.first_name, domain)
        if guess:
            contact.email = guess
            filled += 1
    if filled:
        await db.flush()
        logger.info("Campaign discovery: guessed %d email(s) as firstname@domain", filled)
    return filled


async def _search_companies(db: AsyncSession, campaign: Campaign) -> int:
    from app.services import gemini_discovery
    from app.services.ingestion import get_ingestion_service

    icp = Icp.model_validate(campaign.icp_json or {})
    found = await gemini_discovery.search_companies(
        db,
        campaign.org_id,
        query=icp.search_query or icp.summary,
        industry=", ".join(icp.target_industries[:3]),
        location=", ".join(icp.target_locations[:3]),
        keywords=icp.keywords[:5],
        min_employees=icp.company_size.min,
        max_employees=icp.company_size.max,
        limit=DISCOVERY_COMPANIES,
    )
    result = await get_ingestion_service().ingest(db, found.records)
    company_ids = [o.company_id for o in result.outcomes if o.company_id]
    if not company_ids:
        return 0
    await _fill_guessed_emails(db, company_ids)
    rows = (
        await db.execute(
            select(Contact)
            .where(Contact.company_id.in_(company_ids), Contact.email.isnot(None))
            .order_by(Contact.created_at, Contact.id)
        )
    ).scalars().all()
    # One person per company: a cold campaign that mails three people at the
    # same firm on the same day reads as a blast to all three.
    first: dict[uuid.UUID, Contact] = {}
    for c in rows:
        first.setdefault(c.company_id, c)
    return await queue_contacts(db, campaign, list(first.values()), source=found.provider)


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
    return total


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

_PLACEHOLDER = re.compile(r"\[[^\]]{2,40}\]|\{\{|\}\}|<[A-Z][A-Z _]{2,}>")


def draft_prompt(campaign: Campaign, contact: Optional[Contact], company: Optional[Company],
                 sender_name: str, sender_org: str) -> str:
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
    lines.append("")
    lines.append(
        "Write a first-touch cold email: 60-120 words, a subject under 8 words, open with "
        "something relevant to their company, one clear value point, and end with a "
        "low-pressure question. Greet them by first name if known. Sign off with the "
        f"sender's first name only.\n\nReturn: {{\"subject\": str, \"body\": str}}"
    )
    return "\n".join(lines)


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
    """Draft emails for queued prospects. Stops at the first LLM failure (logged)."""
    queued = (
        await db.execute(
            select(CampaignProspect)
            .where(CampaignProspect.campaign_id == campaign.id, CampaignProspect.status == "queued")
            .order_by(CampaignProspect.created_at, CampaignProspect.id)
            .limit(limit)
        )
    ).scalars().all()
    if not queued:
        return 0
    sender_name, sender_org = await _sender_identity(db, campaign)
    drafted = 0
    for p in queued:
        contact = await db.get(Contact, p.contact_id) if p.contact_id else None
        company = await db.get(Company, p.company_id) if p.company_id else None
        if contact is None or not contact.email:
            p.status, p.skip_reason = "skipped", "Contact no longer has an email address"
            continue
        try:
            draft = await org_gemini.generate_json(
                db, campaign.org_id, system=DRAFT_SYSTEM,
                prompt=draft_prompt(campaign, contact, company, sender_name, sender_org),
                schema=DraftEmail, max_tokens=2048, temperature=0.6,
            )
        except (org_gemini.GeminiKeyMissing, org_gemini.GeminiError) as exc:
            msg = str(exc) or "No AI key is saved in Settings → API Keys"
            await log_error(db, campaign, f"Drafting stopped: {msg}", now)
            break
        subject = " ".join(draft.subject.split())[:150]
        body = draft.body.strip()[:4000]
        if not subject or not body:
            await log_error(db, campaign, f"Empty draft for {p.to_email}; will retry", now)
            continue
        p.email_subject, p.email_body = subject, body
        # A leftover "[Company Name]" must never go out on autopilot.
        needs_review = campaign.review_before_send or bool(_PLACEHOLDER.search(subject + body))
        if needs_review:
            p.status = "draft_created"
        else:
            p.status, p.approved_at = "approved", _aware(now)
        drafted += 1
    await db.flush()
    return drafted


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
async def sending_account(db: AsyncSession, campaign: Campaign) -> Optional[GmailAccount]:
    if campaign.gmail_account_id is None:
        return None
    acct = await db.get(GmailAccount, campaign.gmail_account_id)
    if acct is None or acct.status != "connected" or not acct.sending_enabled:
        return None
    return acct


async def schedule_sends(
    db: AsyncSession, campaign: Campaign, now: Optional[datetime] = None, *, schedule: ScheduleFn
) -> int:
    """Hand this tick's sends to ``schedule(org_id, prospect_id, countdown_seconds)``."""
    now = _aware(now)
    is_open = window_open(campaign, now)
    if not is_open:
        logger.info("schedule_sends campaign=%s: window closed at %s", campaign.id, now)
        return 0
    acct = await sending_account(db, campaign)
    if acct is None:
        logger.info("schedule_sends campaign=%s: no sending account (gmail_account_id=%s)",
                    campaign.id, campaign.gmail_account_id)
        campaign.status, campaign.pause_reason = "paused", "gmail_disconnected"
        await log_error(
            db, campaign,
            "Paused automatically: the Gmail account for this campaign is disconnected "
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
    cap, already_sent = ramp_cap(campaign), await sent_today(db, campaign, now)
    quota = cap - already_sent - in_flight
    if quota <= 0:
        logger.info("schedule_sends campaign=%s: no quota (ramp_cap=%d sent_today=%d in_flight=%d)",
                    campaign.id, cap, already_sent, in_flight)
        return 0
    ready = (
        await db.execute(
            select(CampaignProspect)
            .where(
                CampaignProspect.campaign_id == campaign.id,
                CampaignProspect.status == "approved",
                or_(CampaignProspect.scheduled_for.is_(None), CampaignProspect.scheduled_for < lease_floor),
            )
            .order_by(CampaignProspect.approved_at, CampaignProspect.id)
            .limit(quota)
        )
    ).scalars().all()
    _, window_end = window_bounds(campaign, now)
    horizon = min(now + TICK, window_end)
    at = max(now, _aware(campaign.next_send_at) if campaign.next_send_at else now)
    logger.info(
        "schedule_sends campaign=%s: window=%s, account=%s, quota=%d, ready=%d, "
        "now=%s, next_send_at=%s, at=%s, horizon=%s, window_end=%s",
        campaign.id, is_open, acct is not None, quota, len(ready),
        now, campaign.next_send_at, at, horizon, window_end,
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
    if not campaign.attach_documents:
        return []
    from app.services.template_attachments import OutgoingAttachment

    ids = [uuid.UUID(str(i)) for i in campaign.document_ids or []]
    docs = (
        await db.execute(
            select(DiscoveryDocument)
            .options(undefer(DiscoveryDocument.content))
            .where(
                DiscoveryDocument.id.in_(ids),
                DiscoveryDocument.file_type.in_(list(ATTACHABLE_TYPES)),
            )
        )
    ).scalars().all()
    return [
        OutgoingAttachment(filename=d.filename, content_type=ATTACHABLE_TYPES[d.file_type], data=d.content)
        for d in docs if d.content
    ]


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
    if not window_open(campaign, now) or await sent_today(db, campaign, now) >= ramp_cap(campaign):
        p.scheduled_for = None
        await db.commit()
        return SendOutcome("held", "outside window or campaign cap reached")
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

    if campaign.last_discovery_on != local_today(campaign, now) and (
        await _pending_count(db, campaign) < 2 * campaign.daily_send_limit
    ):
        try:
            out["discovered"] = await discover_prospects(db, campaign, now)
        except (org_gemini.GeminiKeyMissing, org_gemini.GeminiError) as exc:
            # Mark the day done anyway: a failing search retried every 15
            # minutes is a quota burner, not a recovery strategy.
            campaign.last_discovery_on = local_today(campaign, now)
            await log_error(db, campaign, f"Discovery failed: {exc or 'no AI key saved'}", now)
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
