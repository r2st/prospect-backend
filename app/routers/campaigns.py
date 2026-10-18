"""Autopilot Outreach campaigns: documents → ICP → prospects → paced Gmail sends.

Mounted at ``/campaigns`` (authenticated, org-scoped). The pipeline itself runs
in ``app.services.campaigns``, driven by the 15-minute beat; this router only
creates campaigns, changes their settings and lets an operator review drafts.

Unsubscribe is deliberately *not* here: every campaign send goes through
``send_outreach_email``, which adds the footer link and the RFC 8058 headers for
the platform's ``GET/POST /api/v1/unsubscribe``. That records the opt-out on the
org's durable suppression list, which every send gate reads — a campaign-local
list would be one the gate can't see.

Agents see their own campaigns; managers and admins see every campaign in their
org. Org isolation itself comes from the read filter (``OrgScopedMixin``).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models.campaign import Campaign, CampaignDailyLog, CampaignProspect
from app.models.company import Company
from app.models.contact import Contact
from app.models.discovery_document import STATUS_READY, DiscoveryDocument
from app.models.gmail_account import GmailAccount
from app.models.user import User, UserRole, normalize_role
from app.services import campaigns as svc
from app.services import org_gemini

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/campaigns", tags=["campaigns"])

AI_KEY_REQUIRED = {
    "code": "gemini_key_required",
    "message": (
        "Add a Gemini API key in Settings → API Keys to run Autopilot Outreach "
        "(a Groq or OpenRouter key also works)."
    ),
}
_SEES_ALL = {UserRole.ORG_MANAGER.value, UserRole.ORG_ADMIN.value, UserRole.PLATFORM_ADMIN.value}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class _Settings(BaseModel):
    daily_send_limit: Optional[int] = Field(None, ge=1, le=svc.MAX_DAILY_LIMIT)
    send_window_start: Optional[str] = None
    send_window_end: Optional[str] = None
    timezone: Optional[str] = None
    send_on_weekends: Optional[bool] = None
    review_before_send: Optional[bool] = None
    attach_documents: Optional[bool] = None
    # Legacy name; ``sender_account_id`` takes a Gmail *or* SMTP account id.
    gmail_account_id: Optional[uuid.UUID] = None
    sender_account_id: Optional[uuid.UUID] = None
    # Draft from an email template; template_id null clears it (see TEMPLATE_MODES).
    template_id: Optional[uuid.UUID] = None
    template_mode: Optional[Literal["ai", "template", "ai_template"]] = None

    @field_validator("send_window_start", "send_window_end")
    @classmethod
    def _hhmm(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return svc.parse_hhmm(v).strftime("%H:%M")

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"Unknown timezone: {v}")
        return v


class CampaignCreate(_Settings):
    name: str = Field(..., min_length=1, max_length=255)
    document_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=20)


class CampaignUpdate(_Settings):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    status: Optional[Literal["active", "paused", "stopped"]] = None
    icp_json: Optional[dict[str, Any]] = None


class CampaignOut(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    name: str
    status: str
    pause_reason: Optional[str] = None
    document_ids: list[str]
    document_summary: Optional[str] = None
    icp_json: Optional[dict[str, Any]] = None
    gmail_account_id: Optional[uuid.UUID] = None
    smtp_account_id: Optional[uuid.UUID] = None
    template_id: Optional[uuid.UUID] = None
    template_mode: str = "ai"
    daily_send_limit: int
    send_window_start: str
    send_window_end: str
    timezone: str
    send_on_weekends: bool
    review_before_send: bool
    attach_documents: bool
    warmup_day: int
    todays_cap: int
    last_discovery_on: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CampaignDetail(CampaignOut):
    metrics: dict[str, Any]
    daily: list[dict[str, Any]]
    # Filenames every email of this campaign carries (documents + template files).
    attachments: list[str] = []


class ProspectOut(BaseModel):
    id: uuid.UUID
    campaign_id: uuid.UUID
    company_id: Optional[uuid.UUID] = None
    company_name: Optional[str] = None
    contact_id: Optional[uuid.UUID] = None
    contact_name: Optional[str] = None
    to_email: Optional[str] = None
    # Hunter's verdict on the contact's address ("deliverable" is the only one
    # that is ever sent); None = not verified yet.
    email_verification: Optional[str] = None
    email_verified: bool = False
    # "hunter_finder" | "pattern" for addresses discovery supplied; None otherwise.
    email_origin: Optional[str] = None
    status: str
    source: Optional[str] = None
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    email_subject: Optional[str] = None
    email_body: Optional[str] = None
    gmail_message_id: Optional[str] = None
    sent_at: Optional[datetime] = None
    replied_at: Optional[datetime] = None
    bounced_at: Optional[datetime] = None
    created_at: datetime
    # Set on approve only: "dispatched" (sending now, paced), "window_closed"
    # (goes out when the send window opens) or "unavailable" (the broker
    # couldn't be reached; the next tick sends it).
    instant_send: Optional[str] = None


class ApproveIn(BaseModel):
    """Optional edits made while reviewing the draft."""

    email_subject: Optional[str] = Field(None, min_length=1, max_length=150)
    email_body: Optional[str] = Field(None, min_length=1, max_length=4000)


class SkipIn(BaseModel):
    reason: Optional[str] = Field(None, max_length=500)


class AddContactsIn(BaseModel):
    """Contacts picked from the contact database."""
    contact_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=500)


class ManualProspectIn(BaseModel):
    """One person typed in by the operator."""
    company_name: str = Field(..., min_length=1, max_length=255)
    domain: Optional[str] = Field(None, max_length=255)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    title: Optional[str] = Field(None, max_length=255)
    email: Optional[str] = Field(None, max_length=320)

    @field_validator("email")
    @classmethod
    def _email(cls, v: Optional[str]) -> Optional[str]:
        from app.services.email_validation import is_valid_syntax

        v = (v or "").strip().lower() or None
        if v is not None and not is_valid_syntax(v):
            raise ValueError("That doesn't look like an email address.")
        return v


class AddProspectsOut(BaseModel):
    queued: int
    skipped: list[dict[str, Any]]
    # "dispatched" (verification and drafting start now) or "unavailable"
    # (the broker is down; the next 15-minute tick picks them up).
    processing: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dispatch_processing(c: Campaign) -> str:
    """Verify, draft and schedule newly added contacts now, not at the next tick.

    Dispatched whatever the send window: verification and drafting are worth
    doing at 3 a.m. even though the send waits for the window to open.
    """
    try:
        from app.tasks.campaigns import process_new_prospects

        process_new_prospects.delay(str(c.org_id), str(c.id))
    except Exception as exc:  # pragma: no cover - broker-dependent
        logger.warning("Could not dispatch processing for campaign %s: %s", c.id, exc)
        return "unavailable"
    return "dispatched"


def _require_open(c: Campaign) -> None:
    if c.status == "stopped" or c.deleted_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This campaign is stopped; create a new one.")


def _dispatch_instant_send(c: Campaign) -> str:
    """Schedule the campaign's approved emails now instead of at the next tick.

    Best-effort, after the commit that made the prospect approved: if the
    broker is unreachable the 15-minute beat sends it as before. Not dispatched
    while the window is shut — the task would only find nothing to do.
    """
    if not svc.window_open(c):
        return "window_closed"
    try:
        from app.tasks.campaigns import send_approved_now

        send_approved_now.delay(str(c.org_id), str(c.id))
    except Exception as exc:  # pragma: no cover - broker-dependent
        logger.warning("Could not dispatch instant send for campaign %s: %s", c.id, exc)
        return "unavailable"
    return "dispatched"


def _out(c: Campaign) -> dict:
    return {
        "id": c.id,
        "user_id": c.user_id,
        "name": c.name,
        "status": c.status,
        "pause_reason": c.pause_reason,
        "document_ids": [str(i) for i in c.document_ids or []],
        "document_summary": c.document_summary,
        "icp_json": c.icp_json,
        "gmail_account_id": c.gmail_account_id,
        "smtp_account_id": c.smtp_account_id,
        "template_id": c.template_id,
        "template_mode": c.template_mode or "ai",
        "daily_send_limit": c.daily_send_limit,
        "send_window_start": c.send_window_start,
        "send_window_end": c.send_window_end,
        "timezone": c.timezone,
        "send_on_weekends": c.send_on_weekends,
        "review_before_send": c.review_before_send,
        "attach_documents": c.attach_documents,
        "warmup_day": c.warmup_day,
        "todays_cap": svc.ramp_cap(c),
        "last_discovery_on": c.last_discovery_on.isoformat() if c.last_discovery_on else None,
        "created_at": c.created_at,
        "updated_at": c.updated_at,
    }


def _sees_all(user: User) -> bool:
    return normalize_role(user.role) in _SEES_ALL


async def _get_campaign(db: AsyncSession, campaign_id: uuid.UUID, user: User) -> Campaign:
    c = await db.get(Campaign, campaign_id)
    # db.get bypasses nothing here: Campaign is org-scoped, but an identity-map
    # hit skips the SELECT, so the org is checked explicitly as well.
    if (
        c is None
        or c.deleted_at is not None
        or (user.org_id is not None and c.org_id != user.org_id and not _is_platform_admin(user))
        or (not _sees_all(user) and c.user_id != user.id)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Campaign not found")
    return c


def _is_platform_admin(user: User) -> bool:
    return normalize_role(user.role) == UserRole.PLATFORM_ADMIN.value


async def _get_prospect(db: AsyncSession, campaign: Campaign, prospect_id: uuid.UUID) -> CampaignProspect:
    p = (
        await db.execute(
            select(CampaignProspect)
            .where(CampaignProspect.id == prospect_id, CampaignProspect.campaign_id == campaign.id)
            .limit(1)
        )
    ).scalars().first()
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Prospect not found")
    return p


class SenderOut(BaseModel):
    id: uuid.UUID
    kind: Literal["gmail", "smtp"]
    email: str
    display_name: Optional[str] = None
    mine: bool = False


async def _resolve_sender(db: AsyncSession, user: User, account_id: Optional[uuid.UUID]):
    """``(kind, account)`` for a connected, sending-enabled Gmail or SMTP mailbox.

    With an id, exactly that account (org-scoped by the read filter, so another
    tenant's id is simply not found). Without one, the caller's own Gmail, then
    their own SMTP account.
    """
    from app.models.smtp_account import SmtpAccount

    for kind, model, order in (
        ("gmail", GmailAccount, (GmailAccount.connected_at, GmailAccount.id)),
        ("smtp", SmtpAccount, (SmtpAccount.email, SmtpAccount.id)),
    ):
        stmt = select(model).where(model.status == "connected", model.sending_enabled.is_(True))
        stmt = stmt.where(model.id == account_id) if account_id is not None else stmt.where(model.user_id == user.id)
        acct = (await db.execute(stmt.order_by(*order).limit(1))).scalars().first()
        if acct is not None:
            return kind, acct
    if account_id is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"code": "sender_unavailable",
             "message": "That sending account isn't connected or has sending turned off."},
        )
    raise HTTPException(
        status.HTTP_409_CONFLICT,
        {"code": "gmail_required",
         "message": "Connect a Gmail or SMTP account with sending turned on before starting a campaign."},
    )


async def _apply_template(db: AsyncSession, c: Campaign, data: dict) -> None:
    """Apply ``template_id`` / ``template_mode`` from a create or update payload."""
    from app.models.email_template import EmailTemplate

    if "template_id" in data:
        tid = data["template_id"]
        if tid is not None:
            t = await db.get(EmailTemplate, tid)  # org-scoped: another tenant's id isn't found
            if t is None or t.is_archived:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "That email template wasn't found.")
        c.template_id = tid
    if data.get("template_mode"):
        c.template_mode = data["template_mode"]
    if c.template_mode != "ai" and c.template_id is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Choose an email template, or let the AI write the drafts.")


def _set_sender(c: Campaign, kind: str, acct) -> None:
    c.gmail_account_id = acct.id if kind == "gmail" else None
    c.smtp_account_id = acct.id if kind == "smtp" else None


async def _resolve_account(
    db: AsyncSession, user: User, account_id: Optional[uuid.UUID]
) -> GmailAccount:
    """Kept for callers that need a Gmail account specifically."""
    kind, acct = await _resolve_sender(db, user, account_id)
    if kind != "gmail":
        raise HTTPException(status.HTTP_409_CONFLICT, {"code": "gmail_required",
                            "message": "This needs a Gmail account."})
    return acct


async def _metrics(db: AsyncSession, c: Campaign) -> dict:
    rows = (
        await db.execute(
            select(CampaignProspect.status, func.count(CampaignProspect.id))
            .where(CampaignProspect.campaign_id == c.id)
            .group_by(CampaignProspect.status)
        )
    ).all()
    by_status = {s: n for s, n in rows}
    mailed = sum(by_status.get(s, 0) for s in svc.WIRE)
    companies_discovered = (
        await db.execute(
            select(func.count(func.distinct(CampaignProspect.company_id))).where(
                CampaignProspect.campaign_id == c.id, CampaignProspect.company_id.isnot(None)
            )
        )
    ).scalar_one()
    # Of the prospects still in the pipeline: verified (sendable) vs not yet
    # checked. Checked-and-not-deliverable ones are retired by verify_prospects.
    verified_pending = unverified_pending = 0
    for result, checked, n in (
        await db.execute(
            select(
                Contact.email_verification_result,
                Contact.email_verified_at.isnot(None),
                func.count(CampaignProspect.id),
            )
            .join(Contact, Contact.id == CampaignProspect.contact_id)
            .where(CampaignProspect.campaign_id == c.id, CampaignProspect.status.in_(svc.PENDING))
            .group_by(Contact.email_verification_result, Contact.email_verified_at.isnot(None))
        )
    ).all():
        if result == svc.VERIFIED:
            verified_pending += n
        elif not checked:
            unverified_pending += n
    replied = by_status.get("replied", 0)
    bounced = by_status.get("bounced", 0)
    return {
        "by_status": by_status,
        "total_prospects": sum(n for s, n in by_status.items() if s != "no_email"),
        "emails_sent": mailed,
        "replies": replied,
        "bounces": bounced,
        "reply_rate": round(replied / mailed, 4) if mailed else 0.0,
        "bounce_rate": round(bounced / mailed, 4) if mailed else 0.0,
        "sent_today": await svc.sent_today(db, c),
        "todays_cap": svc.ramp_cap(c),
        "awaiting_review": by_status.get("draft_created", 0),
        "companies_discovered": companies_discovered,
        "companies_without_email": by_status.get("no_email", 0),
        # Drafted and approved, waiting for a send slot.
        "queued_to_send": by_status.get("approved", 0),
        "pending_verified": verified_pending,
        "pending_unverified": unverified_pending,
    }


def _apply_settings(c: Campaign, data: dict) -> None:
    for field in (
        "daily_send_limit", "send_window_start", "send_window_end", "timezone",
        "send_on_weekends", "review_before_send", "attach_documents",
    ):
        if field in data and data[field] is not None:
            setattr(c, field, data[field])
    if svc.parse_hhmm(c.send_window_start) >= svc.parse_hhmm(c.send_window_end):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "The send window must end after it starts.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("", response_model=CampaignOut, status_code=status.HTTP_201_CREATED)
async def create_campaign(
    payload: CampaignCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Create a campaign from discovery documents; the AI writes the ICP now."""
    ids = list(dict.fromkeys(payload.document_ids))
    docs = (
        await db.execute(select(DiscoveryDocument).where(DiscoveryDocument.id.in_(ids)))
    ).scalars().all()
    found = {d.id for d in docs}
    missing = [str(i) for i in ids if i not in found]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Documents not found: {', '.join(missing)}")
    not_ready = [d.filename for d in docs if d.status != STATUS_READY]
    if not_ready:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            {"code": "documents_not_ready", "message": f"Still processing: {', '.join(not_ready)}"},
        )
    if not await org_gemini.has_org_llm_key(db, user.org_id):
        raise HTTPException(status.HTTP_409_CONFLICT, AI_KEY_REQUIRED)
    sender_kind, account = await _resolve_sender(
        db, user, payload.sender_account_id or payload.gmail_account_id
    )

    try:
        icp, summary = await svc.build_icp(db, user.org_id, list(docs))
    except svc.CampaignError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, {"code": exc.code, "message": exc.message})
    except org_gemini.GeminiKeyMissing:
        raise HTTPException(status.HTTP_409_CONFLICT, AI_KEY_REQUIRED)
    except org_gemini.GeminiError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            {"code": "ai_failed", "message": f"Couldn't build a customer profile: {exc}"},
        )

    c = Campaign(
        user_id=user.id,
        name=payload.name.strip(),
        status="active",
        document_ids=[str(i) for i in ids],
        document_summary=summary,
        icp_json=icp.model_dump(),
        warmup_day=1,
        # Column defaults only apply at flush; _apply_settings reads them first.
        daily_send_limit=20,
        send_window_start="09:00",
        send_window_end="18:00",
        timezone="America/Toronto",
        send_on_weekends=False,
        # On unless the caller says otherwise: the campaign is live, and runs
        # its first search and drafting, the moment it is created. The wizard
        # used to send its settings under a key this endpoint ignores, so every
        # campaign got this default — and "off" meant auto-approving drafts the
        # operator had asked to review.
        review_before_send=True,
        attach_documents=False,
    )
    # Attach the campaign's documents by default when there is something to
    # attach: the brief the operator uploaded is usually the thing they want
    # the prospect to read.
    c.attach_documents = any(d.file_type in svc.ATTACHABLE_TYPES for d in docs)
    c.template_mode = "ai"
    _set_sender(c, sender_kind, account)
    _apply_settings(c, payload.model_dump(
        exclude={"name", "document_ids", "gmail_account_id", "sender_account_id"}))
    await _apply_template(db, c, payload.model_dump(
        include={"template_id", "template_mode"}, exclude_unset=True))
    db.add(c)
    await db.commit()
    await db.refresh(c)
    # Best-effort: search now rather than at the next 15-minute tick. If the
    # broker is unreachable the beat picks the campaign up as before.
    try:
        from app.tasks.campaigns import run_campaign_now

        run_campaign_now.delay(str(c.org_id), str(c.id))
    except Exception as exc:  # pragma: no cover - broker-dependent
        logger.warning("Could not dispatch first run for campaign %s: %s", c.id, exc)
    return _out(c)


@router.get("/senders", response_model=list[SenderOut])
async def list_senders(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Connected, sending-enabled mailboxes a campaign can send from — the
    caller's own first, then the rest of the org's."""
    from app.models.smtp_account import SmtpAccount

    out: list[SenderOut] = []
    for kind, model in (("gmail", GmailAccount), ("smtp", SmtpAccount)):
        rows = (await db.execute(
            select(model).where(model.status == "connected", model.sending_enabled.is_(True))
            .order_by(model.email, model.id)
        )).scalars().all()
        out += [SenderOut(id=a.id, kind=kind, email=a.email,
                          display_name=getattr(a, "display_name", None),
                          mine=a.user_id == user.id) for a in rows]
    return sorted(out, key=lambda a: (not a.mine, a.kind, a.email))


@router.get("", response_model=list[CampaignOut])
async def list_campaigns(
    status_filter: Optional[str] = Query(None, alias="status"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    stmt = select(Campaign).where(Campaign.deleted_at.is_(None))
    if not _sees_all(user):
        stmt = stmt.where(Campaign.user_id == user.id)
    if status_filter:
        stmt = stmt.where(Campaign.status == status_filter)
    rows = (
        await db.execute(stmt.order_by(Campaign.created_at.desc(), Campaign.id))
    ).scalars().all()
    return [_out(c) for c in rows]


@router.get("/{campaign_id}", response_model=CampaignDetail)
async def get_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    c = await _get_campaign(db, campaign_id, user)
    logs = (
        await db.execute(
            select(CampaignDailyLog)
            .where(CampaignDailyLog.campaign_id == c.id)
            .order_by(CampaignDailyLog.date.desc())
            .limit(30)
        )
    ).scalars().all()
    daily = [
        {
            "date": log.date.isoformat(),
            "prospects_found": log.prospects_found,
            "emails_sent": log.emails_sent,
            "emails_bounced": log.emails_bounced,
            "errors": log.errors or [],
        }
        for log in logs
    ]
    return {**_out(c), "metrics": await _metrics(db, c), "daily": daily,
            "attachments": await svc.attachment_names(db, c)}


@router.patch("/{campaign_id}", response_model=CampaignOut)
async def update_campaign(
    campaign_id: uuid.UUID,
    payload: CampaignUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Change settings, or pause / resume / stop. A stopped campaign stays stopped."""
    c = await _get_campaign(db, campaign_id, user)
    data = payload.model_dump(exclude_unset=True)
    if c.status == "stopped" and data.get("status") in ("active", "paused"):
        raise HTTPException(status.HTTP_409_CONFLICT, "A stopped campaign can't be restarted; create a new one.")
    if data.get("name"):
        c.name = data["name"].strip()
    if "icp_json" in data and data["icp_json"] is not None:
        c.icp_json = svc.Icp.model_validate(data["icp_json"]).model_dump()
    await _apply_template(db, c, {k: data[k] for k in ("template_id", "template_mode") if k in data})
    if data.get("sender_account_id") or data.get("gmail_account_id"):
        _set_sender(c, *await _resolve_sender(
            db, user, data.get("sender_account_id") or data["gmail_account_id"]))
    _apply_settings(c, data)

    new_status = data.get("status")
    if new_status and new_status != c.status:
        if new_status == "active":
            # Resuming after an automatic pause: the cause must be fixed first.
            if await svc.sending_account(db, c) is None:
                _set_sender(c, *await _resolve_sender(db, user, None))
            c.pause_reason = None
            c.next_send_at = None
        elif new_status in ("paused", "stopped"):
            c.pause_reason = None
            # Release scheduled-but-unsent leases; send_prospect would refuse
            # them anyway, this just keeps the counts honest.
            pending = (
                await db.execute(
                    select(CampaignProspect).where(
                        CampaignProspect.campaign_id == c.id,
                        CampaignProspect.status == "approved",
                        CampaignProspect.scheduled_for.isnot(None),
                    )
                )
            ).scalars().all()
            for p in pending:
                p.scheduled_for = None
        if new_status == "stopped":
            unsent = (
                await db.execute(
                    select(CampaignProspect).where(
                        CampaignProspect.campaign_id == c.id,
                        CampaignProspect.status.in_(svc.PENDING),
                    )
                )
            ).scalars().all()
            for p in unsent:
                p.status, p.skip_reason = "skipped", "Campaign stopped"
        c.status = new_status
    await db.commit()
    await db.refresh(c)
    if new_status == "active":
        # Resuming: whatever was approved while paused goes out now.
        _dispatch_instant_send(c)
    return _out(c)


@router.delete("/{campaign_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_campaign(
    campaign_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Soft delete: the prospect rows stay, because they are what makes "never
    the same address twice" true for the campaigns that come after this one."""
    c = await _get_campaign(db, campaign_id, user)
    c.status = "stopped"
    c.deleted_at = datetime.now(timezone.utc)
    unsent = (
        await db.execute(
            select(CampaignProspect).where(
                CampaignProspect.campaign_id == c.id, CampaignProspect.status.in_(svc.PENDING)
            )
        )
    ).scalars().all()
    for p in unsent:
        p.status, p.skip_reason = "skipped", "Campaign deleted"
    await db.commit()


@router.get("/{campaign_id}/prospects")
async def list_prospects(
    campaign_id: uuid.UUID,
    status_filter: Optional[str] = Query(None, alias="status"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    c = await _get_campaign(db, campaign_id, user)
    base = select(CampaignProspect).where(CampaignProspect.campaign_id == c.id)
    if status_filter:
        base = base.where(CampaignProspect.status == status_filter)
    total = (
        await db.execute(select(func.count()).select_from(base.subquery()))
    ).scalar_one()
    rows = (
        await db.execute(
            base.order_by(CampaignProspect.created_at.desc(), CampaignProspect.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).scalars().all()

    company_ids = {p.company_id for p in rows if p.company_id}
    contact_ids = {p.contact_id for p in rows if p.contact_id}
    companies = {
        cid: name for cid, name in (
            await db.execute(select(Company.id, Company.name).where(Company.id.in_(company_ids)))
        ).all()
    } if company_ids else {}
    contact_rows = (
        await db.execute(
            select(
                Contact.id, Contact.first_name, Contact.last_name,
                Contact.email_verification_result, Contact.email_verification_meta,
            ).where(Contact.id.in_(contact_ids))
        )
    ).all() if contact_ids else []
    contacts = {r.id: " ".join(x for x in (r.first_name, r.last_name) if x) or None for r in contact_rows}
    verdicts = {
        r.id: (r.email_verification_result, (r.email_verification_meta or {}).get("email_origin"))
        for r in contact_rows
    }

    items = [
        ProspectOut(
            id=p.id, campaign_id=p.campaign_id,
            company_id=p.company_id, company_name=companies.get(p.company_id),
            contact_id=p.contact_id, contact_name=contacts.get(p.contact_id),
            to_email=p.to_email,
            email_verification=verdicts.get(p.contact_id, (None, None))[0],
            email_verified=verdicts.get(p.contact_id, (None, None))[0] == svc.VERIFIED,
            email_origin=verdicts.get(p.contact_id, (None, None))[1],
            status=p.status, source=p.source,
            skip_reason=p.skip_reason, error=p.error,
            email_subject=p.email_subject, email_body=p.email_body,
            gmail_message_id=p.gmail_message_id,
            sent_at=p.sent_at, replied_at=p.replied_at, bounced_at=p.bounced_at,
            created_at=p.created_at,
        )
        for p in rows
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.post("/{campaign_id}/prospects/{prospect_id}/approve", response_model=ProspectOut)
async def approve_prospect(
    campaign_id: uuid.UUID,
    prospect_id: uuid.UUID,
    payload: Optional[ApproveIn] = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Approve a reviewed draft (optionally edited) for sending."""
    c = await _get_campaign(db, campaign_id, user)
    p = await _get_prospect(db, c, prospect_id)
    if p.status not in ("draft_created", "approved"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Can't approve a prospect that is '{p.status}'.")
    if payload is not None:
        if payload.email_subject:
            p.email_subject = " ".join(payload.email_subject.split())
        if payload.email_body:
            p.email_body = payload.email_body.strip()
    if not p.email_subject or not p.email_body:
        raise HTTPException(status.HTTP_409_CONFLICT, "There is no draft to approve yet.")
    p.status = "approved"
    p.approved_at = p.approved_at or datetime.now(timezone.utc)
    p.error = None
    await db.commit()
    await db.refresh(p)
    out = ProspectOut.model_validate(p, from_attributes=True)
    if c.status == "active":
        out.instant_send = _dispatch_instant_send(c)
    return out


@router.post("/{campaign_id}/prospects/{prospect_id}/skip", response_model=ProspectOut)
async def skip_prospect(
    campaign_id: uuid.UUID,
    prospect_id: uuid.UUID,
    payload: Optional[SkipIn] = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    c = await _get_campaign(db, campaign_id, user)
    p = await _get_prospect(db, c, prospect_id)
    if p.status not in svc.PENDING:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Can't skip a prospect that is '{p.status}'.")
    p.status = "skipped"
    p.skip_reason = (payload.reason if payload and payload.reason else None) or "Skipped by operator"
    p.scheduled_for = None
    await db.commit()
    await db.refresh(p)
    return ProspectOut.model_validate(p, from_attributes=True)


@router.post("/{campaign_id}/prospects/add", response_model=AddProspectsOut)
async def add_contacts_to_campaign(
    campaign_id: uuid.UUID,
    payload: AddContactsIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Add contacts from the contact database. Each is verified by Hunter before
    anything is drafted or sent; ``skipped`` says why any were passed over."""
    c = await _get_campaign(db, campaign_id, user)
    _require_open(c)
    ids = list(dict.fromkeys(payload.contact_ids))
    contacts = (await db.execute(select(Contact).where(Contact.id.in_(ids)))).scalars().all()
    found = {ct.id for ct in contacts}
    missing = [str(i) for i in ids if i not in found]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Contacts not found: {', '.join(missing[:5])}")
    result = await svc.add_contacts(db, c, list(contacts), source="contact_db")
    await db.commit()
    processing = _dispatch_processing(c) if result["queued"] else "nothing_queued"
    return AddProspectsOut(**result, processing=processing)


@router.post("/{campaign_id}/prospects/manual", response_model=AddProspectsOut)
async def add_manual_prospect(
    campaign_id: uuid.UUID,
    payload: ManualProspectIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Add one person by hand. They join the central contact database (merged
    with the company and person if already there) and the campaign; with no
    email given, Hunter looks for one at the company's domain. Either way the
    address is verified before anything is sent."""
    from app.services.crm_import import domain_from_email
    from app.services.ingestion import CandidateRecord, ContactRecord, get_ingestion_service
    from app.services.normalize import normalize_domain

    c = await _get_campaign(db, campaign_id, user)
    _require_open(c)
    domain = normalize_domain(payload.domain) or domain_from_email(payload.email)
    person = ContactRecord(
        first_name=payload.first_name.strip(), last_name=(payload.last_name or "").strip() or None,
        email=payload.email, title=(payload.title or "").strip() or None, source="manual",
    )
    ingest = get_ingestion_service()
    result = await ingest.ingest(db, [CandidateRecord(
        name=payload.company_name.strip(), domain=domain,
        website=f"https://{domain}" if domain else None, source="manual", contacts=[person],
    )])
    outcome = result.outcomes[0] if result.outcomes else None
    if outcome is None or outcome.company is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            (outcome.error if outcome else None) or "Couldn't save that company.",
        )
    if payload.email:
        contact = (await db.execute(
            select(Contact).where(func.lower(Contact.email) == payload.email)
            .order_by(Contact.created_at, Contact.id).limit(1)
        )).scalars().first()
    else:
        contact = await ingest._find_named_contact(db, outcome.company.id, person)
    if contact is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Couldn't save that contact.")
    added = await svc.add_contacts(db, c, [contact], source="manual")
    await db.commit()
    processing = _dispatch_processing(c) if added["queued"] else "nothing_queued"
    return AddProspectsOut(**added, processing=processing)



class SaveTemplateIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    # Also draft the rest of this campaign from it: "template" fills it in as
    # written, "ai_template" has the AI personalise it per prospect.
    use_for_campaign: Optional[Literal["template", "ai_template"]] = None


class SavedTemplateOut(BaseModel):
    id: uuid.UUID
    name: str
    subject: str
    body: str
    campaign_template_mode: str


@router.post("/{campaign_id}/prospects/{prospect_id}/save-template", response_model=SavedTemplateOut,
             status_code=status.HTTP_201_CREATED)
async def save_draft_as_template(
    campaign_id: uuid.UUID,
    prospect_id: uuid.UUID,
    payload: SaveTemplateIn,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Save a prospect's draft as a reusable email template.

    The prospect's name, company and title become merge fields, so the
    template reads right for the next person. It lands in Email Templates like
    any other, editable there, and can become this campaign's template at once.
    """
    from app.models.email_template import EmailTemplate

    c = await _get_campaign(db, campaign_id, user)
    p = await _get_prospect(db, c, prospect_id)
    if not p.email_subject or not p.email_body:
        raise HTTPException(status.HTTP_409_CONFLICT, "There is no draft to save yet.")
    name = " ".join(payload.name.split())
    clash = (await db.execute(
        select(EmailTemplate.id).where(func.lower(EmailTemplate.name) == name.lower()).limit(1)
    )).first()
    if clash is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"A template called \"{name}\" already exists.")
    contact = await db.get(Contact, p.contact_id) if p.contact_id else None
    company = await db.get(Company, p.company_id) if p.company_id else None
    t = EmailTemplate(
        name=name,
        subject=svc.generalize_draft(p.email_subject, contact, company)[:500],
        body=svc.body_to_html(svc.generalize_draft(p.email_body, contact, company)),
    )
    db.add(t)
    await db.flush()
    if payload.use_for_campaign:
        c.template_id, c.template_mode = t.id, payload.use_for_campaign
    await db.commit()
    await db.refresh(t)
    return SavedTemplateOut(id=t.id, name=t.name, subject=t.subject, body=t.body,
                            campaign_template_mode=c.template_mode)



@router.get("/{campaign_id}/attachments/file")
async def get_campaign_attachment(
    campaign_id: uuid.UUID,
    name: str = Query(..., min_length=1, max_length=255),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """The bytes of one file this campaign's emails carry, for the draft preview.

    Exactly what the send path attaches (:func:`svc._attachments`), looked up
    by filename — so the preview can never show a file the email won't carry,
    or miss one it will. PDFs are served inline for the viewer; anything else
    as a download. Never cached: drafts change, and this is a tenant's file.
    """
    from urllib.parse import quote

    from fastapi import Response

    c = await _get_campaign(db, campaign_id, user)
    match = next((a for a in await svc._attachments(db, c) if a.filename == name), None)
    if match is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This campaign doesn't attach a file by that name.")
    inline = match.content_type == "application/pdf"
    return Response(
        content=match.data,
        media_type=match.content_type,
        headers={
            "Content-Disposition": f"{'inline' if inline else 'attachment'}; filename*=UTF-8''{quote(match.filename)}",
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
