"""The MCP tools, as plain async functions over a session and a context.

Nothing here knows about MCP: each tool takes an ``AsyncSession`` and an
authenticated :class:`~app.mcp_server.auth.McpContext` and returns
JSON-serialisable data, so the tests call them directly.
:mod:`app.mcp_server.server` is the adapter that registers them.

Two rules hold throughout:

* **Writes go through the existing handlers.** ``create_campaign`` and
  ``add_leads_to_campaign`` call the ``/campaigns`` route functions themselves,
  not a re-implementation, so the document, AI-key, sender, suppression and
  already-mailed checks are the ones the UI gets. An MCP client must not be a
  way around a compliance gate.
* **Reads lean on the org filter, and say so where they don't.** The key's org
  is bound by ``authenticate``, so ORM SELECTs are scoped by
  ``OrgScopedMixin``'s ``do_orm_execute`` hook. ``get_unsubscribes`` builds a
  UNION and adds the org predicate explicitly as well.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import HTTPException
from pydantic import ValidationError
from pydantic_core import to_jsonable_python
from sqlalchemy import and_, exists, func, literal, or_, select, union_all

from app.mcp_server.auth import McpContext, ToolFailure, acting_user, detail_message
from app.models.campaign import Campaign, CampaignDailyLog, CampaignProspect
from app.models.company import Company
from app.models.contact import Contact
from app.models.discovery_document import DiscoveryDocument
from app.models.email_template import EmailSend
from app.models.gmail_account import GmailAccount
from app.models.outreach import OutreachEnrollment
from app.models.smtp_account import SmtpAccount
from app.models.suppression import SUPPRESSION_REASONS, SuppressionEntry
from app.models.user import User
from app.services import suppression
from app.services.last_contact import last_send_by_contact
from app.services.sql_search import LIKE_ESCAPE, contains_pattern

MAX_PAGE = 100
PIPELINE_STAGES = (
    "new_lead", "researching", "outreach", "engaged", "negotiating", "client", "churned",
)


def _json(value: Any) -> Any:
    return to_jsonable_python(value)


def _uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, TypeError):
        raise ToolFailure(f"'{value}' is not a valid {what} id.") from None


def _page(limit: int, offset: int) -> tuple[int, int]:
    return max(1, min(int(limit), MAX_PAGE)), max(0, int(offset))


async def _handler(coro):
    """Await a route handler, turning its HTTP refusals into tool failures."""
    try:
        return await coro
    except HTTPException as exc:
        raise ToolFailure(detail_message(exc.detail)) from None


# ---------------------------------------------------------------------------
# Leads
# ---------------------------------------------------------------------------

def _company_out(co: Optional[Company]) -> Optional[dict]:
    if co is None:
        return None
    return {
        "id": co.id,
        "name": co.name,
        "domain": co.domain,
        "industry_segment": co.industry_segment,
        "pipeline_stage": co.pipeline_stage,
        "website": co.website,
        "city": co.city,
        "state_province": co.state_province,
        "country": co.country,
    }


def _lead_summary(ct: Contact, co: Optional[Company]) -> dict:
    return {
        "id": ct.id,
        "first_name": ct.first_name,
        "last_name": ct.last_name,
        "title": ct.title,
        "email": ct.email,
        "phone": ct.phone,
        "company": _company_out(co),
        "created_at": ct.created_at,
    }


async def search_leads(
    db,
    ctx: McpContext,
    *,
    query: Optional[str] = None,
    company_id: Optional[str] = None,
    pipeline_stage: Optional[str] = None,
    has_email: Optional[bool] = None,
    limit: int = 25,
    offset: int = 0,
) -> dict:
    ctx.require("contacts:read")
    limit, offset = _page(limit, offset)
    # Erased subjects keep a row (anonymised); they are not leads.
    filters: list = [Contact.anonymized_at.is_(None)]
    if query and query.strip():
        # Every word must match some field, so "ada acme" finds Ada at Acme.
        for term in query.split():
            pattern = contains_pattern(term)
            filters.append(or_(
                *(col.ilike(pattern, escape=LIKE_ESCAPE) for col in (
                    Contact.first_name, Contact.last_name, Contact.email,
                    Contact.title, Company.name, Company.domain,
                ))
            ))
    if company_id:
        filters.append(Contact.company_id == _uuid(company_id, "company"))
    if pipeline_stage:
        if pipeline_stage not in PIPELINE_STAGES:
            raise ToolFailure(f"pipeline_stage must be one of: {', '.join(PIPELINE_STAGES)}")
        filters.append(Company.pipeline_stage == pipeline_stage)
    if has_email is True:
        filters.append(and_(Contact.email.isnot(None), Contact.email != ""))
    elif has_email is False:
        filters.append(or_(Contact.email.is_(None), Contact.email == ""))

    total = (await db.execute(
        select(func.count(Contact.id))
        .select_from(Contact)
        .outerjoin(Company, Company.id == Contact.company_id)
        .where(*filters)
    )).scalar_one()
    rows = (await db.execute(
        select(Contact, Company)
        .outerjoin(Company, Company.id == Contact.company_id)
        .where(*filters)
        # A total order: bulk imports share one created_at to the microsecond.
        .order_by(Contact.created_at.desc(), Contact.id.desc())
        .offset(offset).limit(limit)
    )).all()
    return _json({
        "total": total,
        "offset": offset,
        "limit": limit,
        "leads": [_lead_summary(ct, co) for ct, co in rows],
    })


async def get_lead_details(db, ctx: McpContext, *, lead_id: str) -> dict:
    ctx.require("contacts:read")
    cid = _uuid(lead_id, "lead")
    ct = (await db.execute(select(Contact).where(Contact.id == cid).limit(1))).scalars().first()
    if ct is None:
        raise ToolFailure("Lead not found.")
    co = None
    if ct.company_id is not None:
        co = (await db.execute(
            select(Company).where(Company.id == ct.company_id).limit(1)
        )).scalars().first()

    last = last_send_by_contact()
    last_emailed_at = (await db.execute(
        select(last.c.last_sent_at).where(last.c.contact_id == ct.id)
    )).scalars().first()

    campaigns = (await db.execute(
        select(CampaignProspect, Campaign.name)
        .join(Campaign, Campaign.id == CampaignProspect.campaign_id)
        .where(CampaignProspect.contact_id == ct.id, Campaign.deleted_at.is_(None))
        .order_by(CampaignProspect.created_at.desc(), CampaignProspect.id)
    )).all()
    enrollments = (await db.execute(
        select(OutreachEnrollment)
        .where(OutreachEnrollment.contact_id == ct.id)
        .order_by(OutreachEnrollment.enrolled_at.desc(), OutreachEnrollment.id)
    )).scalars().all()

    blocked = await suppression.blocking_reason(db, ct)
    lead = _lead_summary(ct, co)
    lead.update({
        "department": ct.department,
        "mobile": ct.mobile,
        "linkedin_url": ct.linkedin_url,
        "website": ct.website,
        "city": ct.city,
        "state": ct.state,
        "country": ct.country,
        "timezone": ct.timezone,
        "notes": ct.notes,
        "tags": ct.tags or [],
        "source": ct.source,
        "status": ct.status,
        "is_primary": ct.is_primary,
        "email_verification": {
            "result": ct.email_verification_result,
            "score": ct.email_verification_score,
            "verified_at": ct.email_verified_at,
        },
        "bounce_count": ct.bounce_count,
        "updated_at": ct.updated_at,
        # Across both send tables (EmailSend and sequence OutreachEvents).
        "last_emailed_at": last_emailed_at,
        "mailable": blocked is None,
        "do_not_contact_reason": blocked,
        "campaigns": [
            {
                "campaign_id": p.campaign_id,
                "campaign_name": name,
                "status": p.status,
                "sent_at": p.sent_at,
                "replied_at": p.replied_at,
            }
            for p, name in campaigns
        ],
        "sequence_enrollments": [
            {
                "sequence_id": e.sequence_id,
                "status": e.status,
                "current_step": e.current_step,
                "next_send_at": e.next_send_at,
            }
            for e in enrollments
        ],
    })
    return _json(lead)


# ---------------------------------------------------------------------------
# Campaigns
# ---------------------------------------------------------------------------

async def _campaign(db, campaign_id: str) -> Campaign:
    cid = _uuid(campaign_id, "campaign")
    c = (await db.execute(
        select(Campaign).where(Campaign.id == cid, Campaign.deleted_at.is_(None)).limit(1)
    )).scalars().first()
    if c is None:
        raise ToolFailure("Campaign not found.")
    return c


def _campaign_summary(c: Campaign) -> dict:
    from app.routers import campaigns as campaigns_router

    out = campaigns_router._out(c)
    # The ICP and document summary are long; list_campaigns stays scannable.
    out.pop("icp_json", None)
    out.pop("document_summary", None)
    for legacy in ("window_open", "send_window_opens_at"):
        out.pop(legacy, None)
    return out


async def list_discovery_documents(db, ctx: McpContext, *, status: Optional[str] = None) -> dict:
    ctx.require("campaigns:read")
    stmt = select(
        DiscoveryDocument.id, DiscoveryDocument.filename, DiscoveryDocument.file_type,
        DiscoveryDocument.status, DiscoveryDocument.error, DiscoveryDocument.created_at,
    )
    if status:
        stmt = stmt.where(DiscoveryDocument.status == status)
    rows = (await db.execute(
        stmt.order_by(DiscoveryDocument.created_at.desc(), DiscoveryDocument.id)
    )).all()
    return _json({"documents": [dict(r._mapping) for r in rows]})


async def list_campaigns(db, ctx: McpContext, *, status: Optional[str] = None) -> dict:
    ctx.require("campaigns:read")
    stmt = select(Campaign).where(Campaign.deleted_at.is_(None))
    if status:
        stmt = stmt.where(Campaign.status == status)
    rows = (await db.execute(
        stmt.order_by(Campaign.created_at.desc(), Campaign.id)
    )).scalars().all()
    return _json({"campaigns": [_campaign_summary(c) for c in rows]})


async def get_campaign_stats(db, ctx: McpContext, *, campaign_id: str) -> dict:
    ctx.require("campaigns:read")
    from app.routers import campaigns as campaigns_router

    c = await _campaign(db, campaign_id)
    metrics = await campaigns_router._metrics(db, c)

    # Opens and clicks live on the EmailSend row each campaign send writes.
    opened, clicked = (await db.execute(
        select(func.count(EmailSend.opened_at), func.count(EmailSend.clicked_at))
        .select_from(CampaignProspect)
        .join(EmailSend, EmailSend.id == CampaignProspect.email_send_id)
        .where(CampaignProspect.campaign_id == c.id)
    )).one()
    sent = metrics["emails_sent"]
    metrics.update({
        "opens": opened,
        "clicks": clicked,
        "open_rate": round(opened / sent, 4) if sent else 0.0,
        "click_rate": round(clicked / sent, 4) if sent else 0.0,
    })
    daily = (await db.execute(
        select(CampaignDailyLog)
        .where(CampaignDailyLog.campaign_id == c.id)
        .order_by(CampaignDailyLog.date.desc(), CampaignDailyLog.id)
        .limit(14)
    )).scalars().all()
    return _json({
        "campaign": {"id": c.id, "name": c.name, "status": c.status, "pause_reason": c.pause_reason},
        "stats": metrics,
        "daily": [
            {
                "date": d.date,
                "prospects_found": d.prospects_found,
                "emails_sent": d.emails_sent,
                "emails_bounced": d.emails_bounced,
            }
            for d in daily
        ],
        "note": "Opens come from a tracking pixel, so image blocking and mail "
                "privacy proxies make them approximate. Replies are exact.",
    })


async def create_campaign(
    db,
    ctx: McpContext,
    *,
    name: str,
    document_ids: list[str],
    sender_account_id: Optional[str] = None,
    daily_send_limit: Optional[int] = None,
    review_before_send: Optional[bool] = None,
    send_window_start: Optional[str] = None,
    send_window_end: Optional[str] = None,
    timezone: Optional[str] = None,
    send_on_weekends: Optional[bool] = None,
    template_id: Optional[str] = None,
) -> dict:
    ctx.require("campaigns:write")
    from app.routers import campaigns as campaigns_router

    user = await acting_user(db, ctx)
    fields = {
        "name": name,
        "document_ids": document_ids,
        "sender_account_id": sender_account_id,
        "daily_send_limit": daily_send_limit,
        "review_before_send": review_before_send,
        "send_window_start": send_window_start,
        "send_window_end": send_window_end,
        "timezone": timezone,
        "send_on_weekends": send_on_weekends,
    }
    # template_id is "unset vs. explicit" to the handler: only pass it if given.
    if template_id is not None:
        fields["template_id"] = template_id
    try:
        payload = campaigns_router.CampaignCreate(**fields)
    except ValidationError as exc:
        raise ToolFailure(_validation_message(exc)) from None
    out = await _handler(campaigns_router.create_campaign(payload=payload, db=db, user=user))
    out = dict(out)
    out["next_steps"] = (
        "The campaign is live and starts finding prospects now."
        + (" Drafts wait for review in the Prospect app before anything is sent."
           if out.get("review_before_send") else "")
    )
    return _json(out)


async def add_leads_to_campaign(db, ctx: McpContext, *, campaign_id: str, lead_ids: list[str]) -> dict:
    ctx.require("campaigns:write")
    from app.routers import campaigns as campaigns_router

    user = await acting_user(db, ctx)
    cid = _uuid(campaign_id, "campaign")
    try:
        payload = campaigns_router.AddContactsIn.model_validate({"contact_ids": lead_ids})
    except ValidationError as exc:
        raise ToolFailure(_validation_message(exc)) from None
    out = await _handler(campaigns_router.add_contacts_to_campaign(
        campaign_id=cid, payload=payload, db=db, user=user,
    ))
    return _json(out.model_dump())


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "Invalid input — " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Mailboxes
# ---------------------------------------------------------------------------

async def search_mailboxes(
    db,
    ctx: McpContext,
    *,
    query: Optional[str] = None,
    status: Optional[str] = None,
    sending_enabled: Optional[bool] = None,
) -> dict:
    """Connected mailboxes. Never returns tokens or passwords — the columns are
    named one by one below rather than serialising the row."""
    ctx.require("mailboxes:read")
    from app.dependencies import scope_users_to_current_org
    from app.services.warmup import used_today

    out: list[dict] = []
    for kind, model in (("gmail", GmailAccount), ("smtp", SmtpAccount)):
        stmt = select(model)
        if query and query.strip():
            pattern = contains_pattern(query.strip())
            stmt = stmt.where(or_(
                model.email.ilike(pattern, escape=LIKE_ESCAPE),
                model.display_name.ilike(pattern, escape=LIKE_ESCAPE),
            ))
        if status:
            stmt = stmt.where(model.status == status)
        if sending_enabled is not None:
            stmt = stmt.where(model.sending_enabled.is_(sending_enabled))
        for a in (await db.execute(stmt.order_by(model.email, model.id))).scalars().all():
            row = {
                "id": a.id,
                "kind": kind,
                "email": a.email,
                "display_name": a.display_name,
                "status": a.status,
                "sending_enabled": a.sending_enabled,
                "is_default": a.is_active,
                "sent_today": used_today(a),
                "connected_at": a.connected_at,
                "last_used_at": a.last_used_at,
                "owner_user_id": a.user_id,
            }
            if kind == "smtp":
                row["last_error"] = a.last_error
            out.append(row)

    owner_ids = {r["owner_user_id"] for r in out if r["owner_user_id"]}
    owners = {}
    if owner_ids:
        owners = {
            uid: email for uid, email in (await db.execute(
                scope_users_to_current_org(
                    select(User.id, User.email).where(User.id.in_(owner_ids))
                )
            )).all()
        }
    for r in out:
        r["owner_email"] = owners.get(r["owner_user_id"])
    return _json({"mailboxes": out})


# ---------------------------------------------------------------------------
# Unsubscribes
# ---------------------------------------------------------------------------

async def get_unsubscribes(
    db,
    ctx: McpContext,
    *,
    reason: str = "unsubscribe",
    query: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """The org's do-not-contact list, unsubscribes by default.

    Two sources, because opt-outs recorded by the old link handler set
    ``contact.unsubscribed`` and wrote no list entry. Those rows are still in
    the database, so a contact flagged that way is listed (once per address)
    unless the list already has the address.
    """
    ctx.require("suppression:read")
    limit, offset = _page(limit, offset)
    if reason != "all" and reason not in SUPPRESSION_REASONS:
        raise ToolFailure(f"reason must be 'all' or one of: {', '.join(SUPPRESSION_REASONS)}")
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise ToolFailure("since must be an ISO date, e.g. 2026-10-01") from None

    # Explicit org predicates: this is a UNION of two tables, and the list is the
    # one surface where a cross-tenant row would be a disclosure of who opted out.
    entry_filters: list = [SuppressionEntry.org_id == ctx.org_id]
    if reason != "all":
        entry_filters.append(SuppressionEntry.reason == reason)
    if since_dt is not None:
        entry_filters.append(SuppressionEntry.created_at >= since_dt)
    if query and query.strip():
        entry_filters.append(
            SuppressionEntry.email.ilike(contains_pattern(query.strip()), escape=LIKE_ESCAPE)
        )
    entries = select(
        func.lower(SuppressionEntry.email).label("email"),
        SuppressionEntry.reason.label("reason"),
        SuppressionEntry.source.label("source"),
        SuppressionEntry.created_at.label("recorded_at"),
    ).where(*entry_filters)

    parts = [entries]
    if reason in ("all", "unsubscribe"):
        listed = exists().where(
            SuppressionEntry.org_id == ctx.org_id,
            func.lower(SuppressionEntry.email) == func.lower(Contact.email),
        )
        legacy_filters: list = [
            Contact.org_id == ctx.org_id,
            Contact.unsubscribed.is_(True),
            Contact.email.isnot(None),
            Contact.email != "",
            ~listed,
        ]
        if since_dt is not None:
            legacy_filters.append(Contact.updated_at >= since_dt)
        if query and query.strip():
            legacy_filters.append(
                Contact.email.ilike(contains_pattern(query.strip()), escape=LIKE_ESCAPE)
            )
        # contacts.email is not unique: one row per address, not per duplicate.
        parts.append(
            select(
                func.lower(Contact.email).label("email"),
                literal("unsubscribe").label("reason"),
                literal("contact_flag").label("source"),
                func.max(Contact.updated_at).label("recorded_at"),
            )
            .where(*legacy_filters)
            .group_by(func.lower(Contact.email))
        )
    both = (union_all(*parts) if len(parts) > 1 else parts[0]).subquery()

    total = (await db.execute(select(func.count()).select_from(both))).scalar_one()
    rows = (await db.execute(
        select(both)
        .order_by(both.c.recorded_at.desc(), both.c.email)
        .offset(offset).limit(limit)
    )).all()
    return _json({
        "total": total,
        "offset": offset,
        "limit": limit,
        "reason": reason,
        "entries": [dict(r._mapping) for r in rows],
    })
