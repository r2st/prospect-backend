"""Company CRUD, search, and pipeline management endpoints."""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, or_, and_, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.constants import COUNTRY_QUERY_PATTERN
from app.database import get_db
from app.models.company import Company
from app.models.contact import Contact
from app.models.lead_score import LeadScore
from app.org_context import scope_write_to_current_org
from app.schemas.company import (
    BulkExportRequest,
    BulkOpResult,
    BulkTagRequest,
    BulkUpdateStageRequest,
    CompanyContact,
    CompanyCreate,
    CompanyDetailRead,
    CompanyListResponse,
    CompanyMergeRequest,
    CompanyMergeResponse,
    CompanyRead,
    CompanyStageUpdate,
    CompanyUpdate,
)
from app.services.company_contacts import (
    display_name,
    pick_primary,
    rank_contacts,
)
from app.services.sql_search import LIKE_ESCAPE, contains_pattern

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/companies", tags=["companies"])


def _to_company_contact(contact: Contact) -> CompanyContact:
    """Contact row → the summary shape a company response carries."""
    return CompanyContact.model_validate(contact).model_copy(
        update={"full_name": display_name(contact)}
    )


async def _attach_extras(db: AsyncSession, companies: list[Company]) -> None:
    """Attach derived ``fit_score``, ``contact_count`` and ``primary_contact``
    onto company instances so they serialize into CompanyRead. Bulk-queried to
    avoid N+1.

    Contacts themselves need no query here: ``Company.contacts`` is a
    ``lazy="selectin"`` relationship, so they are already loaded.
    """
    if not companies:
        return
    ids = [c.id for c in companies]

    # Latest rule-based fit score per company.
    latest = (
        select(
            LeadScore.company_id,
            func.max(LeadScore.scored_at).label("max_scored_at"),
        )
        .where(LeadScore.company_id.in_(ids), LeadScore.score_type == "rule_based")
        .group_by(LeadScore.company_id)
        .subquery()
    )
    score_rows = await db.execute(
        select(LeadScore.company_id, LeadScore.fit_score)
        .join(
            latest,
            and_(
                LeadScore.company_id == latest.c.company_id,
                LeadScore.scored_at == latest.c.max_scored_at,
            ),
        )
        .where(LeadScore.score_type == "rule_based")
    )
    score_by_company = {row[0]: row[1] for row in score_rows.all()}

    for c in companies:
        c.fit_score = score_by_company.get(c.id)
        # The count and the fronted contact come from the same rows, so the
        # number and the name can never disagree.
        ordered = rank_contacts(c.contacts)
        c.contact_count = len(ordered)
        primary = pick_primary(ordered)
        c.primary_contact = _to_company_contact(primary) if primary else None


# Fields the lead scorer reads. A PATCH touching only notes or stage doesn't
# change the lead, so it doesn't write a new score row.
_SCORING_FIELDS = frozenset({
    "name", "domain", "city", "state_province", "country", "employee_count",
    "naics_code", "naics_description", "industry_segment", "website", "tags",
})


async def _score_best_effort(db: AsyncSession, company: Company) -> None:
    """Score a lead after it is created or its firmographics change.

    Rule-based fit plus Jev's hot/warm/cold temperature (when configured). Under
    a savepoint, and never raising: a scoring failure must not lose the write
    the caller asked for. Only CSV imports and ingestion scored new companies
    before this; a lead added by hand sat unscored until the nightly rescore.
    """
    from app.services.scoring import score_company_rule_based

    try:
        async with db.begin_nested():
            await score_company_rule_based(db, company)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lead scoring failed for %s: %s", company.id, exc)


@router.post("/", response_model=CompanyRead, status_code=201)
async def create_company(
    payload: CompanyCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new company record."""
    company = Company(**payload.model_dump())
    db.add(company)
    await db.flush()
    await db.refresh(company)
    logger.info("Created company %s (%s)", company.name, company.id)
    await _score_best_effort(db, company)

    # AI Company Research Agent (feature 1.4): kick off a research brief on
    # creation. Best-effort — if the broker is unreachable, the periodic
    # research_stale_companies task backfills it. Gated by a settings flag.
    if get_settings().auto_research_enabled:
        try:
            from app.tasks.research import research_company

            research_company.delay(str(company.id))
        except Exception as exc:  # pragma: no cover - broker-dependent
            logger.warning("Could not dispatch research for %s: %s", company.id, exc)

    return company


def _serialize_research(r) -> dict:
    """Serialize a CompanyResearch row for the API."""
    return {
        "id": str(r.id),
        "company_id": str(r.company_id),
        "research_brief": r.research_brief,
        "pain_points": r.pain_points or [],
        "outreach_angles": r.outreach_angles or [],
        "competitive_intel": r.competitive_intel or {},
        "sources_used": r.sources_used or [],
        "is_stale": r.is_stale,
        "model_used": r.model_used,
        "researched_at": r.researched_at.isoformat() if r.researched_at else None,
    }


@router.get("/{company_id}/research")
async def get_company_research(
    company_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return the freshest AI research brief for a company (feature 1.4).

    Returns ``{"status": "none"}`` when no brief exists yet — trigger one with
    ``POST /companies/{id}/research``.
    """
    from app.services.research_agent import get_active_research

    company = (
        await db.execute(select(Company).where(Company.id == company_id))
    ).scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    research = await get_active_research(db, company_id)
    if research is None:
        return {"status": "none", "company_id": str(company_id)}
    return {"status": "ok", **_serialize_research(research)}


@router.post("/{company_id}/research")
async def generate_company_research(
    company_id: uuid.UUID,
    force: bool = Query(False, description="Re-research even if a fresh brief exists"),
    db: AsyncSession = Depends(get_db),
):
    """Run the AI Company Research Agent for a company now and return the brief.

    Synchronous — scrapes the site, pulls news/enrichment, and synthesizes with
    Claude. Use ``force=true`` to supersede an existing non-stale brief.
    """
    from app.services.research_agent import get_research_agent

    company = (
        await db.execute(select(Company).where(Company.id == company_id))
    ).scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    research = await get_research_agent().research_company(db, company, force=force)
    if research is None:
        raise HTTPException(status_code=502, detail="Research synthesis failed")
    return {"status": "ok", **_serialize_research(research)}


@router.get("/", response_model=CompanyListResponse)
async def list_companies(
    q: Optional[str] = Query(None, description="Search name, domain, or industry"),
    country: Optional[str] = Query(None, pattern=COUNTRY_QUERY_PATTERN),
    stage: Optional[str] = Query(None),
    pipeline_stage: Optional[str] = Query(None, description="Kanban pipeline stage"),
    naics_code: Optional[str] = Query(None),
    industry_segment: Optional[str] = Query(None),
    min_employees: Optional[int] = Query(None, ge=0),
    max_employees: Optional[int] = Query(None, ge=0),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    has_contacts: Optional[bool] = Query(None, description="Only companies with/without contacts"),
    enriched: Optional[str] = Query(None, description="enriched or not_enriched"),
    source: Optional[str] = Query(None, description="Filter by source"),
    sort_by: Optional[str] = Query("created_at", description="Sort field"),
    sort_dir: Optional[str] = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List companies with filtering, search, and pagination."""
    query = select(Company)
    count_query = select(func.count(Company.id))

    # Filters
    conditions = []
    if q:
        search = contains_pattern(q)
        conditions.append(
            or_(
                Company.name.ilike(search, escape=LIKE_ESCAPE),
                Company.domain.ilike(search, escape=LIKE_ESCAPE),
                Company.industry_segment.ilike(search, escape=LIKE_ESCAPE),
                Company.naics_description.ilike(search, escape=LIKE_ESCAPE),
            )
        )
    if country:
        conditions.append(Company.country == country)
    if stage:
        conditions.append(Company.stage == stage)
    if pipeline_stage:
        conditions.append(Company.pipeline_stage == pipeline_stage)
    if naics_code:
        conditions.append(Company.naics_code.startswith(naics_code, autoescape=True))
    if industry_segment:
        conditions.append(
            Company.industry_segment.ilike(
                contains_pattern(industry_segment), escape=LIKE_ESCAPE
            )
        )
    if min_employees is not None:
        conditions.append(Company.employee_count >= min_employees)
    if max_employees is not None:
        conditions.append(Company.employee_count <= max_employees)
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        for tag in tag_list:
            conditions.append(Company.tags.any(tag))

    # has_contacts filter: correlated subquery checking for contact existence
    if has_contacts is not None:
        contact_exists = (
            select(func.count(Contact.id))
            .where(Contact.company_id == Company.id)
            .correlate(Company)
            .scalar_subquery()
        )
        if has_contacts:
            conditions.append(contact_exists > 0)
        else:
            conditions.append(contact_exists == 0)

    # enriched filter: companies with/without a domain
    if enriched is not None:
        if enriched == "enriched":
            conditions.append(Company.domain.isnot(None))
        elif enriched == "not_enriched":
            conditions.append(Company.domain.is_(None))

    # source filter: exact match
    if source:
        conditions.append(Company.source == source)

    if conditions:
        combined = and_(*conditions)
        query = query.where(combined)
        count_query = count_query.where(combined)

    # Count
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Sort — whitelist allowed columns to prevent attribute access abuse
    ALLOWED_SORTS = {"name", "created_at", "stage", "pipeline_stage", "country", "employee_count", "city", "domain"}
    sort_column = getattr(Company, sort_by) if sort_by in ALLOWED_SORTS else Company.created_at
    if sort_dir == "asc":
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    # Paginate
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    result = await db.execute(query)
    companies = list(result.scalars().all())

    await _attach_extras(db, companies)

    return CompanyListResponse(
        items=companies,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/bulk-update-stage", response_model=BulkOpResult)
async def bulk_update_stage(
    payload: BulkUpdateStageRequest,
    db: AsyncSession = Depends(get_db),
):
    """Move a set of companies to a new kanban pipeline stage in one call.

    The ids come straight off the request body, and a Core ``update()`` gets no
    ``org_id`` predicate from the global read filter — so this moved any company
    on the platform to any stage, for any authenticated caller. Ids are not
    secrets; they come back in exports, webhooks and error text.

    ``bulk_tag`` directly below takes the same client-supplied ids and was never
    exposed, because it routes through ``select(Company)`` and mutates the ORM
    objects it gets back — the SELECT is scoped, so foreign ids simply don't come
    back. That is the whole difference between the two endpoints.
    """
    result = await db.execute(
        scope_write_to_current_org(
            update(Company).where(Company.id.in_(payload.company_ids)), Company
        ).values(pipeline_stage=payload.pipeline_stage)
    )
    # Report how many of the requested ids actually exist vs were updated.
    matched_result = await db.execute(
        select(func.count(Company.id)).where(Company.id.in_(payload.company_ids))
    )
    matched = matched_result.scalar() or 0
    await db.flush()
    logger.info(
        "Bulk pipeline-stage -> %s for %d companies", payload.pipeline_stage, result.rowcount or 0
    )
    return BulkOpResult(matched=matched, updated=result.rowcount or 0)


@router.post("/bulk-tag", response_model=BulkOpResult)
async def bulk_tag(
    payload: BulkTagRequest,
    db: AsyncSession = Depends(get_db),
):
    """Add and/or remove tags across a set of companies in one call.

    Tags to add are unioned in; tags to remove are stripped out. A company whose
    tag set is unchanged is counted as matched but not updated.
    """
    if not payload.add and not payload.remove:
        raise HTTPException(status_code=400, detail="Provide tags to add and/or remove")

    add = [t.strip() for t in payload.add if t.strip()]
    remove = {t.strip() for t in payload.remove if t.strip()}

    result = await db.execute(
        select(Company).where(Company.id.in_(payload.company_ids))
    )
    companies = list(result.scalars().all())

    updated = 0
    for company in companies:
        current = list(company.tags or [])
        new_tags = [t for t in current if t not in remove]
        for t in add:
            if t not in new_tags:
                new_tags.append(t)
        if new_tags != current:
            company.tags = new_tags
            updated += 1

    await db.flush()
    logger.info(
        "Bulk-tag: +%s -%s across %d companies (%d changed)",
        add, sorted(remove), len(companies), updated,
    )
    return BulkOpResult(matched=len(companies), updated=updated)


@router.post("/bulk-export")
async def bulk_export(
    payload: BulkExportRequest,
    db: AsyncSession = Depends(get_db),
):
    """Export companies as structured JSON with all fields, optionally including
    each company's contacts, lead scores, and full activity timeline.

    When ``company_ids`` is omitted, every company is exported.
    """
    from datetime import datetime, timezone

    query = select(Company)
    if payload.company_ids:
        query = query.where(Company.id.in_(payload.company_ids))
    query = query.order_by(Company.created_at.desc())
    companies = list((await db.execute(query)).scalars().all())

    company_ids = [c.id for c in companies]

    # Bulk-load contacts and scores to avoid per-company round-trips.
    contacts_by_company: dict = {}
    if payload.include_contacts and company_ids:
        rows = await db.execute(
            select(Contact).where(Contact.company_id.in_(company_ids))
        )
        for c in rows.scalars().all():
            contacts_by_company.setdefault(c.company_id, []).append(c)

    scores_by_company: dict = {}
    if payload.include_scores and company_ids:
        rows = await db.execute(
            select(LeadScore)
            .where(LeadScore.company_id.in_(company_ids))
            .order_by(LeadScore.scored_at.desc())
        )
        for s in rows.scalars().all():
            scores_by_company.setdefault(s.company_id, []).append(s)

    def _company_dict(c: Company) -> dict:
        return {
            "id": str(c.id),
            "name": c.name,
            "domain": c.domain,
            "city": c.city,
            "state_province": c.state_province,
            "country": c.country,
            "employee_count": c.employee_count,
            "naics_code": c.naics_code,
            "naics_description": c.naics_description,
            "industry_segment": c.industry_segment,
            "website": c.website,
            "phone": c.phone,
            "source": c.source,
            "source_url": c.source_url,
            "stage": c.stage,
            "pipeline_stage": c.pipeline_stage,
            "tags": list(c.tags or []),
            "notes": c.notes,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }

    def _contact_dict(c: Contact) -> dict:
        return {
            "id": str(c.id),
            "first_name": c.first_name,
            "last_name": c.last_name,
            "full_name": c.full_name,
            "title": c.title,
            "department": c.department,
            "email": c.email,
            "phone": c.phone,
            "mobile": c.mobile,
            "linkedin_url": c.linkedin_url,
            "is_primary": c.is_primary,
            "unsubscribed": c.unsubscribed,
            "source": c.source,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }

    def _score_dict(s: LeadScore) -> dict:
        return {
            "id": str(s.id),
            "score_type": s.score_type,
            "fit_score": s.fit_score,
            "industry_score": s.industry_score,
            "size_score": s.size_score,
            "location_score": s.location_score,
            "engagement_score": s.engagement_score,
            "factors": s.factors,
            "scored_at": s.scored_at.isoformat() if s.scored_at else None,
        }

    exported = []
    for c in companies:
        record = _company_dict(c)
        if payload.include_contacts:
            record["contacts"] = [
                _contact_dict(ct) for ct in contacts_by_company.get(c.id, [])
            ]
        if payload.include_scores:
            record["scores"] = [
                _score_dict(s) for s in scores_by_company.get(c.id, [])
            ]
        if payload.include_activity:
            from app.routers.activity import _collect_activities
            items = await _collect_activities(db, c.id)
            items.sort(key=lambda i: i.timestamp, reverse=True)
            record["activity"] = [i.model_dump(mode="json") for i in items]
        exported.append(record)

    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(exported),
        "companies": exported,
    }


@router.get("/{company_id}", response_model=CompanyDetailRead)
async def get_company(
    company_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single company by ID, with every contact attached."""
    result = await db.execute(select(Company).where(Company.id == company_id))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    await _attach_extras(db, [company])
    # Built explicitly rather than by attribute injection: ``contacts`` is a
    # mapped relationship, so assigning schema objects onto it would corrupt
    # the ORM instance.
    return CompanyDetailRead.model_validate(company).model_copy(
        update={
            "contacts": [_to_company_contact(ct) for ct in rank_contacts(company.contacts)]
        }
    )


@router.patch("/{company_id}", response_model=CompanyRead)
async def update_company(
    company_id: uuid.UUID,
    payload: CompanyUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update company fields."""
    result = await db.execute(select(Company).where(Company.id == company_id))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "stage" in update_data:
        from app.schemas.company import VALID_STAGES
        if update_data["stage"] not in VALID_STAGES:
            raise HTTPException(status_code=400, detail=f"Invalid stage. Must be one of: {VALID_STAGES}")

    for field, value in update_data.items():
        setattr(company, field, value)

    await db.flush()
    if _SCORING_FIELDS & update_data.keys():
        await _score_best_effort(db, company)
    await db.refresh(company)
    logger.info("Updated company %s", company_id)
    return company


@router.patch("/{company_id}/stage", response_model=CompanyRead)
async def update_company_stage(
    company_id: uuid.UUID,
    payload: CompanyStageUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Transition a company to a new pipeline stage."""
    result = await db.execute(select(Company).where(Company.id == company_id))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    old_stage = company.stage
    company.stage = payload.stage
    await db.flush()
    await db.refresh(company)
    logger.info("Company %s stage: %s -> %s", company_id, old_stage, payload.stage)
    return company


@router.delete("/{company_id}", status_code=204)
async def delete_company(
    company_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a company and all related data (cascading)."""
    result = await db.execute(select(Company).where(Company.id == company_id))
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    await db.delete(company)
    await db.flush()
    logger.info("Deleted company %s (%s)", company.name, company_id)


@router.post("/{company_id}/merge", response_model=CompanyMergeResponse)
async def merge_company(
    company_id: uuid.UUID,
    payload: CompanyMergeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Merge a duplicate company (``source_id``) into the company in the path.

    Reassigns all related records to the survivor, backfills any empty survivor
    fields from the source, then deletes the source. Idempotent guard: refuses
    to merge a company into itself.
    """
    if payload.source_id == company_id:
        raise HTTPException(status_code=400, detail="Cannot merge a company into itself")

    survivor = (await db.execute(select(Company).where(Company.id == company_id))).scalar_one_or_none()
    if not survivor:
        raise HTTPException(status_code=404, detail="Survivor company not found")
    source = (await db.execute(select(Company).where(Company.id == payload.source_id))).scalar_one_or_none()
    if not source:
        raise HTTPException(status_code=404, detail="Source company not found")

    from app.models.interaction import Interaction
    from app.models.company_news import CompanyNews
    from app.models.follow_up import FollowUp
    from app.models.company_group import company_group_members

    async def _reassign(model, fk="company_id") -> int:
        result = await db.execute(
            update(model)
            .where(getattr(model, fk) == source.id)
            .values(**{fk: survivor.id})
        )
        return result.rowcount or 0

    contacts_moved = await _reassign(Contact)
    scores_moved = await _reassign(LeadScore)
    news_moved = await _reassign(CompanyNews)
    interactions_moved = await _reassign(Interaction)
    follow_ups_moved = await _reassign(FollowUp)

    # Group memberships: move only those the survivor isn't already in (composite PK).
    existing_groups = await db.execute(
        select(company_group_members.c.group_id).where(
            company_group_members.c.company_id == survivor.id
        )
    )
    survivor_group_ids = {r[0] for r in existing_groups.all()}
    src_groups = await db.execute(
        select(company_group_members.c.group_id).where(
            company_group_members.c.company_id == source.id
        )
    )
    for (gid,) in src_groups.all():
        if gid in survivor_group_ids:
            continue
        await db.execute(
            update(company_group_members)
            .where(
                company_group_members.c.group_id == gid,
                company_group_members.c.company_id == source.id,
            )
            .values(company_id=survivor.id)
        )

    # Snapshot the source's values, then delete it BEFORE writing them onto the
    # survivor — otherwise the survivor UPDATE could flush a duplicate `domain`
    # before the source DELETE and trip the unique index.
    backfill_fields = (
        "domain", "city", "state_province", "employee_count", "naics_code",
        "naics_description", "industry_segment", "website", "phone", "source_url",
    )
    source_values = {f: getattr(source, f) for f in backfill_fields}
    source_tags = list(source.tags or [])

    # Delete via a Core statement (not db.delete) so the ORM's selectin-loaded
    # Company.contacts/lead_scores collections don't cascade-delete the rows we
    # just reassigned to the survivor.
    db.expunge(source)
    await db.execute(delete(Company).where(Company.id == payload.source_id))
    await db.flush()

    for field in backfill_fields:
        if not getattr(survivor, field) and source_values[field]:
            setattr(survivor, field, source_values[field])
    if source_tags:
        survivor.tags = sorted(set(survivor.tags or []) | set(source_tags))

    await db.flush()
    await db.refresh(survivor)
    await _attach_extras(db, [survivor])
    logger.info("Merged company %s into %s", source.id, survivor.id)
    return CompanyMergeResponse(
        survivor_id=survivor.id,
        merged_id=source.id,
        contacts_moved=contacts_moved,
        scores_moved=scores_moved,
        news_moved=news_moved,
        interactions_moved=interactions_moved,
        follow_ups_moved=follow_ups_moved,
    )


@router.get("/{company_id}/scores")
async def get_company_scores(
    company_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get all lead scores for a company."""
    result = await db.execute(select(Company).where(Company.id == company_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Company not found")

    scores_result = await db.execute(
        select(LeadScore)
        .where(LeadScore.company_id == company_id)
        .order_by(LeadScore.scored_at.desc())
    )
    scores = list(scores_result.scalars().all())
    return scores
