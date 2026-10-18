"""Contact CRUD endpoints linked to companies."""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import undefer
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import resolve_country
from app.database import get_db
from app.dependencies import get_current_user_optional
from app.models.company import Company
from app.models.contact import Contact
from app.models.user import User
from app.schemas.communication import CommunicationHistoryResponse
from app.services.communication_history import get_contact_communications
from app.services.contact_grouping import classify_group
from app.schemas.contact import (
    ContactFacets,
    ContactFacetCount,
    BulkSuppressRequest,
    BulkSuppressResponse,
    BulkTagRequest,
    BulkTagResponse,
    ContactCreate,
    ContactDetailRead,
    ContactEngagement,
    ContactListResponse,
    ContactRead,
    ContactUpdate,
    DuplicateCheckResponse,
    DuplicateMatch,
    TagCount,
    TagListResponse,
    TagRenameRequest,
    TagUpdate,
)
from app.services import contact_tags as tag_service
from app.services.sql_search import LIKE_ESCAPE, contains_pattern

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/contacts", tags=["contacts"])


@router.post("/", response_model=ContactRead, status_code=201)
async def create_contact(
    payload: ContactCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new contact linked to a company."""
    # Verify company exists
    result = await db.execute(
        select(Company).where(Company.id == payload.company_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Company not found")

    data = payload.model_dump()
    data["tags"] = tag_service.normalize_tags(data.get("tags"))
    # Auto-classify genuine vs test when the caller didn't set a group.
    if not data.get("contact_group"):
        data["contact_group"] = classify_group(
            email=str(data.get("email") or "") or None,
            source=data.get("source"),
            name=" ".join(p for p in (data.get("first_name"), data.get("last_name")) if p),
        )
    contact = Contact(**data)
    db.add(contact)
    await db.flush()
    await db.refresh(contact)
    logger.info("Created contact %s for company %s", contact.id, payload.company_id)
    return contact


_CONTACT_SORTS = {"created_at", "first_name", "last_name", "email", "title", "company_name"}


@router.get("/", response_model=ContactListResponse)
async def list_contacts(
    company_id: Optional[uuid.UUID] = Query(None),
    q: Optional[str] = Query(None, description="Search name, email, or title"),
    unsubscribed: Optional[bool] = Query(None, description="Filter by unsubscribe status"),
    has_email: Optional[bool] = Query(None, description="Only contacts with/without an email"),
    status: Optional[str] = Query(None, description="subscribed, unsubscribed, or no_email"),
    city: Optional[str] = Query(None, description="Filter by city"),
    source: Optional[str] = Query(None, description="Filter by source"),
    group: Optional[str] = Query(None, description="Filter by contact group: 'genuine' or 'test'"),
    company_name: Optional[str] = Query(None, description="Filter by company name"),
    tags: Optional[str] = Query(
        None,
        description="Comma-separated tags. Matches contacts carrying ALL of them "
                    "(see tag_match to widen to ANY).",
    ),
    tag_match: str = Query("all", pattern="^(all|any)$"),
    suppressed: Optional[bool] = Query(
        None, description="Filter by hard compliance suppression"
    ),
    email_status: Optional[str] = Query(
        None, description="Verification verdict: valid | invalid | unknown"
    ),
    has_bounced: Optional[bool] = Query(
        None, description="Only contacts that have (or haven't) ever bounced"
    ),
    verification: Optional[str] = Query(
        None, description="Hunter verdict: deliverable, risky, undeliverable, unknown, or unverified",
    ),
    industry: Optional[str] = Query(None, description="Company industry (partial match)"),
    min_employees: Optional[int] = Query(None, ge=0),
    max_employees: Optional[int] = Query(None, ge=0),
    campaign_id: Optional[uuid.UUID] = Query(None, description="Only prospects of this campaign"),
    sort_by: str = Query("created_at", description="created_at, first_name, last_name, email, title, company_name"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List contacts with filtering, search, sorting, and pagination."""
    from sqlalchemy import or_, and_

    query = select(Contact)
    count_query = select(func.count(Contact.id))

    conditions = []
    # Track whether we've already joined Company on both queries so we
    # don't join twice (the `q` search, `company_name` filter, and
    # `company_name` sort can all require the Company join).
    company_joined = False

    def _ensure_company_join():
        nonlocal query, count_query, company_joined
        if not company_joined:
            query = query.join(Company, Contact.company_id == Company.id, isouter=True)
            count_query = count_query.join(Company, Contact.company_id == Company.id, isouter=True)
            company_joined = True

    if company_id:
        conditions.append(Contact.company_id == company_id)
    if unsubscribed is not None:
        conditions.append(Contact.unsubscribed == unsubscribed)
    if has_email is not None:
        conditions.append(Contact.email.isnot(None) if has_email else Contact.email.is_(None))

    # Status filter (combines unsubscribed and email presence logic)
    if status is not None:
        if status == "subscribed":
            conditions.append(Contact.unsubscribed == False)
            conditions.append(Contact.email.isnot(None))
        elif status == "unsubscribed":
            conditions.append(Contact.unsubscribed == True)
        elif status == "no_email":
            conditions.append(Contact.email.is_(None))

    # City filter (case-insensitive partial match)
    if city:
        conditions.append(Contact.city.ilike(contains_pattern(city), escape=LIKE_ESCAPE))

    # Source filter (exact match)
    if source:
        conditions.append(Contact.source == source)

    # Contact group filter (genuine / test, exact match)
    if group:
        conditions.append(Contact.contact_group == group)

    if suppressed is not None:
        conditions.append(Contact.suppressed == suppressed)

    if email_status:
        conditions.append(Contact.email_status == email_status)

    if has_bounced is not None:
        conditions.append(
            Contact.bounce_count > 0 if has_bounced else
            or_(Contact.bounce_count == 0, Contact.bounce_count.is_(None))
        )

    # Tag filter. Expressed as per-tag JSON containment rather than one
    # ``tags @> '[...]'`` so ``any`` is expressible at all, and so the same code
    # path works on SQLite (tests) where JSONB containment doesn't exist —
    # ``.contains`` renders to the dialect's own form.
    if tags:
        wanted = tag_service.normalize_tags(tags.split(","))
        if wanted:
            dialect = db.bind.dialect.name if db.bind is not None else ""
            clauses = [tag_service.tag_clause(Contact.tags, t, dialect) for t in wanted]
            conditions.append(and_(*clauses) if tag_match == "all" else or_(*clauses))

    if verification == "unverified":
        conditions.append(Contact.email_verification_result.is_(None))
    elif verification:
        conditions.append(Contact.email_verification_result == verification)

    if industry:
        _ensure_company_join()
        conditions.append(Company.industry_segment.ilike(contains_pattern(industry), escape=LIKE_ESCAPE))
    if min_employees is not None:
        _ensure_company_join()
        conditions.append(Company.employee_count >= min_employees)
    if max_employees is not None:
        _ensure_company_join()
        conditions.append(Company.employee_count <= max_employees)

    if campaign_id is not None:
        from app.models.campaign import CampaignProspect

        conditions.append(Contact.id.in_(
            select(CampaignProspect.contact_id).where(CampaignProspect.campaign_id == campaign_id)
        ))

    # Company name filter (case-insensitive partial match, requires Company join)
    if company_name:
        _ensure_company_join()
        conditions.append(
            Company.name.ilike(contains_pattern(company_name), escape=LIKE_ESCAPE)
        )

    if q:
        search = contains_pattern(q)
        _ensure_company_join()
        conditions.append(or_(
            Contact.first_name.ilike(search, escape=LIKE_ESCAPE),
            Contact.last_name.ilike(search, escape=LIKE_ESCAPE),
            Contact.email.ilike(search, escape=LIKE_ESCAPE),
            Contact.title.ilike(search, escape=LIKE_ESCAPE),
            Company.name.ilike(search, escape=LIKE_ESCAPE),
        ))

    if conditions:
        combined = and_(*conditions)
        query = query.where(combined)
        count_query = count_query.where(combined)

    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Sorting — company_name requires joining Company
    if sort_by == "company_name":
        _ensure_company_join()
        sort_col = Company.name
    elif sort_by in _CONTACT_SORTS:
        sort_col = getattr(Contact, sort_by)
    else:
        sort_col = Contact.created_at
    query = query.order_by(sort_col.asc() if sort_dir == "asc" else sort_col.desc())

    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    result = await db.execute(query)
    contacts = list(result.scalars().all())

    return ContactListResponse(items=contacts, total=total, page=page, page_size=page_size)


_VERIFICATION_LABELS = {
    "deliverable": "Deliverable",
    "risky": "Risky (catch-all)",
    "undeliverable": "Undeliverable",
    "unknown": "Unknown",
    None: "Not verified",
}


@router.get("/facets", response_model=ContactFacets)
async def contact_facets(db: AsyncSession = Depends(get_db)):
    """Counts by category — Hunter verdict, source, industry, campaign.

    The contact table is already the central, org-wide database (every
    discovered person lands in it, deduplicated by email); this is what lets a
    page group it. Contacts without an email are counted too — they are
    people a search found and Hunter may still find an address for.
    """
    from app.models.campaign import Campaign, CampaignProspect

    total = (await db.execute(select(func.count(Contact.id)))).scalar_one()
    by_verdict = (await db.execute(
        select(Contact.email_verification_result, func.count(Contact.id))
        .group_by(Contact.email_verification_result)
    )).all()
    by_source = (await db.execute(
        select(Contact.source, func.count(Contact.id))
        .group_by(Contact.source).order_by(func.count(Contact.id).desc()).limit(20)
    )).all()
    by_industry = (await db.execute(
        select(Company.industry_segment, func.count(Contact.id))
        .join(Company, Contact.company_id == Company.id)
        .where(Company.industry_segment.isnot(None))
        .group_by(Company.industry_segment).order_by(func.count(Contact.id).desc()).limit(20)
    )).all()
    by_campaign = (await db.execute(
        select(Campaign.id, Campaign.name, func.count(func.distinct(CampaignProspect.contact_id)))
        .join(CampaignProspect, CampaignProspect.campaign_id == Campaign.id)
        .where(Campaign.deleted_at.is_(None), CampaignProspect.contact_id.isnot(None))
        .group_by(Campaign.id, Campaign.name).order_by(Campaign.name, Campaign.id)
    )).all()
    order = list(_VERIFICATION_LABELS)
    return ContactFacets(
        total=total,
        verification=[
            ContactFacetCount(value=v or "unverified", label=_VERIFICATION_LABELS.get(v, v or ""), count=n)
            for v, n in sorted(by_verdict, key=lambda r: order.index(r[0]) if r[0] in order else 99)
        ],
        source=[ContactFacetCount(value=v, label=v or "Unknown", count=n) for v, n in by_source],
        industry=[ContactFacetCount(value=v, label=v, count=n) for v, n in by_industry],
        campaign=[ContactFacetCount(value=str(i), label=name, count=n) for i, name, n in by_campaign],
    )


@router.get("/check-duplicate", response_model=DuplicateCheckResponse)
async def check_duplicate_contact(
    email: Optional[str] = Query(None),
    phone: Optional[str] = Query(None, description="Any phone number from the card"),
    first_name: Optional[str] = Query(None),
    last_name: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Find existing contacts that look like the one about to be saved.

    Used by the card scanner before a quick-add to warn about (and offer to open)
    a likely duplicate. Matches on exact email, on a shared phone number (compared
    by significant digits across every stored number), or on a full first+last
    name match. Phone/name comparison is done in Python so it works the same on
    Postgres and the SQLite test DB and tolerates formatting differences.
    """
    email_n = (email or "").strip().lower()
    first_n = (first_name or "").strip().lower()
    last_n = (last_name or "").strip().lower()
    # Significant digits of the probe phone (drop a leading US country code so
    # "+1 555…" matches "555…"); require 7+ to avoid matching on extensions.
    probe_digits = _digits(phone or "")
    if len(probe_digits) == 11 and probe_digits.startswith("1"):
        probe_digits = probe_digits[1:]

    if not email_n and not probe_digits and not (first_n and last_n):
        return DuplicateCheckResponse(matches=[])

    # Pull a bounded, recent candidate set with its company name. additional_phones
    # is a normal (non-deferred) column, so it loads here for the phone comparison.
    rows = await db.execute(
        select(Contact, Company.name)
        .join(Company, Contact.company_id == Company.id)
        .order_by(Contact.created_at.desc())
        .limit(2000)
    )

    def phone_digits_of(c: Contact) -> set[str]:
        out: set[str] = set()
        nums = [c.phone, c.mobile, c.fax]
        for p in (c.additional_phones or []):
            if isinstance(p, dict):
                nums.append(p.get("number"))
        for n in nums:
            d = _digits(n or "")
            if len(d) == 11 and d.startswith("1"):
                d = d[1:]
            if len(d) >= 7:
                out.add(d)
        return out

    matches: list[DuplicateMatch] = []
    for contact, company_name in rows.all():
        reasons: list[str] = []
        if email_n and (contact.email or "").strip().lower() == email_n:
            reasons.append("email")
        if probe_digits and len(probe_digits) >= 7:
            if any(d == probe_digits or d.endswith(probe_digits) or probe_digits.endswith(d)
                   for d in phone_digits_of(contact)):
                reasons.append("phone")
        if first_n and last_n and (contact.first_name or "").strip().lower() == first_n \
                and (contact.last_name or "").strip().lower() == last_n:
            reasons.append("name")
        if reasons:
            matches.append(DuplicateMatch(
                id=contact.id,
                first_name=contact.first_name,
                last_name=contact.last_name,
                title=contact.title,
                email=contact.email,
                company_id=contact.company_id,
                company_name=company_name,
                source=contact.source,
                has_card_image=contact.has_card_image,
                created_at=contact.created_at,
                match_reasons=reasons,
            ))

    return DuplicateCheckResponse(matches=matches)


# --- Tags -------------------------------------------------------------------
#
# Declared before ``/{contact_id}`` so "tags" isn't swallowed as a contact id.


@router.get("/tags", response_model=TagListResponse)
async def list_tags(db: AsyncSession = Depends(get_db)):
    """Every tag in use, with the number of contacts carrying it.

    Aggregated in Python rather than SQL because the counts come from unnesting a
    JSON array, whose syntax differs per dialect, and the cardinality here is
    "how many distinct labels has one team typed" — tens, not millions.

    Untagged contacts are excluded in SQL. They are the overwhelming majority and
    contribute nothing to the result, so shipping them to Python only to discard
    them is the difference between a payload the size of the tag vocabulary and
    one the size of the contact list.
    """
    from sqlalchemy import Text as _Text, cast as _cast

    rows = (
        await db.execute(
            select(Contact.tags).where(
                Contact.tags.isnot(None),
                _cast(Contact.tags, _Text) != "[]",
            )
        )
    ).scalars().all()
    counts: dict[str, int] = {}
    for tags in rows:
        for tag in set(tag_service.normalize_tags(tags)):
            counts[tag] = counts.get(tag, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return TagListResponse(tags=[TagCount(tag=t, count=c) for t, c in ordered])


@router.post("/tags/rename", response_model=BulkTagResponse)
async def rename_tag(payload: TagRenameRequest, db: AsyncSession = Depends(get_db)):
    """Rename a tag everywhere it appears.

    Free-form labels acquire typos and near-duplicates ("tradeshow" next to
    "trade show"), and without this the only fix is re-tagging by hand. Merging
    into an existing tag is allowed and de-duplicates rather than doubling up.
    """
    old = tag_service.normalize_tag(payload.old)
    new = tag_service.normalize_tag(payload.new)
    if not old or not new:
        raise HTTPException(status_code=400, detail="tag names cannot be empty")
    if old == new:
        return BulkTagResponse(updated=0)

    dialect = db.bind.dialect.name if db.bind is not None else ""
    contacts = (
        await db.execute(
            select(Contact).where(tag_service.tag_clause(Contact.tags, old, dialect))
        )
    ).scalars().all()

    updated = 0
    for contact in contacts:
        contact.tags = tag_service.apply_tag_delta(
            contact.tags, add=[new], remove=[old]
        )
        updated += 1
    await db.flush()
    logger.info("Renamed tag %r → %r on %d contact(s)", old, new, updated)
    return BulkTagResponse(updated=updated)


@router.post("/bulk-tag", response_model=BulkTagResponse)
async def bulk_tag(payload: BulkTagRequest, db: AsyncSession = Depends(get_db)):
    """Add and/or remove tags across many contacts in one call.

    Deltas, not a replacement set: a bulk action driven from a list view has no
    business knowing what other tags those contacts carry, and sending a whole
    set would wipe every tag it didn't happen to know about.
    """
    add = tag_service.normalize_tags(payload.add)
    remove = tag_service.normalize_tags(payload.remove)
    if not add and not remove:
        raise HTTPException(status_code=400, detail="nothing to add or remove")

    contacts = (
        await db.execute(select(Contact).where(Contact.id.in_(payload.contact_ids)))
    ).scalars().all()
    found = {c.id for c in contacts}

    for contact in contacts:
        contact.tags = tag_service.apply_tag_delta(contact.tags, add=add, remove=remove)
    await db.flush()

    return BulkTagResponse(
        updated=len(contacts),
        not_found=[cid for cid in payload.contact_ids if cid not in found],
    )


@router.post("/bulk-suppress", response_model=BulkSuppressResponse)
async def bulk_suppress(
    payload: BulkSuppressRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional),
):
    """Suppress (or un-suppress) many contacts at once.

    Suppression is the hard do-not-email gate every send path reads, so
    suppressing also **stops their live enrollments** — leaving those active
    would mean the queue keeps selecting a contact it will then refuse to mail,
    forever. Enrollments are marked ``unsubscribed``, not ``bounced``: a
    compliance stop must not dent sender deliverability metrics.

    Un-suppressing lifts the flag but deliberately does **not** restart
    sequences. Re-enrolling someone is a decision, and quietly resuming cold
    outreach because a checkbox was un-ticked is not a decision anyone made.
    """
    from datetime import datetime, timezone

    from app.models.consent import ConsentRecord
    from app.models.outreach import OutreachEnrollment
    from app.services import suppression as _suppression

    contacts = (
        await db.execute(select(Contact).where(Contact.id.in_(payload.contact_ids)))
    ).scalars().all()
    found = {c.id for c in contacts}
    now = datetime.now(timezone.utc)
    actor = user.email if user is not None else "system"
    stopped = 0

    for contact in contacts:
        contact.suppressed = payload.suppressed
        if not payload.suppressed:
            # Only clear the legacy flag when it was suppression that set it —
            # a genuine link-based unsubscribe must survive.
            #
            # Lift this route's own list entry too, or un-suppressing does
            # nothing observable: the flag comes off, the entry stays, the send
            # gate goes on refusing and the operator gets silence. Narrow by
            # design — ``remove_entry`` only touches ``manual`` entries, so a
            # bounce, a complaint, an unsubscribe or an erasure is not
            # reversible by un-ticking a checkbox on one contact row.
            await _suppression.remove_entry(db, contact.email)
            continue

        contact.unsubscribed = True
        db.add(ConsentRecord(
            contact_id=contact.id,
            consent_type="none",
            basis=(payload.reason or f"bulk suppression by {actor}")[:2000],
            granted_at=None,
            revoked_at=now,
        ))
        # The durable half. Everything above lives on a contact *row*, and the
        # scraper makes rows: suppress someone today, rediscover them next month
        # through another source, and the new row arrives with
        # ``suppressed=False``. The list is keyed by address and outlives the
        # row. This is the route named "compliance" and it was, until now, the
        # only one of the three opt-out paths that recorded nothing durable —
        # the reply path and the unsubscribe link both write here.
        await _suppression.add_entry(
            db, contact.email, reason="manual", source="manual"
        )
        enrollments = (
            await db.execute(
                select(OutreachEnrollment).where(
                    OutreachEnrollment.contact_id == contact.id,
                    OutreachEnrollment.status.in_(("active", "paused", "pending_review")),
                )
            )
        ).scalars().all()
        for enr in enrollments:
            enr.status = "unsubscribed"
            enr.completed_at = now
            stopped += 1

    await db.flush()
    logger.info(
        "Bulk %s %d contact(s) by %s — %d enrollment(s) stopped",
        "suppressed" if payload.suppressed else "un-suppressed",
        len(contacts), actor, stopped,
    )
    return BulkSuppressResponse(
        updated=len(contacts),
        enrollments_stopped=stopped,
        not_found=[cid for cid in payload.contact_ids if cid not in found],
    )


@router.get("/{contact_id}", response_model=ContactDetailRead)
async def get_contact(
    contact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a single contact by ID, including its scanned card image (if any)."""
    result = await db.execute(
        select(Contact).where(Contact.id == contact_id).options(undefer(Contact.card_image))
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


@router.post("/{contact_id}/tags", response_model=ContactRead)
async def update_contact_tags(
    contact_id: uuid.UUID,
    payload: TagUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Add and/or remove tags on one contact."""
    contact = (
        await db.execute(select(Contact).where(Contact.id == contact_id))
    ).scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    contact.tags = tag_service.apply_tag_delta(
        contact.tags, add=payload.add, remove=payload.remove
    )
    await db.flush()
    await db.refresh(contact)
    return contact


@router.get("/{contact_id}/engagement", response_model=ContactEngagement)
async def get_contact_engagement(
    contact_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """What we've sent this contact and what came back.

    Reads **both** send tables — sequence sends exist only in
    ``outreach_events`` and manual/template sends only in ``email_sends``, so
    reading one of them under-reports whichever half the operator happened to
    use (see CLAUDE.md).
    """
    contact = (
        await db.execute(select(Contact.id).where(Contact.id == contact_id))
    ).scalar_one_or_none()
    if contact is None:
        raise HTTPException(status_code=404, detail="Contact not found")

    from app.services.contact_engagement import contact_engagement

    return ContactEngagement(**await contact_engagement(db, contact_id, limit=limit))


@router.get("/{contact_id}/timeline")
async def get_contact_timeline(
    contact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get activity timeline for a contact."""
    result = await db.execute(select(Contact).where(Contact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    timeline = []

    # Contact creation event
    timeline.append({
        "type": "contact_created",
        "description": f"Contact created via {contact.source or 'unknown'}",
        "timestamp": contact.created_at.isoformat() if contact.created_at else None,
    })

    # Interactions linked to this contact
    from app.models.interaction import Interaction
    interactions_result = await db.execute(
        select(Interaction)
        .where(Interaction.contact_id == contact_id)
        .order_by(Interaction.created_at.desc())
        .limit(50)
    )
    for interaction in interactions_result.scalars().all():
        timeline.append({
            "type": f"interaction_{interaction.interaction_type}",
            "description": interaction.subject or interaction.notes or str(interaction),
            "timestamp": interaction.created_at.isoformat() if interaction.created_at else None,
        })

    # Outreach enrollments for this contact
    from app.models.outreach import OutreachEnrollment
    enrollments_result = await db.execute(
        select(OutreachEnrollment)
        .where(OutreachEnrollment.contact_id == contact_id)
        .order_by(OutreachEnrollment.enrolled_at.desc())
        .limit(20)
    )
    for enrollment in enrollments_result.scalars().all():
        timeline.append({
            "type": "outreach_enrolled",
            "description": f"Enrolled in sequence (status: {enrollment.status})",
            "timestamp": enrollment.enrolled_at.isoformat() if enrollment.enrolled_at else None,
        })

    # Sort by timestamp descending
    timeline.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

    return {"contact_id": str(contact_id), "timeline": timeline}


@router.get("/{contact_id}/communications", response_model=CommunicationHistoryResponse)
async def get_contact_communications_endpoint(
    contact_id: uuid.UUID,
    mine_only: bool = Query(
        False, description="When authenticated, return only your own communications"
    ),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """Full communication history for a contact — who contacted them, when, and
    via which email account.

    Aggregates outbound emails, inbound replies, calls, and logged interactions,
    each attributed to the acting user. Pass ``mine_only=true`` (with a Bearer
    token) to scope the history to the authenticated user's own touchpoints.
    """
    scope_user_id = current_user.id if (mine_only and current_user is not None) else None
    result = await get_contact_communications(db, contact_id, user_id=scope_user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Contact not found")
    return result


@router.get("/{contact_id}/card-image")
async def get_contact_card_image(
    contact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return the raw scanned business-card image for a contact.

    Serves the decoded bytes with the correct content type so the image can be
    fetched as a blob (the API-key auth header rules out a bare <img src>)."""
    import base64
    from fastapi import Response

    result = await db.execute(
        select(Contact).where(Contact.id == contact_id).options(undefer(Contact.card_image))
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    if not contact.card_image:
        raise HTTPException(status_code=404, detail="No card image for this contact")

    data_url = contact.card_image
    media_type = "image/jpeg"
    b64 = data_url
    if data_url.startswith("data:"):
        header, _, b64 = data_url.partition(",")
        # header looks like "data:image/png;base64"
        if ":" in header and ";" in header:
            media_type = header.split(":", 1)[1].split(";", 1)[0] or media_type
    try:
        raw = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=500, detail="Stored card image is corrupt")
    return Response(content=raw, media_type=media_type)


@router.patch("/{contact_id}", response_model=ContactRead)
async def update_contact(
    contact_id: uuid.UUID,
    payload: ContactUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update contact fields."""
    result = await db.execute(select(Contact).where(Contact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "tags" in update_data:
        # Normalise here too, not only on the tag endpoints — otherwise a PATCH
        # is a back door that writes un-canonicalised labels the filter menu
        # then shows as separate tags.
        update_data["tags"] = tag_service.normalize_tags(update_data["tags"])
    for field, value in update_data.items():
        setattr(contact, field, value)

    await db.flush()
    await db.refresh(contact)
    logger.info("Updated contact %s", contact_id)
    return contact


@router.delete("/{contact_id}", status_code=204)
async def delete_contact(
    contact_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a contact."""
    result = await db.execute(select(Contact).where(Contact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    await db.delete(contact)
    await db.flush()
    logger.info("Deleted contact %s", contact_id)


@router.get("/company/{company_id}", response_model=ContactListResponse)
async def get_company_contacts(
    company_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get all contacts for a specific company."""
    result = await db.execute(
        select(Company).where(Company.id == company_id)
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Company not found")

    contacts_result = await db.execute(
        select(Contact)
        .where(Contact.company_id == company_id)
        .order_by(Contact.is_primary.desc(), Contact.created_at.desc())
    )
    contacts = list(contacts_result.scalars().all())

    return ContactListResponse(items=contacts, total=len(contacts))


# --- Quick Add (compose shortcut + business-card scanner) ---

from app.schemas.contact import QuickAddRequest, QuickAddResponse

def _normalize_country(raw: Optional[str]) -> str:
    """Map an OCR'd country onto a code the ``companies`` constraint accepts.

    This used to carry its own US/CA-only table, on a comment saying companies
    are "constrained to these two countries". They are constrained to nine.
    So a card reading "United Kingdom" or "Germany" did not fail the scan — it
    filed the company as ``US``, which is the mislabelling ``ingestion`` refuses
    to do on the grounds that a rejected row is recoverable and a relabelled one
    is not. The consent gate then read that ``US`` and applied CAN-SPAM to a
    German company.

    ``US`` remains the fallback for a blank or unreadable country, which is what
    a scanner mostly produces and what the endpoint has always defaulted to.
    """
    return resolve_country(raw) or "US"


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def _collect_phones(payload: "QuickAddRequest") -> list[dict]:
    """Merge phone / mobile / fax / additional_phones into one de-duplicated,
    labelled list so a card with several numbers keeps all of them."""
    entries: list[dict] = []
    seen: set[str] = set()

    def add(label: Optional[str], number: Optional[str]):
        num = (number or "").strip()
        if not num:
            return
        key = _digits(num) or num.lower()
        if key in seen:
            return
        seen.add(key)
        entries.append({"label": label, "number": num})

    add("Office", payload.phone)
    add("Mobile", payload.mobile)
    add("Fax", payload.fax)
    for p in (payload.additional_phones or []):
        # p is a PhoneEntry pydantic model
        add(p.label, p.number)
    return entries


def _valid_email(raw: Optional[str]) -> Optional[str]:
    """Return a syntactically valid email or None, so bad OCR never blocks a save."""
    from pydantic import TypeAdapter, EmailStr

    value = (raw or "").strip()
    if not value:
        return None
    try:
        return TypeAdapter(EmailStr).validate_python(value)
    except Exception:
        return None


@router.post("/quick-add", response_model=QuickAddResponse, status_code=201)
async def quick_add_contact(
    payload: QuickAddRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a contact on the fly (compose view or business-card scanner).

    Finds or creates the Company by name, then attaches the contact. Person
    fields (name, title, email, phone, LinkedIn) live on the Contact; company
    and location fields are stored on the Company. The caller's ``source`` is
    preserved unchanged so the frontend "Scanned" filter keeps working.
    """
    source = (payload.source or "manual").strip() or "manual"
    company_name = (payload.company_name or "").strip()
    company_created = False

    company = None
    if company_name:
        # First match, not "the" match. ``companies`` is unique on ``domain``
        # and on nothing else, so two firms sharing a name is ordinary B2B data
        # — and the scrapers create companies from several sources. A
        # ``scalar_one_or_none`` here asserted otherwise and returned 500 for
        # every contact entered against a duplicated name, which is the
        # business-card scanner's entire job. Oldest first so repeat adds keep
        # filing into the same company instead of scattering between the
        # duplicates.
        result = await db.execute(
            select(Company)
            .where(func.lower(Company.name) == company_name.lower())
            .order_by(Company.created_at.asc(), Company.id.asc())
            .limit(1)
        )
        company = result.scalars().first()

    if company is None:
        # Company.name is required; fall back to the person's name or email.
        fallback_name = (
            company_name
            or " ".join(p for p in (payload.first_name, payload.last_name) if p).strip()
            or (payload.email or "").strip()
            or "Unknown Company"
        )
        company = Company(
            name=fallback_name,
            city=(payload.city or None),
            state_province=(payload.state or None),
            country=_normalize_country(payload.country),
            website=(payload.website or None),
            phone=(payload.phone or None),
            notes=(payload.address or None),
            source=source,
        )
        db.add(company)
        await db.flush()
        company_created = True
        logger.info("Quick-add created company '%s' (%s)", company.name, company.id)

    # Preserve every phone number the scan found, labelled, with no duplicates.
    extra_phones = _collect_phones(payload)

    valid_email = _valid_email(payload.email)
    # Classify genuine vs test from the email/source/name unless told explicitly.
    contact_group = payload.contact_group or classify_group(
        email=valid_email,
        source=source,
        name=" ".join(p for p in (payload.first_name, payload.last_name) if p),
    )

    contact = Contact(
        company_id=company.id,
        first_name=(payload.first_name or None),
        last_name=(payload.last_name or None),
        title=(payload.title or None),
        department=(payload.department or None),
        email=valid_email,
        phone=(payload.phone or None),
        mobile=(payload.mobile or None),
        fax=(payload.fax or None),
        additional_phones=(extra_phones or None),
        linkedin_url=(payload.linkedin_url or None),
        website=(payload.website or None),
        # Location mirrored onto the contact so the detail view shows the card
        # exactly as scanned (the company keeps its own normalized copy).
        address=(payload.address or None),
        city=(payload.city or None),
        state=(payload.state or None),
        postal_code=(payload.postal_code or None),
        country=(payload.country or None),
        notes=(payload.notes or None),
        card_image=(payload.card_image or None),
        source=source,
        contact_group=contact_group,
    )
    db.add(contact)
    await db.flush()
    await db.refresh(contact)
    if payload.card_image:
        # refresh() re-expires the deferred card_image column; mark the value we
        # just persisted as already-loaded (no extra UPDATE) so has_card_image
        # is accurate in the response without shipping the blob itself.
        set_committed_value(contact, "card_image", payload.card_image)
    logger.info(
        "Quick-add created contact %s <%s> for company '%s' (source=%s)",
        contact.id, contact.email, company.name, source,
    )
    return QuickAddResponse(
        contact=contact,
        company_id=company.id,
        company_name=company.name,
        company_created=company_created,
    )
