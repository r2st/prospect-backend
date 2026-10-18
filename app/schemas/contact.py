"""Pydantic schemas for Contact endpoints."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class PhoneEntry(BaseModel):
    """A single labelled phone number (e.g. {"label": "Office", "number": "..."})."""
    label: Optional[str] = None
    number: str


class ContactBase(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    title: Optional[str] = Field(None, max_length=150)
    department: Optional[str] = Field(None, max_length=150)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=30)
    mobile: Optional[str] = Field(None, max_length=30)
    fax: Optional[str] = Field(None, max_length=30)
    additional_phones: Optional[list[PhoneEntry]] = None
    linkedin_url: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    postal_code: Optional[str] = Field(None, max_length=20)
    country: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None
    is_primary: bool = False
    # 'genuine' or 'test' — see app.services.contact_grouping. Omit to let the
    # server auto-classify from the email/source/name.
    contact_group: Optional[str] = Field(None, max_length=20)
    # Free-form operator labels. Normalised server-side (trimmed, lowercased,
    # de-duplicated) so "IMTS 2026", "imts 2026 " and "imts 2026" are one tag and
    # not three near-identical entries in the filter list.
    tags: list[str] = Field(default_factory=list)


class ContactCreate(ContactBase):
    company_id: uuid.UUID
    source: Optional[str] = Field(None, max_length=50)


class ContactUpdate(BaseModel):
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    title: Optional[str] = Field(None, max_length=150)
    department: Optional[str] = Field(None, max_length=150)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=30)
    mobile: Optional[str] = Field(None, max_length=30)
    fax: Optional[str] = Field(None, max_length=30)
    additional_phones: Optional[list[PhoneEntry]] = None
    linkedin_url: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    postal_code: Optional[str] = Field(None, max_length=20)
    country: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None
    is_primary: Optional[bool] = None
    source: Optional[str] = Field(None, max_length=50)
    contact_group: Optional[str] = Field(None, max_length=20)
    # Replaces the whole tag set when given. Use the dedicated tag endpoints for
    # add/remove semantics — a PATCH that half-knows the current tags would
    # clobber a tag added by someone else between the read and the write.
    tags: Optional[list[str]] = None


class ContactRead(ContactBase):
    """Contact as returned in lists. Excludes the heavy ``card_image`` payload;
    use ``has_card_image`` to know whether to fetch the detail endpoint."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    company_id: uuid.UUID
    source: Optional[str] = None
    has_card_image: bool = False
    # Compliance state, surfaced in lists so the management view can show at a
    # glance who is off-limits without a second call per row.
    suppressed: bool = False
    unsubscribed: bool = False
    status: Optional[str] = None
    email_status: Optional[str] = None
    # Hunter's verdict (deliverable / risky / undeliverable / unknown); None =
    # never verified. Only "deliverable" is ever sent by a campaign.
    email_verification_result: Optional[str] = None
    bounce_count: int = 0
    created_at: datetime
    updated_at: datetime


class ContactFacetCount(BaseModel):
    value: Optional[str] = None
    label: str
    count: int


class ContactFacets(BaseModel):
    """The contact database grouped by category, for the browse page's filters."""
    total: int
    verification: list[ContactFacetCount]
    source: list[ContactFacetCount]
    industry: list[ContactFacetCount]
    campaign: list[ContactFacetCount]


class ContactDetailRead(ContactRead):
    """Single-contact view — adds the full base64 card image data URL."""
    card_image: Optional[str] = None


class ContactListResponse(BaseModel):
    items: list[ContactRead]
    total: int
    # Pagination metadata (mirrors CompanyListResponse). Optional so callers that
    # return a full, unpaginated set (all contacts for one company) can omit them.
    page: Optional[int] = None
    page_size: Optional[int] = None


# --- Quick Add (compose shortcut + business-card scanner) ---

class QuickAddRequest(BaseModel):
    """Add a contact on the fly — creates the company if needed.

    Used both by the compose view (name + email + company) and by the business
    card scanner, which also supplies phone, links and the company's location /
    website. Everything is optional so a partial OCR read still saves, and
    `email` is a plain string (validated, then dropped if malformed) so an
    imperfect scan never 422s the whole save.
    """
    # Person — stored on the Contact.
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    title: Optional[str] = Field(None, max_length=150)
    department: Optional[str] = Field(None, max_length=150)
    email: Optional[str] = None
    phone: Optional[str] = Field(None, max_length=30)
    mobile: Optional[str] = Field(None, max_length=30)
    fax: Optional[str] = Field(None, max_length=30)
    # Every number found on the card, labelled — nothing is dropped.
    additional_phones: Optional[list[PhoneEntry]] = None
    linkedin_url: Optional[str] = None
    # Company / location — stored on the find-or-created Company AND, for the
    # detail view, mirrored onto the Contact.
    company_name: Optional[str] = Field(None, max_length=255)
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    postal_code: Optional[str] = Field(None, max_length=20)
    country: Optional[str] = None
    notes: Optional[str] = None
    # The scanned card image as a base64 data URL ("data:image/...;base64,...").
    card_image: Optional[str] = None
    # Where the contact came from ('manual', 'card_scan', …). Drives the
    # frontend "Scanned" filter, so it must round-trip unchanged.
    source: Optional[str] = Field(None, max_length=50)
    # 'genuine' or 'test'. Omit to let the server auto-classify.
    contact_group: Optional[str] = Field(None, max_length=20)


class QuickAddResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    contact: ContactRead
    company_id: uuid.UUID
    company_name: str
    company_created: bool


# --- Duplicate detection (run before saving a scanned card) ---

class DuplicateMatch(BaseModel):
    """An existing contact that looks like the one about to be saved."""
    id: uuid.UUID
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    email: Optional[str] = None
    company_id: uuid.UUID
    company_name: Optional[str] = None
    source: Optional[str] = None
    has_card_image: bool = False
    created_at: datetime
    # Why this was flagged: any of "email", "phone", "name".
    match_reasons: list[str] = []


class DuplicateCheckResponse(BaseModel):
    matches: list[DuplicateMatch] = []


# --- Contact management (tags, bulk suppression, engagement) ---

class TagUpdate(BaseModel):
    """Add and/or remove tags on one contact.

    Deliberately *not* a whole-set replacement. Two people tagging from the same
    stale list view would each write the set they last read, and whoever saved
    second would silently drop the other's tag. Add/remove deltas commute.
    """
    add: list[str] = Field(default_factory=list, max_length=50)
    remove: list[str] = Field(default_factory=list, max_length=50)


class BulkTagRequest(BaseModel):
    contact_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=1000)
    add: list[str] = Field(default_factory=list, max_length=50)
    remove: list[str] = Field(default_factory=list, max_length=50)


class TagCount(BaseModel):
    tag: str
    count: int


class TagListResponse(BaseModel):
    """Every tag in use, with how many contacts carry it.

    The counts are what make the list usable as a filter menu — a tag on two
    contacts and a tag on two thousand should not look the same.
    """
    tags: list[TagCount] = []


class TagRenameRequest(BaseModel):
    old: str = Field(..., min_length=1, max_length=60)
    new: str = Field(..., min_length=1, max_length=60)


class BulkSuppressRequest(BaseModel):
    contact_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=1000)
    # False lifts the suppression. Un-suppressing is deliberately possible: the
    # usual reason is that someone suppressed the wrong row, and forcing a
    # database edit to undo a UI mistake is worse than allowing the reversal.
    suppressed: bool = True
    reason: Optional[str] = Field(None, max_length=500)


class BulkSuppressResponse(BaseModel):
    updated: int
    enrollments_stopped: int
    # Ids asked for that don't exist (or aren't visible to this org). Reported
    # rather than silently ignored, so a bad paste doesn't look like a success.
    not_found: list[uuid.UUID] = []


class BulkTagResponse(BaseModel):
    updated: int
    not_found: list[uuid.UUID] = []


class EngagementEvent(BaseModel):
    """One thing that happened between us and this contact."""
    type: str          # sent | opened | clicked | replied | bounced
    occurred_at: datetime
    subject: Optional[str] = None
    step_number: Optional[int] = None
    sequence_name: Optional[str] = None
    source: str = "outreach"   # outreach | email_send | reply


class ContactEngagement(BaseModel):
    """Per-contact engagement roll-up + recent history.

    Reads BOTH send tables. Sequence sends live only in ``outreach_events`` and
    manual/template sends only in ``email_sends``; counting one of them is how a
    surface ends up telling an operator we never emailed someone we have emailed
    nine times.
    """
    contact_id: uuid.UUID
    sent: int = 0
    opened: int = 0
    clicked: int = 0
    replied: int = 0
    bounced: int = 0
    open_rate: float = 0.0
    reply_rate: float = 0.0
    first_touch_at: Optional[datetime] = None
    last_touch_at: Optional[datetime] = None
    active_enrollments: int = 0
    events: list[EngagementEvent] = []
