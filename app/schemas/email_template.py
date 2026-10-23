"""Pydantic schemas for email templates, preview, and compose/send."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --- Templates ---

class EmailTemplateBase(BaseModel):
    name: str = Field(..., max_length=255)
    subject: str = Field(..., max_length=500)
    body: str = Field(..., description="HTML body, supports {{merge_field}} placeholders")


class EmailTemplateCreate(EmailTemplateBase):
    is_default: bool = False


class EmailTemplateUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    subject: Optional[str] = Field(None, max_length=500)
    body: Optional[str] = None
    is_default: Optional[bool] = None


class EmailTemplateAttachmentRead(BaseModel):
    """An attachment's metadata. The bytes are served by the download route."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    created_at: datetime


class EmailTemplateRead(EmailTemplateBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    is_default: bool
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    attachments: list[EmailTemplateAttachmentRead] = []


# --- Validation ---

class TemplateValidateRequest(BaseModel):
    """Check a template without saving it.

    The point of a separate endpoint is that an author wants the warnings while
    they are still editing. Create/update reject only on *errors*, so warnings
    are otherwise invisible until someone reads the mail that went out.
    """

    subject: Optional[str] = None
    body: Optional[str] = None
    template_id: Optional[uuid.UUID] = None


class TemplateValidateResponse(BaseModel):
    """Errors block a save; warnings do not.

    The split matters: an unknown merge field is not a style preference, it is a
    field that will render as empty text in real mail — ``{{firstname}}`` for
    ``first_name`` produces ``Hi ,`` and no error anywhere. A missing unsubscribe
    line, by contrast, has legitimate exceptions.
    """

    valid: bool
    errors: list[str] = []
    warnings: list[str] = []
    fields_used: list[str] = []
    valid_fields: list[str] = []


# --- Per-template performance ---

class TemplatePerformance(BaseModel):
    """What one template actually achieved, from its own send log.

    Scope is ``email_sends`` — the manual and template send paths. Sequence-step
    copy lives in ``sequence_steps`` and reports through
    ``/dialtone/sequences/{id}/analytics``; the two are separate send paths with
    separate tables and merging them here would produce a number belonging to
    neither. See ``failed``: a template whose sends are erroring is a very
    different problem from one whose sends are ignored, and a bare open rate
    cannot tell them apart because failed sends never open.
    """

    template_id: uuid.UUID
    name: str
    is_default: bool
    is_archived: bool

    sent: int
    delivered: int
    failed: int
    opened: int
    clicked: int
    replied: int

    open_rate: float
    click_rate: float
    reply_rate: float
    failure_rate: float

    first_send_at: Optional[datetime] = None
    last_send_at: Optional[datetime] = None


# --- Preview ---

class TemplatePreviewRequest(BaseModel):
    """Render a template with a contact's data merged in.

    Provide ``template_id`` to preview a stored template, or pass inline
    ``subject``/``body`` to preview unsaved edits. Provide ``contact_id`` to
    merge a real contact's data; omit it to use sample placeholder data.
    """

    template_id: Optional[uuid.UUID] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    contact_id: Optional[uuid.UUID] = None


class TemplatePreviewResponse(BaseModel):
    subject: str
    body: str
    merge_context: dict


# --- Compose / Send ---

class EmailComposeRequest(BaseModel):
    contact_ids: list[uuid.UUID] = Field(..., min_length=1, max_length=50)
    template_id: Optional[uuid.UUID] = None
    # Connected Gmail account to send as. Defaults to the active account, then
    # falls back to SES when no Gmail account is connected.
    sender_account_id: Optional[uuid.UUID] = None
    subject_b: Optional[str] = Field(None, max_length=500, description="Variant B subject for A/B testing")
    scheduled_at: Optional[datetime] = Field(None, description="Schedule send for later")
    track_opens: bool = Field(True, description="Inject open-tracking pixel into email body")
    signature_id: Optional[uuid.UUID] = Field(None, description="Email signature to use; falls back to org/default")


class EmailComposeResult(BaseModel):
    contact_id: uuid.UUID
    to_email: Optional[str] = None
    status: str
    detail: Optional[str] = None


class EmailComposeResponse(BaseModel):
    template_id: uuid.UUID
    sent: int
    # Queued for a later ``scheduled_at`` and not yet on the wire. Counted apart
    # from ``sent`` because it isn't one: the batch is handed to
    # ``app.tasks.scheduled_sends``, and folding it into ``sent`` told the
    # operator their mail had gone out when nothing had left the building.
    scheduled: int = 0
    skipped: int
    errors: int
    # How the batch was sent: "gmail" or "ses", and the visible From address.
    transport: str
    sender: str
    results: list[EmailComposeResult]


# --- Send history (tracking) ---

class EmailSendRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    template_id: Optional[uuid.UUID] = None
    contact_id: Optional[uuid.UUID] = None
    to_email: Optional[str] = None
    subject: Optional[str] = None
    status: str
    sent_at: datetime
    opened_at: Optional[datetime] = None
    clicked_at: Optional[datetime] = None
    ab_variant: Optional[str] = None
    scheduled_at: Optional[datetime] = None

# --- Paginated Send History ---

class EmailSendPaginatedResponse(BaseModel):
    """Paginated email send history response."""
    items: list[EmailSendRead]
    total: int
    page: int
    per_page: int
    pages: int


# --- A/B Testing ---

class ABStats(BaseModel):
    variant: str
    total: int
    opened: int
    clicked: int
    open_rate: float
    click_rate: float


class ABStatsResponse(BaseModel):
    template_id: uuid.UUID
    stats: list[ABStats]


# --- Campaign Analytics ---

class CampaignAnalytics(BaseModel):
    total_sent_7d: int = 0
    total_sent_30d: int = 0
    total_sent_all: int = 0
    open_rate: float = 0.0
    click_rate: float = 0.0
    reply_rate: float = 0.0
    bounce_rate: float = 0.0
    daily_sends: list[dict] = Field(default_factory=list)


# --- Send Test ---

class SendTestRequest(BaseModel):
    template_id: uuid.UUID
    test_email: EmailStr
    sender_account_id: Optional[uuid.UUID] = None


class SendTestResponse(BaseModel):
    status: str  # "sent" or "failed"
    detail: Optional[str] = None
    subject: Optional[str] = None


# --- AI Email Generation ---

class AIEmailGenerateRequest(BaseModel):
    """Request body for AI-powered email generation."""
    first_name: str = Field(..., max_length=100)
    company_name: str = Field("", max_length=255)
    industry: str = Field("", max_length=255)
    title: str = Field("", max_length=255)
    city: str = Field("", max_length=255)
    template_type: str = Field("first_touch", max_length=50)
    custom_instructions: Optional[str] = Field(None, max_length=500)


class AIEmailGenerateResponse(BaseModel):
    subject: str
    body: str
    demo_link: str
    ai_generated: bool
    model: Optional[str] = None
    template_type: str
