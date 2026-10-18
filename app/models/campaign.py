"""Autopilot Outreach: a campaign that finds prospects, writes to them and sends.

A :class:`Campaign` is built from discovery documents: the AI reads them into an
ideal-customer profile (``icp_json``), and ``app.services.campaigns`` then
finds matching companies daily, drafts a personalised email per prospect, and
sends from the owner's connected Gmail inside the campaign's send window.

Sending goes through ``email_dispatch.send_outreach_email`` like every other
outreach path — kill switch, do-not-contact list, spam gate, mailbox warm-up and
the org's daily cap all apply — and each send also writes an ``EmailSend`` row,
so the inbox, reply matching and the cap's seed see campaign mail. The
campaign's own warm-up ramp and ``daily_send_limit`` are an *additional* cap on
top of those, never a replacement.
"""

import datetime as dt
import uuid
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import OrgScopedMixin

CAMPAIGN_STATUSES = ("active", "paused", "stopped")

# queued → draft_created → approved → sending → sent → replied | bounced
# skipped / failed are terminal. ``sending`` is the claim taken (and committed)
# before the wire, so a crash mid-send can never make a prospect eligible again.
PROSPECT_STATUSES = (
    "queued",
    "draft_created",
    "approved",
    "sending",
    "sent",
    "replied",
    "bounced",
    "skipped",
    "failed",
    # Discovered, but no address could be found or guessed. Never drafted or sent.
    "no_email",
)
# Statuses that mean "this address has been (or is being) mailed by a
# campaign". The partial unique index below makes "never the same address
# twice across campaigns" a database fact rather than a check that can race.
MAILED_STATUSES = ("sending", "sent", "replied", "bounced")


class Campaign(Base, OrgScopedMixin):
    __tablename__ = "campaigns"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    # Why an automatic pause happened ("bounce_rate", "gmail_disconnected"); NULL
    # for an operator's own pause/stop.
    pause_reason: Mapped[Optional[str]] = mapped_column(String(50))
    document_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    document_summary: Mapped[Optional[str]] = mapped_column(Text)
    icp_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    # The owner's Gmail mailbox campaign mail is sent from.
    gmail_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("gmail_accounts.id", ondelete="SET NULL")
    )
    # The sender is one of these two; the router clears the other (migration 125).
    smtp_account_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("smtp_accounts.id", ondelete="SET NULL")
    )
    # Drafting from an email template (migration 126): "ai" | "template" | "ai_template".
    template_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_templates.id", ondelete="SET NULL")
    )
    template_mode: Mapped[str] = mapped_column(
        String(20), nullable=False, default="ai", server_default="ai"
    )
    daily_send_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    send_window_start: Mapped[str] = mapped_column(String(5), nullable=False, default="09:00")
    send_window_end: Mapped[str] = mapped_column(String(5), nullable=False, default="18:00")
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="America/Toronto")
    send_on_weekends: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_before_send: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Off by default: discovery documents are usually lead lists, and attaching
    # one to a cold email would hand every recipient the other prospects' data.
    # When on, only PDF/DOCX documents are ever attached.
    attach_documents: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    warmup_day: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # The local date ``warmup_day`` was last advanced for.
    warmup_on: Mapped[Optional[date]] = mapped_column(Date)
    last_discovery_on: Mapped[Optional[date]] = mapped_column(Date)
    # When the next send may be scheduled (random 120–300s gaps between sends).
    next_send_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    deleted_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class CampaignProspect(Base, OrgScopedMixin):
    __tablename__ = "campaign_prospects"
    __table_args__ = (
        UniqueConstraint("campaign_id", "contact_id", name="uq_campaign_prospect_contact"),
        Index(
            "uq_campaign_prospects_mailed_email",
            "org_id",
            "to_email",
            unique=True,
            postgresql_where=text("status IN ('sending','sent','replied','bounced')"),
            sqlite_where=text("status IN ('sending','sent','replied','bounced')"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("companies.id", ondelete="SET NULL")
    )
    # The spec's "person": people are ``contacts`` in this schema.
    contact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("contacts.id", ondelete="SET NULL")
    )
    # Lower-cased snapshot of the address mailed — what dedup keys on, and what
    # stays true if the contact row is later edited or erased.
    to_email: Mapped[Optional[str]] = mapped_column(String(320), index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    source: Mapped[Optional[str]] = mapped_column(String(30))  # "document" | gemini/groq/openrouter
    skip_reason: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(Text)
    email_subject: Mapped[Optional[str]] = mapped_column(String(500))
    email_body: Mapped[Optional[str]] = mapped_column(Text)
    # The template this draft was written from (migration 127); None = the AI
    # wrote it freely. Can differ from the campaign's when switched per draft.
    template_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_templates.id", ondelete="SET NULL")
    )
    email_send_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("email_sends.id", ondelete="SET NULL")
    )
    gmail_message_id: Mapped[Optional[str]] = mapped_column(String(255))
    gmail_thread_id: Mapped[Optional[str]] = mapped_column(String(255))
    approved_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    # Set when a delayed send task is dispatched for this prospect; a lease, so
    # the next tick neither schedules it twice nor strands it if the task is lost.
    scheduled_for: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    sent_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    replied_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    bounced_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class CampaignDailyLog(Base, OrgScopedMixin):
    __tablename__ = "campaign_daily_logs"
    __table_args__ = (UniqueConstraint("campaign_id", "date", name="uq_campaign_daily_log_day"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("campaigns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    prospects_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    emails_sent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    emails_bounced: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    # Discovery searches run this day and when the last one ran (migration 128):
    # a short day is topped up, a few hours apart, instead of waiting for tomorrow.
    discovery_runs: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    last_discovery_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
