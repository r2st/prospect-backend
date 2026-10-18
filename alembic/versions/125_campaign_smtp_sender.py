"""Campaigns can send from an SMTP account, not only Gmail.

Revision ID: 125_campaign_smtp_sender
Revises: 124_autopilot_campaigns

``campaigns.smtp_account_id`` sits beside ``gmail_account_id``; a campaign
holds exactly one of them (the router clears the other when either is set).
``send_outreach_email`` already resolves a sender id across both tables, so
this column is all the campaign side lacked. SET NULL like the Gmail column: a
deleted mailbox pauses the campaign (``gmail_disconnected``) rather than
deleting it.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "125_campaign_smtp_sender"
down_revision = "124_autopilot_campaigns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "campaigns",
        sa.Column(
            "smtp_account_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("smtp_accounts.id", ondelete="SET NULL"), nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("campaigns", "smtp_account_id")
