"""Campaigns can draft from an email template.

Revision ID: 126_campaign_templates
Revises: 125_campaign_smtp_sender

``template_id`` points at an ``email_templates`` row (SET NULL: deleting the
template returns the campaign to AI drafting rather than deleting it).
``template_mode`` is how drafts use it:

* ``ai``          — the AI writes every draft (the behaviour before this).
* ``template``    — the template's merge fields are filled per prospect; no AI.
* ``ai_template`` — the AI personalises the template for each prospect.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "126_campaign_templates"
down_revision = "125_campaign_smtp_sender"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "campaigns",
        sa.Column(
            "template_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("email_templates.id", ondelete="SET NULL"), nullable=True,
        ),
    )
    op.add_column(
        "campaigns",
        sa.Column("template_mode", sa.String(20), nullable=False, server_default="ai"),
    )


def downgrade() -> None:
    op.drop_column("campaigns", "template_mode")
    op.drop_column("campaigns", "template_id")
