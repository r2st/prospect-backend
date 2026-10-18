"""Each campaign draft remembers the email template it was written from.

Revision ID: 127_prospect_template
Revises: 126_campaign_templates

``campaign_prospects.template_id`` is set when a draft is written from a
template, switched to another one from the draft preview, or saved as a new
template. The preview uses it to show the template — merge fields and all —
next to the filled-in draft. SET NULL: deleting the template keeps the draft.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "127_prospect_template"
down_revision = "126_campaign_templates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "campaign_prospects",
        sa.Column(
            "template_id", postgresql.UUID(as_uuid=True),
            sa.ForeignKey("email_templates.id", ondelete="SET NULL"), nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("campaign_prospects", "template_id")
