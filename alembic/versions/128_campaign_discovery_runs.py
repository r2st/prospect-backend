"""Count discovery searches per campaign day, so a short day can be topped up.

Revision ID: 128_campaign_discovery_runs
Revises: 127_prospect_template

Discovery ran once per local day. A day whose search came back mostly
undeliverable or email-less left the campaign with nothing to send until
tomorrow. ``discovery_runs`` and ``last_discovery_at`` let the tick search
again a few hours later (capped), and cap the operator's "Find more prospects".
"""

import sqlalchemy as sa
from alembic import op

revision = "128_campaign_discovery_runs"
down_revision = "127_prospect_template"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "campaign_daily_logs",
        sa.Column("discovery_runs", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "campaign_daily_logs",
        sa.Column("last_discovery_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("campaign_daily_logs", "last_discovery_at")
    op.drop_column("campaign_daily_logs", "discovery_runs")
