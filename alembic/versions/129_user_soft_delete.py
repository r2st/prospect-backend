"""Soft-delete users from the admin console.

Revision ID: 129_user_soft_delete
Revises: 128_campaign_discovery_runs

DELETE /admin/users/{id} only flipped ``is_active``, which is what the
Deactivate button already did — the "deleted" user stayed in the list.
``deleted_at`` marks the account as removed: it is hidden from the console and
can't be edited or reactivated there, while the row survives so the history
attributed to it (emails sent, tasks, deals) keeps its owner.
"""

import sqlalchemy as sa
from alembic import op

revision = "129_user_soft_delete"
down_revision = "128_campaign_discovery_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("deleted_at", sa.TIMESTAMP(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "deleted_at")
