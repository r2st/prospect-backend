"""User model — authentication and role-based access control.

Supports a four-level hierarchy (SaaS multi-tenancy):
* ``platform_admin`` — manages ALL organizations; ``org_id`` is NULL, not bound
  to any org, and is exempt from org-level data isolation.
* ``org_admin``   — full access within their org, can manage org users/settings.
* ``org_manager`` — can view all org data, manage team members' work.
* ``org_agent``   — scoped to own data (emails, tasks, calls, etc.).

The legacy roles map ``admin→org_admin``, ``manager→org_manager``,
``agent→org_agent`` (migration 078).
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class UserRole(str, enum.Enum):
    """Roles a user can hold. Value is the string stored in the ``role`` column.

    Ordered low→high privilege; :meth:`hierarchy` / :meth:`rank` back the
    role-hierarchy checks in ``OrgScope.check_permission``.
    """

    ORG_AGENT = "org_agent"
    ORG_MANAGER = "org_manager"
    ORG_ADMIN = "org_admin"
    PLATFORM_ADMIN = "platform_admin"

    # Legacy aliases (pre-multi-tenancy). Same values as the canonical members
    # above, so Python makes them *aliases*: ``UserRole.ADMIN is UserRole.ORG_ADMIN``
    # and ``UserRole.ADMIN.value == "org_admin"``. Keeps older call sites working.
    ADMIN = "org_admin"
    MANAGER = "org_manager"
    AGENT = "org_agent"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        # Iterating an Enum skips aliases → only the 4 canonical values.
        return tuple(r.value for r in cls)

    @classmethod
    def hierarchy(cls) -> tuple[str, ...]:
        """Role values ordered from least to most privileged."""
        return (
            cls.ORG_AGENT.value,
            cls.ORG_MANAGER.value,
            cls.ORG_ADMIN.value,
            cls.PLATFORM_ADMIN.value,
        )

    @classmethod
    def rank(cls, role: str) -> int:
        """Index of *role* in the hierarchy; -1 if unknown."""
        try:
            return cls.hierarchy().index(role)
        except ValueError:
            return -1


# Roles that manage their whole org (or the platform) — used for authz checks.
ORG_PRIVILEGED_ROLES = (UserRole.ORG_ADMIN.value, UserRole.ORG_MANAGER.value, UserRole.PLATFORM_ADMIN.value)

# Pre-multi-tenancy role strings → canonical values. Any lingering legacy role
# (in a stale JWT or a test fixture) is normalized before authz checks.
LEGACY_ROLE_MAP = {
    "admin": UserRole.ORG_ADMIN.value,
    "manager": UserRole.ORG_MANAGER.value,
    "agent": UserRole.ORG_AGENT.value,
}


def normalize_role(role: str) -> str:
    """Map a legacy role string to its canonical value; pass through otherwise."""
    return LEGACY_ROLE_MAP.get(role, role)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "role IN ('org_agent','org_manager','org_admin','platform_admin')",
            name="ck_users_role",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default=UserRole.ORG_AGENT.value)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Set when an admin deletes the user. Deletion is soft: the row (and the
    # attribution hanging off it) stays, is_active goes False so every auth
    # gate refuses it, and the admin console stops listing it.
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    # Multi-tenant org membership. NULL is reserved for ``platform_admin`` users,
    # who are not bound to any single org and bypass org-level isolation. Every
    # other role has a non-null org_id (enforced at the application layer).
    org_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Which user (admin) created this account, if any. Self-referential.
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Self-referential relationship to the creating admin.
    creator: Mapped["User | None"] = relationship("User", remote_side=[id], foreign_keys=[created_by])
    # The organization this user belongs to.
    organization = relationship("Organization", back_populates="users", foreign_keys=[org_id])

    def __repr__(self) -> str:
        return f"<User {self.email} role={self.role} active={self.is_active}>"
