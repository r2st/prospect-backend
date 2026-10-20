"""API key model — org-scoped credentials for the public REST API.

Each row is a long-lived credential a customer uses to call the public API
(``/api/v1/public/*``) server-to-server, distinct from the JWT a human uses in
the app. Keys are **org-scoped** (they mix in :class:`OrgScopedMixin`, so the
key both belongs to an org and — once the auth layer binds that org as the
request context — every query the key makes is isolated to it).

Security invariants:

* **Never stored in plaintext.** Only the SHA-256 hash of the raw key
  (``key_hash``, a 64-char hex digest) is persisted; the raw key is shown to the
  caller exactly once, at creation. A leaked database therefore cannot be used to
  call the API.
* **``key_prefix``** stores the first few characters of the raw key (e.g.
  ``llk_ab12cd``) so the management UI can show a recognizable, non-secret label
  for each key.
* **Scopes** gate what a key may do — a JSON array drawn from
  :data:`ALL_SCOPES` (``contacts:read``, ``contacts:write``, …). The auth layer
  enforces the required scope per endpoint.
* **Lifecycle** — ``expires_at`` (optional hard expiry) and ``revoked_at``
  (soft, immediate kill switch) both make ``is_usable`` return False.
"""

import hashlib
import uuid
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.mixins import OrgScopedMixin

# The complete set of scopes a key may hold. A resource split into read/write so
# a reporting integration can be granted read-only access without write. Kept as
# a plain tuple (not a DB enum) so scopes can be added without a migration.
ALL_SCOPES = (
    "contacts:read",
    "contacts:write",
    "companies:read",
    "companies:write",
    "deals:read",
    "deals:write",
    "sequences:read",
    "sequences:write",
    # The MCP server (app.mcp_server). Separate scopes so a key minted for the
    # REST API does not silently gain campaign or mailbox access.
    "campaigns:read",
    "campaigns:write",
    "mailboxes:read",
    "suppression:read",
)

# Default requests-per-minute budget for a new key when the caller doesn't set
# one. Enforced per-key by the public-API rate limiter (see
# app.services.api_key_auth).
DEFAULT_RATE_LIMIT = 60

# The raw-key prefix — namespaces Landline keys so a leaked key is recognizable
# (and greppable) as ours. "llk" = LandLine Key.
KEY_PREFIX = "llk_"


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of a raw API key.

    The single place the key→hash transform is defined, used both when a key is
    created (to store the hash) and on every request (to look a presented key
    up). SHA-256 is appropriate here — unlike a user password, an API key is a
    high-entropy random token, so a fast hash is fine and lets the lookup be a
    single indexed equality match.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class ApiKey(Base, OrgScopedMixin):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Human label for the key ("Zapier integration", "Reporting readonly").
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    # SHA-256 hex digest (64 chars) of the raw key. Unique + indexed: the auth
    # layer looks a presented key up by its hash in one indexed equality match.
    key_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    # First chars of the raw key (non-secret), so the UI can show which key is
    # which without ever exposing the secret.
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    # Granted scopes, a subset of ALL_SCOPES.
    scopes: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    # Per-key requests-per-minute budget.
    rate_limit: Mapped[int] = mapped_column(
        Integer, nullable=False, default=DEFAULT_RATE_LIMIT, server_default=str(DEFAULT_RATE_LIMIT)
    )
    # Last time this key authenticated a request (throttled write — see
    # api_key_auth.touch_last_used). NULL = never used.
    last_used_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    # Optional hard expiry. NULL = never expires.
    expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    # The user (org admin) that minted the key.
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Soft, immediate kill switch. Once set, the key can never authenticate
    # again; the row is retained for audit rather than deleted.
    revoked_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    creator: Mapped["User | None"] = relationship("User", foreign_keys=[created_by])

    @property
    def is_expired(self) -> bool:
        """Whether the key has passed its (optional) hard expiry."""
        if self.expires_at is None:
            return False
        exp = self.expires_at
        # Rows read back from SQLite (tests) can be naive; treat naive as UTC so
        # the comparison never raises on an offset-naive/aware mismatch.
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        return exp <= datetime.now(timezone.utc)

    @property
    def is_revoked(self) -> bool:
        return self.revoked_at is not None

    @property
    def is_usable(self) -> bool:
        """Whether the key may currently authenticate a request."""
        return not self.is_revoked and not self.is_expired

    def has_scope(self, scope: str) -> bool:
        return scope in (self.scopes or [])

    def __repr__(self) -> str:
        return f"<ApiKey {self.name} prefix={self.key_prefix} org={self.org_id}>"
