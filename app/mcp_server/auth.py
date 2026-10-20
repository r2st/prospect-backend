"""API key → org-bound context, once per tool call.

The MCP server reuses the public-API keys rather than inventing a token of its
own, and it reuses the *check* as well: :func:`authenticate` is a thin wrapper
around :func:`app.services.api_key_auth.get_api_key_context`, the dependency
every ``/api/v1/public`` route runs. So revocation, expiry, org suspension and
the per-key rate limit mean the same thing here as there.

It runs on **every call**, not once at startup. The process is long-lived (Claude
Desktop keeps it up for the whole session), and a key validated at launch would
keep working after an admin revoked it — or after a platform admin suspended the
tenant — until someone happened to restart the app.

``get_api_key_context`` binds the key's org with ``set_current_org_id`` and
leaves it bound, which is right for a request task and wrong for a process that
serves many calls; :mod:`app.mcp_server.server` wraps each call in
``org_scope(None)`` so the binding is undone when the call ends.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import scope_users_to_current_org
from app.models.user import User
from app.services.api_key_auth import get_api_key_context


class ToolFailure(Exception):
    """A refusal the caller should see verbatim (bad key, missing scope, 404…)."""


def detail_message(detail) -> str:
    """Flatten an ``HTTPException.detail`` (str, or ``{code, message}``) to text."""
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("code") or detail)
    return str(detail)


@dataclass
class McpContext:
    """The authenticated caller for one tool call."""

    org_id: uuid.UUID
    scopes: list[str]
    api_key_id: uuid.UUID
    created_by: Optional[uuid.UUID]
    _acting_user: Optional[User] = field(default=None, repr=False)

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise ToolFailure(
                f"This API key is missing the '{scope}' scope. "
                "An org admin can mint a key with it under Settings → API Keys."
            )


async def authenticate(db: AsyncSession, raw_key: Optional[str]) -> McpContext:
    """Resolve ``raw_key`` and bind its org as the current context."""
    if not raw_key:
        raise ToolFailure(
            "PROSPECT_API_KEY is not set. Add an API key (llk_…) to the MCP "
            "server's env in your Claude config."
        )
    try:
        ctx = await get_api_key_context(presented_key=raw_key, db=db)
    except HTTPException as exc:
        raise ToolFailure(detail_message(exc.detail)) from None
    return McpContext(
        org_id=ctx.org_id,
        scopes=list(ctx.scopes),
        api_key_id=ctx.api_key.id,
        created_by=ctx.api_key.created_by,
    )


async def acting_user(db: AsyncSession, ctx: McpContext) -> User:
    """The user a write tool acts as: whoever minted the key.

    Creating a campaign needs an owner and a default sending mailbox, and the
    campaign handlers take both from a ``User``. A key has no user of its own,
    so the one that created it stands in — and if that user is gone, inactive or
    no longer in the key's org, the write is refused rather than attributed to
    someone else.
    """
    if ctx._acting_user is not None:
        return ctx._acting_user
    if ctx.created_by is None:
        raise ToolFailure(
            "This API key has no owning user (its creator was removed), so it "
            "can't create or change campaigns. Mint a new key."
        )
    user = (
        await db.execute(
            scope_users_to_current_org(select(User).where(User.id == ctx.created_by)).limit(1)
        )
    ).scalars().first()
    if user is None or not user.is_active or user.org_id != ctx.org_id:
        raise ToolFailure(
            "The user who created this API key is no longer an active member of "
            "this organization, so the key can't create or change campaigns."
        )
    ctx._acting_user = user
    return user
