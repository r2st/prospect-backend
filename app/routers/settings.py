"""User-facing settings — per-org LLM key management.

Mounted at ``/settings``. Lets org admins bring their own API keys for the
Discovery features: Gemini, plus Groq and OpenRouter as fallbacks used when
Gemini is out of quota or unavailable (see ``services.org_gemini``). One key per
provider; ``?provider=`` on GET/DELETE defaults to ``gemini`` so clients that
predate the fallbacks keep working.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import OrgScope, get_org_scope
from app.models.org_llm_key import OrgLlmKey
from app.models.user import UserRole
from app.services import crypto
from app.services.org_gemini import DISCOVERY_PROVIDERS, PROVIDER_LABELS, SEARCH_PROVIDERS
from app.services.credential_store import mask

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])


# LLMs (Discovery's fallback chain) and web-search APIs (Discovery's
# grounding when Gemini's isn't available), stored in the same table.
Provider = Literal["gemini", "anthropic", "groq", "openrouter", "tavily", "serper", "brave"]
assert set(Provider.__args__) == set(DISCOVERY_PROVIDERS) | set(SEARCH_PROVIDERS)


class LlmKeyUpdate(BaseModel):
    api_key: str = Field(..., min_length=1, max_length=500)
    # Exactly the providers ``services.org_gemini`` reads back; a key stored
    # under any other name would report "saved" and never be used.
    provider: Provider = "gemini"

    @field_validator("api_key")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # Whitespace-only would be stored as "" — has_key=True here while
        # Discovery still sees no key.
        v = v.strip()
        if not v:
            raise ValueError("API key must not be blank")
        return v


class LlmKeyRead(BaseModel):
    provider: str
    key_hint: str
    has_key: bool
    # "llm" | "search" — lets the settings page group them.
    kind: str = "llm"
    label: str = ""


def _read(provider: str, row: "OrgLlmKey | None") -> LlmKeyRead:
    return LlmKeyRead(
        provider=provider,
        key_hint=row.key_hint if row else "",
        has_key=row is not None,
        kind="search" if provider in SEARCH_PROVIDERS else "llm",
        label=PROVIDER_LABELS.get(provider, provider),
    )


def _require_org_admin(scope: OrgScope) -> None:
    if scope.org_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No organization context",
        )
    scope.check_permission(UserRole.ORG_ADMIN.value)


async def _key_row(db: AsyncSession, provider: str) -> OrgLlmKey | None:
    # No unique constraint on (org_id, provider): newest wins, id breaks ties —
    # the same row ``org_gemini.get_org_llm_keys`` reads.
    return (
        await db.execute(
            select(OrgLlmKey)
            .where(OrgLlmKey.provider == provider)
            .order_by(OrgLlmKey.updated_at.desc(), OrgLlmKey.id)
            .limit(1)
        )
    ).scalars().first()


@router.get("/llm-keys", response_model=list[LlmKeyRead])
async def list_llm_keys(
    scope: OrgScope = Depends(get_org_scope),
    db: AsyncSession = Depends(get_db),
):
    """Every supported provider — LLMs in fallback order, then web search —
    with whether a key is saved."""
    _require_org_admin(scope)
    return [
        _read(provider, await _key_row(db, provider))
        for provider in (*DISCOVERY_PROVIDERS, *SEARCH_PROVIDERS)
    ]


@router.get("/llm-key", response_model=LlmKeyRead)
async def get_llm_key(
    provider: Provider = Query("gemini"),
    scope: OrgScope = Depends(get_org_scope),
    db: AsyncSession = Depends(get_db),
):
    """Check whether the org has a key set for ``provider``."""
    _require_org_admin(scope)
    return _read(provider, await _key_row(db, provider))


@router.put("/llm-key", response_model=LlmKeyRead)
async def set_llm_key(
    body: LlmKeyUpdate,
    scope: OrgScope = Depends(get_org_scope),
    db: AsyncSession = Depends(get_db),
):
    """Set or replace the org's custom LLM API key."""
    _require_org_admin(scope)
    api_key = body.api_key

    row = await _key_row(db, body.provider)

    if row is None:
        row = OrgLlmKey(provider=body.provider)
        db.add(row)

    row.key_encrypted = crypto.encrypt(api_key)
    row.key_hint = mask(api_key)
    row.updated_by = scope.user.email
    await db.flush()
    await db.refresh(row)

    logger.info(
        "Org %s LLM key (%s) updated by %s",
        scope.org_id, body.provider, scope.user.email,
    )
    return _read(body.provider, row)


@router.delete("/llm-key", status_code=status.HTTP_204_NO_CONTENT)
async def delete_llm_key(
    provider: Provider = Query("gemini"),
    scope: OrgScope = Depends(get_org_scope),
    db: AsyncSession = Depends(get_db),
):
    """Remove the org's key for ``provider``."""
    _require_org_admin(scope)
    # Every row for the provider: a leftover duplicate would otherwise surface
    # as "still saved" right after the user removed it.
    rows = (
        await db.execute(select(OrgLlmKey).where(OrgLlmKey.provider == provider))
    ).scalars().all()
    for row in rows:
        await db.delete(row)
    if rows:
        await db.flush()
        logger.info(
            "Org %s LLM key (%s) removed by %s", scope.org_id, provider, scope.user.email
        )
