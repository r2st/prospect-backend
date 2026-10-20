"""Tests for public-API key management + the X-API-Key auth layer.

Covers minting/listing/revoking keys under ``/org/api-keys`` (org-admin gated,
secret returned once, never re-exposed) and the ``get_api_key_context`` auth
dependency: missing/invalid/revoked/expired keys, scope enforcement, per-key
rate limiting, and cross-org isolation of key lookup.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.models.api_key import ApiKey, hash_key
from app.models.tenant import Organization
from app.models.user import UserRole
from app.services.api_key_auth import generate_api_key


# --- helpers ----------------------------------------------------------------


async def _mint_key(
    db_session,
    org_id,
    scopes,
    *,
    rate_limit=60,
    expires_at=None,
    revoked_at=None,
    name="test-key",
):
    """Insert an ApiKey directly and return (raw_key, ApiKey)."""
    raw, key_hash, prefix = generate_api_key()
    key = ApiKey(
        org_id=org_id,
        name=name,
        key_hash=key_hash,
        key_prefix=prefix,
        scopes=scopes,
        rate_limit=rate_limit,
        expires_at=expires_at,
        revoked_at=revoked_at,
    )
    db_session.add(key)
    await db_session.commit()
    return raw, key


def _hdr(raw_key):
    return {"X-API-Key": raw_key}


# --- management endpoints ----------------------------------------------------


@pytest.mark.asyncio
async def test_create_key_returns_secret_once(client):
    resp = await client.post(
        "/api/v1/org/api-keys",
        json={"name": "Zapier", "scopes": ["contacts:read", "contacts:write"]},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # The raw secret is present exactly here, and looks like a Landline key.
    assert body["api_key"].startswith("llk_")
    assert body["key_prefix"].startswith("llk_")
    assert body["scopes"] == ["contacts:read", "contacts:write"]
    assert body["rate_limit"] == 60
    key_id = body["id"]

    # Listing never re-exposes the secret.
    lst = await client.get("/api/v1/org/api-keys")
    assert lst.status_code == 200
    keys = lst.json()
    assert len(keys) == 1
    assert keys[0]["id"] == key_id
    assert "api_key" not in keys[0]
    assert keys[0]["key_prefix"] == body["key_prefix"]


@pytest.mark.asyncio
async def test_key_hashed_not_plaintext(client, db_session):
    resp = await client.post(
        "/api/v1/org/api-keys", json={"name": "k", "scopes": ["deals:read"]}
    )
    raw = resp.json()["api_key"]
    # The stored hash matches sha256(raw); the raw value appears nowhere in the row.
    from sqlalchemy import select

    row = (await db_session.execute(select(ApiKey))).scalar_one()
    assert row.key_hash == hash_key(raw)
    assert raw not in (row.key_hash, row.key_prefix)


@pytest.mark.asyncio
async def test_invalid_scope_rejected(client):
    resp = await client.post(
        "/api/v1/org/api-keys", json={"name": "k", "scopes": ["contacts:delete"]}
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_scopes_catalog(client):
    resp = await client.get("/api/v1/org/api-keys/scopes")
    assert resp.status_code == 200
    scopes = resp.json()["scopes"]
    assert set(scopes) == {
        "contacts:read", "contacts:write", "companies:read", "companies:write",
        "deals:read", "deals:write", "sequences:read", "sequences:write",
        "campaigns:read", "campaigns:write", "mailboxes:read", "suppression:read",
    }


@pytest.mark.asyncio
async def test_create_requires_org_admin(client, make_user, auth_headers):
    agent = await make_user("agent-keys@landline.app", role=UserRole.ORG_AGENT.value)
    resp = await client.post(
        "/api/v1/org/api-keys",
        json={"name": "k", "scopes": []},
        headers=auth_headers(agent),
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_create_past_expiry_rejected(client):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    resp = await client.post(
        "/api/v1/org/api-keys",
        json={"name": "k", "scopes": [], "expires_at": past},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_revoke_key(client, db_session):
    created = (await client.post(
        "/api/v1/org/api-keys", json={"name": "k", "scopes": ["contacts:read"]}
    )).json()
    key_id = created["id"]

    resp = await client.delete(f"/api/v1/org/api-keys/{key_id}")
    assert resp.status_code == 200
    assert resp.json()["revoked_at"] is not None

    # The revoked key can no longer authenticate the public API.
    r = await client.get("/api/v1/public/contacts", headers=_hdr(created["api_key"]))
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_revoke_other_org_key_404(client, db_session):
    other = Organization(id=uuid.uuid4(), name="Other", slug="other-org", is_active=True)
    db_session.add(other)
    await db_session.commit()
    _, other_key = await _mint_key(db_session, other.id, ["contacts:read"])

    resp = await client.delete(f"/api/v1/org/api-keys/{other_key.id}")
    assert resp.status_code == 404


# --- auth layer --------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_key_401(client):
    resp = await client.get("/api/v1/public/contacts")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_invalid_key_401(client):
    resp = await client.get("/api/v1/public/contacts", headers=_hdr("llk_nope"))
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_expired_key_401(client, db_session):
    from app.org_context import APPREND_ORG_ID

    raw, _ = await _mint_key(
        db_session, APPREND_ORG_ID, ["contacts:read"],
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    resp = await client.get("/api/v1/public/contacts", headers=_hdr(raw))
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_scope_enforced(client, db_session, sample_company):
    from app.org_context import APPREND_ORG_ID

    # A read-only key may list but not create.
    raw, _ = await _mint_key(db_session, APPREND_ORG_ID, ["contacts:read"])
    ok = await client.get("/api/v1/public/contacts", headers=_hdr(raw))
    assert ok.status_code == 200
    denied = await client.post(
        "/api/v1/public/contacts",
        headers=_hdr(raw),
        json={"company_id": str(sample_company.id), "first_name": "A"},
    )
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_rate_limit_429(client, db_session):
    from app.org_context import APPREND_ORG_ID

    raw, _ = await _mint_key(db_session, APPREND_ORG_ID, ["contacts:read"], rate_limit=3)
    statuses = []
    for _ in range(5):
        r = await client.get("/api/v1/public/contacts", headers=_hdr(raw))
        statuses.append(r.status_code)
    # First 3 allowed, then throttled within the same minute window.
    assert statuses[:3] == [200, 200, 200]
    assert 429 in statuses[3:]


@pytest.mark.asyncio
async def test_last_used_recorded(client, db_session):
    from sqlalchemy import select
    from app.org_context import APPREND_ORG_ID

    raw, key = await _mint_key(db_session, APPREND_ORG_ID, ["contacts:read"])
    assert key.last_used_at is None
    await client.get("/api/v1/public/contacts", headers=_hdr(raw))

    row = (
        await db_session.execute(select(ApiKey).where(ApiKey.id == key.id))
    ).scalar_one()
    assert row.last_used_at is not None
