"""Tests for the admin user-management router (/api/v1/admin/users)."""

import uuid

import pytest
from httpx import AsyncClient

USERS = "/api/v1/admin/users"


def _user_url(user_id) -> str:
    return f"{USERS}/{user_id}"


# ─── List ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_users(client: AsyncClient, admin_user, agent_user, auth_headers):
    resp = await client.get(USERS, headers=auth_headers(admin_user))
    assert resp.status_code == 200, resp.text
    emails = {u["email"] for u in resp.json()}
    assert {"admin@landline.app", "agent@landline.app"} <= emails


# ─── Create ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_user(client: AsyncClient, admin_user, auth_headers):
    resp = await client.post(
        USERS,
        headers=auth_headers(admin_user),
        json={
            "email": "sam@landline.app",
            "name": "Sam Sales",
            "role": "org_manager",
            "password": "initialpw123",
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["email"] == "sam@landline.app"
    assert data["role"] == "org_manager"
    assert data["must_change_password"] is True
    assert data["created_by"] == str(admin_user.id)
    assert "password_hash" not in data


@pytest.mark.asyncio
async def test_created_user_can_log_in(client: AsyncClient, admin_user, auth_headers):
    await client.post(
        USERS,
        headers=auth_headers(admin_user),
        json={"email": "log@landline.app", "name": "Log In", "role": "agent", "password": "initialpw123"},
    )
    login = await client.post(
        "/api/v1/auth/login", json={"email": "log@landline.app", "password": "initialpw123"}
    )
    assert login.status_code == 200
    assert login.json()["must_change_password"] is True


@pytest.mark.asyncio
async def test_create_duplicate_email_409(client: AsyncClient, admin_user, agent_user, auth_headers):
    resp = await client.post(
        USERS,
        headers=auth_headers(admin_user),
        json={"email": "agent@landline.app", "name": "Dup", "role": "agent", "password": "initialpw123"},
    )
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_create_invalid_role_422(client: AsyncClient, admin_user, auth_headers):
    resp = await client.post(
        USERS,
        headers=auth_headers(admin_user),
        json={"email": "x@landline.app", "name": "X", "role": "superuser", "password": "initialpw123"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_short_password_422(client: AsyncClient, admin_user, auth_headers):
    resp = await client.post(
        USERS,
        headers=auth_headers(admin_user),
        json={"email": "x@landline.app", "name": "X", "role": "agent", "password": "short"},
    )
    assert resp.status_code == 422


# ─── Update ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_user_role(client: AsyncClient, admin_user, agent_user, auth_headers):
    resp = await client.patch(
        _user_url(agent_user.id),
        headers=auth_headers(admin_user),
        json={"role": "org_manager"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"] == "org_manager"


@pytest.mark.asyncio
async def test_deactivate_via_patch(client: AsyncClient, admin_user, agent_user, auth_headers):
    resp = await client.patch(
        _user_url(agent_user.id),
        headers=auth_headers(admin_user),
        json={"is_active": False},
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False
    # Deactivated user can no longer log in.
    login = await client.post("/api/v1/auth/login", json={"email": "agent@landline.app", "password": "agentpass123"})
    assert login.status_code == 403


@pytest.mark.asyncio
async def test_update_unknown_user_404(client: AsyncClient, admin_user, auth_headers):
    resp = await client.patch(
        _user_url(uuid.uuid4()), headers=auth_headers(admin_user), json={"role": "agent"}
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_cannot_deactivate_self(client: AsyncClient, admin_user, auth_headers):
    resp = await client.patch(
        _user_url(admin_user.id), headers=auth_headers(admin_user), json={"is_active": False}
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_admin_cannot_demote_self(client: AsyncClient, admin_user, auth_headers):
    resp = await client.patch(
        _user_url(admin_user.id), headers=auth_headers(admin_user), json={"role": "agent"}
    )
    assert resp.status_code == 400


# ─── Delete (soft) ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_soft_delete_user(client: AsyncClient, admin_user, agent_user, auth_headers):
    resp = await client.delete(_user_url(agent_user.id), headers=auth_headers(admin_user))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["is_active"] is False
    assert body["deleted_at"] is not None
    # Gone from the console, unlike a deactivated user.
    listing = await client.get(USERS, headers=auth_headers(admin_user))
    assert all(u["email"] != "agent@landline.app" for u in listing.json())


@pytest.mark.asyncio
async def test_soft_delete_keeps_the_row(
    client: AsyncClient, admin_user, agent_user, auth_headers, db_session
):
    from sqlalchemy import select

    from app.models.user import User

    await client.delete(_user_url(agent_user.id), headers=auth_headers(admin_user))
    row = (
        await db_session.execute(
            select(User).where(User.id == agent_user.id).execution_options(populate_existing=True)
        )
    ).scalar_one()
    assert row.deleted_at is not None
    assert row.is_active is False


@pytest.mark.asyncio
async def test_deleted_user_cannot_log_in(client: AsyncClient, admin_user, agent_user, auth_headers):
    await client.delete(_user_url(agent_user.id), headers=auth_headers(admin_user))
    login = await client.post(
        "/api/v1/auth/login", json={"email": "agent@landline.app", "password": "agentpass123"}
    )
    assert login.status_code == 403


@pytest.mark.asyncio
async def test_deleted_user_cannot_be_reactivated_or_deleted_again(
    client: AsyncClient, admin_user, agent_user, auth_headers
):
    await client.delete(_user_url(agent_user.id), headers=auth_headers(admin_user))
    patch = await client.patch(
        _user_url(agent_user.id), headers=auth_headers(admin_user), json={"is_active": True}
    )
    assert patch.status_code == 404
    again = await client.delete(_user_url(agent_user.id), headers=auth_headers(admin_user))
    assert again.status_code == 404


@pytest.mark.asyncio
async def test_recreating_a_deleted_users_email_409_says_why(
    client: AsyncClient, admin_user, agent_user, auth_headers
):
    await client.delete(_user_url(agent_user.id), headers=auth_headers(admin_user))
    resp = await client.post(
        USERS,
        headers=auth_headers(admin_user),
        json={"email": "agent@landline.app", "name": "Back", "role": "agent", "password": "initialpw123"},
    )
    assert resp.status_code == 409
    assert "deleted" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_admin_cannot_delete_self(client: AsyncClient, admin_user, auth_headers):
    resp = await client.delete(_user_url(admin_user.id), headers=auth_headers(admin_user))
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_delete_unknown_user_404(client: AsyncClient, admin_user, auth_headers):
    resp = await client.delete(_user_url(uuid.uuid4()), headers=auth_headers(admin_user))
    assert resp.status_code == 404
