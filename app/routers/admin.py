"""Admin endpoints — user management (CRUD)."""

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import require_role
from app.models.user import User, UserRole, normalize_role
from app.schemas.auth import (
    AdminResetPasswordRequest,
    UserCreate,
    UserCreateResponse,
    UserRead,
    UserUpdate,
)
from app.security import hash_password
from app.services.welcome_email import send_welcome_email

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def _is_platform_admin(user: User) -> bool:
    return normalize_role(user.role) == UserRole.PLATFORM_ADMIN.value


def _scope_users_to_org(query, admin: User):
    """Restrict a User query to the admin's org (no-op for a platform admin).

    Soft-deleted users are out of scope for everyone: they don't list, and
    get/update/reset/delete answer 404 as if they were gone.
    """
    query = query.where(User.deleted_at.is_(None))
    if not _is_platform_admin(admin):
        query = query.where(User.org_id == admin.org_id)
    return query


@router.get("/users", response_model=list[UserRead])
async def list_users(
    role: str | None = Query(
        None, pattern="^(platform_admin|org_admin|org_manager|org_agent|admin|manager|agent)$"
    ),
    is_active: bool | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    _admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """List users in the admin's org (all users for a platform admin)."""
    query = _scope_users_to_org(select(User), _admin)
    if role is not None:
        query = query.where(User.role == normalize_role(role))
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    query = query.order_by(User.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    return list((await db.execute(query)).scalars().all())


@router.post("/users", response_model=UserCreateResponse, status_code=201)
async def create_user(
    payload: UserCreate,
    _admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Create a new user and email them their login credentials. Admin only.

    A welcome email with the temporary password is sent from the primary
    connected sending account. If no account is configured (or the send fails)
    we log a warning and still create the user — the response's
    ``welcome_email_sent`` flag tells the caller what happened.
    """
    existing = (
        await db.execute(select(User).where(User.email == payload.email))
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This email belongs to a deleted user"
                if existing.deleted_at is not None
                else "A user with this email already exists"
            ),
        )

    role = normalize_role(payload.role)
    # New users join the creating admin's org. Only platform admins are allowed
    # a null org_id; an org admin can't create org-less (platform) users.
    if role != UserRole.PLATFORM_ADMIN.value and _admin.org_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot create an org user without a target organization",
        )
    org_id = _admin.org_id if role != UserRole.PLATFORM_ADMIN.value else None

    user = User(
        email=payload.email,
        name=payload.name,
        role=role,
        org_id=org_id,
        password_hash=hash_password(payload.password),
        must_change_password=payload.must_change_password,
        created_by=_admin.id,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    logger.info("Admin %s created user %s (role=%s)", _admin.email, user.email, user.role)

    # Best-effort welcome email with the temporary password. Uses the plaintext
    # password from the request (we only ever store the hash). Never blocks
    # creation — the flag is surfaced to the client instead.
    welcome_email_sent = await send_welcome_email(db, user, payload.password)

    return UserCreateResponse(
        **UserRead.model_validate(user).model_dump(),
        welcome_email_sent=welcome_email_sent,
    )


@router.get("/users/{user_id}", response_model=UserRead)
async def get_user(
    user_id: uuid.UUID,
    _admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Get a single user by ID (within the admin's org). Admin only."""
    user = (
        await db.execute(_scope_users_to_org(select(User).where(User.id == user_id), _admin))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


async def _apply_user_update(
    db: AsyncSession, admin: User, user_id: uuid.UUID, payload: UserUpdate
) -> User:
    """Shared update logic for both PUT and PATCH.

    Enforces email uniqueness, hashes a new password if given, and prevents an
    admin from locking themselves out (deactivating or demoting their own
    account).
    """
    user = (
        await db.execute(_scope_users_to_org(select(User).where(User.id == user_id), admin))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = payload.model_dump(exclude_unset=True)
    if "role" in update_data and update_data["role"] is not None:
        update_data["role"] = normalize_role(update_data["role"])

    # Self-protection: an admin cannot deactivate or demote themselves.
    if user.id == admin.id:
        if update_data.get("is_active") is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot deactivate your own account",
            )
        if "role" in update_data and update_data["role"] != user.role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot change your own role",
            )

    # Email uniqueness check.
    if "email" in update_data and update_data["email"]:
        existing = (
            await db.execute(
                select(User).where(User.email == update_data["email"], User.id != user_id)
            )
        ).scalar_one_or_none()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A user with this email already exists",
            )

    # Hash password if provided.
    if "password" in update_data and update_data["password"]:
        update_data["password_hash"] = hash_password(update_data.pop("password"))
    else:
        update_data.pop("password", None)

    for field, value in update_data.items():
        setattr(user, field, value)

    await db.flush()
    await db.refresh(user)
    logger.info("Admin %s updated user %s", admin.email, user.email)
    return user


@router.put("/users/{user_id}", response_model=UserRead)
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    _admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Update a user (full/partial). Admin only."""
    return await _apply_user_update(db, _admin, user_id, payload)


@router.patch("/users/{user_id}", response_model=UserRead)
async def patch_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    _admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Partially update a user. Admin only."""
    return await _apply_user_update(db, _admin, user_id, payload)


@router.post("/users/{user_id}/reset-password", response_model=UserRead)
async def reset_password(
    user_id: uuid.UUID,
    payload: AdminResetPasswordRequest,
    _admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Reset another user's password. Admin only.

    Sets a new password and, by default, flags the account so the user must
    change it on next login.
    """
    user = (
        await db.execute(_scope_users_to_org(select(User).where(User.id == user_id), _admin))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.password_hash = hash_password(payload.new_password)
    user.must_change_password = payload.must_change_password
    await db.flush()
    await db.refresh(user)
    logger.info("Admin %s reset password for user %s", _admin.email, user.email)
    return user


@router.delete("/users/{user_id}", response_model=UserRead)
async def delete_user(
    user_id: uuid.UUID,
    admin: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete a user. Admin only. Cannot delete yourself.

    We soft-delete — stamp ``deleted_at`` and set ``is_active = False`` —
    rather than hard-deleting so that historical attribution (who contacted
    which client, who sent which email) is preserved. The account can no
    longer log in and drops out of the admin console.
    """
    if user_id == admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account",
        )

    user = (
        await db.execute(_scope_users_to_org(select(User).where(User.id == user_id), admin))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = False
    user.deleted_at = datetime.now(timezone.utc)
    await db.flush()
    await db.refresh(user)
    logger.info("Admin %s soft-deleted user %s", admin.email, user.email)
    return user
