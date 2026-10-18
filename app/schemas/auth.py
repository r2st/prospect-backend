"""Pydantic schemas for authentication and user management."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# --- Auth request / response --------------------------------------------------

class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    # Surfaced so the client can force a password-change flow on first login.
    must_change_password: bool = False
    # Org context of the authenticated user (multi-tenancy). ``org_id`` is None
    # for platform admins. ``role`` lets the client render role-based UI.
    role: Optional[str] = None
    org_id: Optional[uuid.UUID] = None
    org_name: Optional[str] = None


class RefreshRequest(BaseModel):
    refresh_token: str


class RegisterRequest(BaseModel):
    """Public organization sign-up: create an org + its first admin user."""

    org_name: str = Field(..., min_length=2, max_length=255)
    admin_email: EmailStr
    password: str = Field(..., min_length=8)
    # Optional display name for the first user; defaults to the email local-part.
    admin_name: Optional[str] = Field(None, min_length=1, max_length=255)
    # Optional preferred slug; auto-derived from ``org_name`` when omitted.
    slug: Optional[str] = Field(None, min_length=2, max_length=63)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


class ProfileUpdateRequest(BaseModel):
    """Self-service profile update — a user editing their own details."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None


class AdminResetPasswordRequest(BaseModel):
    """Admin resetting another user's password."""

    new_password: str = Field(..., min_length=6)
    must_change_password: bool = True


# --- User CRUD (admin) -------------------------------------------------------

# Canonical roles + legacy aliases (normalized on write). Ordered high→low is
# not important here; membership is what UserCreate/UserUpdate validate against.
VALID_ROLES = (
    "platform_admin",
    "org_admin",
    "org_manager",
    "org_agent",
    "admin",
    "manager",
    "agent",
)


class UserCreate(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)
    role: str = "org_agent"
    password: str = Field(..., min_length=6)
    # Admin-created accounts get a temporary password by default, so the user
    # is required to change it on first login.
    must_change_password: bool = True

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        if v not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return v


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    role: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=6)
    must_change_password: Optional[bool] = None

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        return v


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    name: str
    role: str
    org_id: Optional[uuid.UUID] = None
    is_active: bool
    must_change_password: bool
    created_by: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class UserCreateResponse(UserRead):
    """Response for admin user creation.

    Extends :class:`UserRead` with the outcome of the welcome-email attempt so
    the frontend can tailor its confirmation message. ``welcome_email_sent`` is
    ``False`` when no sending account is configured or the send failed — user
    creation still succeeds either way.
    """

    welcome_email_sent: bool = False
