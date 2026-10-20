"""Pydantic schemas for the Gmail OAuth connection endpoints."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, computed_field


class AuthorizeResponse(BaseModel):
    """The Google consent URL the frontend should open."""

    authorization_url: str
    state: str


class GmailAccountRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    display_name: Optional[str] = None
    is_active: bool
    sending_enabled: bool = True
    status: str
    daily_send_count: int = 0
    connected_at: datetime
    last_used_at: Optional[datetime] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def deliverability_warning(self) -> Optional[str]:
        """Set for a personal (free-mail) mailbox — see ``is_personal_mailbox``."""
        from app.services.email_dispatch import PERSONAL_MAILBOX_WARNING, is_personal_mailbox

        return PERSONAL_MAILBOX_WARNING if is_personal_mailbox(self.email) else None


class GmailStatusResponse(BaseModel):
    """Overall connection status for the settings UI and compose 'From' line."""

    configured: bool
    connected: bool
    active_email: Optional[str] = None
    active_account_id: Optional[uuid.UUID] = None
    sending_enabled_count: int = 0
    fallback_sender: str
    accounts: list[GmailAccountRead] = []


class ToggleSendingRequest(BaseModel):
    sending_enabled: bool
