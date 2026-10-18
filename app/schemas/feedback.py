"""Pydantic schemas for in-app feedback."""

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class FeedbackCreate(BaseModel):
    page_name: str = Field(..., min_length=1, max_length=200)
    feedback_text: str = Field(..., min_length=1, max_length=5000)
    # Accepted for client convenience but ignored: the submitter is always the
    # authenticated caller, so one user can't file feedback as another.
    user_id: Optional[uuid.UUID] = None


class FeedbackRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: Optional[uuid.UUID]
    user_email: Optional[str] = None
    page_name: str
    feedback_text: str
    created_at: datetime
