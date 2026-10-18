"""In-app feedback: the floating widget posts here, admins read it back."""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user, require_role
from app.models.feedback import Feedback
from app.models.user import User
from app.schemas.feedback import FeedbackCreate, FeedbackRead

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackRead, status_code=201)
async def submit_feedback(
    payload: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    item = Feedback(
        user_id=user.id,
        page_name=payload.page_name.strip(),
        feedback_text=payload.feedback_text.strip(),
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    logger.info("Feedback on %s from %s", item.page_name, user.email)
    return FeedbackRead.model_validate(item).model_copy(update={"user_email": user.email})


@router.get("/feedback", response_model=list[FeedbackRead])
async def list_feedback(
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_role("org_admin")),
):
    """Newest first. Org admins see their org's feedback; platform admins see all."""
    rows = (
        await db.execute(
            select(Feedback, User.email)
            .outerjoin(User, User.id == Feedback.user_id)
            .order_by(Feedback.created_at.desc())
            .limit(limit)
        )
    ).all()
    return [
        FeedbackRead.model_validate(fb).model_copy(update={"user_email": email})
        for fb, email in rows
    ]
