"""Prospects stuck in ``sending`` are failed and reported, never retried."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.campaign import Campaign, CampaignDailyLog, CampaignProspect
from app.models.gmail_account import GmailAccount
from app.services import campaigns as svc

NOW = datetime(2026, 10, 19, 15, 0, tzinfo=timezone.utc)


async def _campaign(db) -> Campaign:
    acct = GmailAccount(
        id=uuid.uuid4(), email=f"s-{uuid.uuid4().hex[:6]}@example.com",
        google_sub=uuid.uuid4().hex, refresh_token_encrypted="x",
        status="connected", sending_enabled=True, is_active=True,
    )
    db.add(acct)
    await db.flush()
    c = Campaign(
        name="T", status="active", document_ids=[], icp_json={"summary": "x"},
        gmail_account_id=acct.id, daily_send_limit=20, send_window_start="09:00",
        send_window_end="18:00", timezone="America/Toronto", send_on_weekends=False,
        review_before_send=False, attach_documents=False, warmup_day=1,
    )
    db.add(c)
    await db.commit()
    return c


@pytest.mark.asyncio
async def test_old_claims_are_failed_and_logged_fresh_ones_left(db_session):
    c = await _campaign(db_session)
    stuck = CampaignProspect(campaign_id=c.id, to_email="old@acme.com", status="sending",
                             sent_at=NOW - timedelta(minutes=45))
    fresh = CampaignProspect(campaign_id=c.id, to_email="new@acme.com", status="sending",
                             sent_at=NOW - timedelta(minutes=5))
    done = CampaignProspect(campaign_id=c.id, to_email="sent@acme.com", status="sent",
                            sent_at=NOW - timedelta(hours=3))
    db_session.add_all([stuck, fresh, done])
    await db_session.commit()

    result = await svc.reap_stuck_sends(db_session, NOW)

    assert result == {"reaped": 1, "campaigns": 1}
    await db_session.refresh(stuck)
    await db_session.refresh(fresh)
    await db_session.refresh(done)
    assert stuck.status == "failed" and "outcome unknown" in stuck.error.lower()
    assert stuck.sent_at is not None  # the claim time is kept
    assert fresh.status == "sending"
    assert done.status == "sent"

    log = (await db_session.execute(
        select(CampaignDailyLog).where(CampaignDailyLog.campaign_id == c.id)
    )).scalars().one()
    assert any("old@acme.com" in e["error"] for e in log.errors)


@pytest.mark.asyncio
async def test_nothing_stuck_is_a_no_op(db_session):
    await _campaign(db_session)
    assert await svc.reap_stuck_sends(db_session, NOW) == {"reaped": 0, "campaigns": 0}
