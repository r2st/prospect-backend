"""Tests for the compliant test-send endpoint, footer address, and warm-up reset."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services import system_settings


# --- Footer: unsubscribe + DoAide attribution only ---


def test_footer_is_unsubscribe_and_powered_by_only():
    from app.services.email_dispatch import _unsubscribe_footer_text, _unsubscribe_footer_html

    settings = get_settings()
    text = _unsubscribe_footer_text("prospect@example.com")
    html = _unsubscribe_footer_html("prospect@example.com")

    for footer in (text, html):
        assert "unsubscribe" in footer.lower()
        assert "Powered by DoAide Prospect" in footer
        assert "https://prospect.doaide.com" in footer
        assert "Bangalore" not in footer
        assert settings.company_postal_address not in footer
        assert settings.company_name not in footer
    assert '<a href="https://prospect.doaide.com"' in html


@pytest.mark.asyncio
async def test_sent_email_carries_the_footer_exactly_once(db_session: AsyncSession, monkeypatch):
    """The text part was built from the HTML *after* the HTML footer went in,
    then the text footer was appended too — every text/plain part had the
    unsubscribe block twice (seen in a real send that landed in spam)."""
    import uuid
    from datetime import datetime, timezone

    from app.models.gmail_account import GmailAccount
    from app.services import email_dispatch

    account = GmailAccount(
        id=uuid.uuid4(), email="rep@example.com", google_sub=str(uuid.uuid4()),
        display_name="Rep", refresh_token_encrypted="x", status="connected",
        sending_enabled=True, connected_at=datetime.now(timezone.utc),
    )
    db_session.add(account)
    await db_session.flush()

    sent = []

    class _Transport:
        def send_email(self, **kwargs):
            sent.append(kwargs)
            return {"status": "sent", "transport": "gmail", "sender": account.email}

    monkeypatch.setattr(email_dispatch, "_make_sender", lambda a: _Transport())
    await email_dispatch.send_outreach_email(
        db_session, to_email="prospect@example.com", subject="Hi",
        body_html="<p>Hello there.</p>", bypass_kill_switch=True,
    )

    (call,) = sent
    for part in (call["body_text"], call["body_html"]):
        assert part.count("Reply UNSUBSCRIBE") == 1
        assert part.count("Powered by DoAide Prospect") == 1
        assert "Bangalore" not in part
    assert call["body_text"].rstrip().endswith(
        "Powered by DoAide Prospect — https://prospect.doaide.com"
    )
    assert call["list_unsubscribe_url"] in call["body_text"]


# --- Test-send endpoint ---


@pytest.mark.asyncio
async def test_test_send_halted_when_disabled(client: AsyncClient, db_session: AsyncSession):
    """A test send still honours the kill switch, but domain auth is reported."""
    await system_settings.set_sending_enabled(db_session, False)
    await db_session.commit()

    with patch("app.routers.email_test.domain_auth.check_domain_auth", return_value={
        "domain": "apprendtechnologies.com", "overall": "fail",
        "spf": None, "dkim": None, "dmarc": None, "recommendations": ["add SPF"],
    }):
        resp = await client.post("/api/v1/email/test-send", json={"to_email": "me@example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sent"] is False
    assert body["status"] == "halted"
    # Deliverability posture is still returned so the operator can diagnose.
    assert body["domain_auth"]["overall"] == "fail"


@pytest.mark.asyncio
async def test_test_send_sends_when_enabled(client: AsyncClient):
    sent = {"status": "sent", "transport": "gmail", "from_email": "s@apprendtechnologies.com"}
    with patch("app.routers.email_test.send_outreach_email", new=AsyncMock(return_value=sent)) as mock_send, \
         patch("app.routers.email_test.domain_auth.check_domain_auth", return_value={
             "domain": "apprendtechnologies.com", "overall": "pass",
             "spf": {"status": "pass"}, "dkim": {"status": "pass"}, "dmarc": {"status": "warn"},
             "recommendations": [],
         }):
        resp = await client.post(
            "/api/v1/email/test-send",
            json={"to_email": "me@example.com", "subject": "hi", "body": "<p>hi</p>"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sent"] is True
    assert body["status"] == "sent"
    assert body["domain_auth"]["overall"] == "pass"
    mock_send.assert_awaited_once()


# --- Warm-up reset / auto-start ---


@pytest.mark.asyncio
async def test_reset_warmup_endpoint(client: AsyncClient):
    resp = await client.post("/api/v1/system/reset-warmup")
    assert resp.status_code == 200
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).date().isoformat()
    assert resp.json()["warmup_start_date"] == today


@pytest.mark.asyncio
async def test_ensure_warmup_started_auto_sets(db_session: AsyncSession):
    from datetime import datetime, timezone

    assert await system_settings.get_value(db_session, system_settings.WARMUP_START_KEY) is None
    date = await system_settings.ensure_warmup_started(db_session)
    today = datetime.now(timezone.utc).date().isoformat()
    assert date == today
    # Idempotent — a second call doesn't move the date.
    again = await system_settings.ensure_warmup_started(db_session)
    assert again == today


@pytest.mark.asyncio
async def test_warmup_persisted_date_overrides_config(db_session: AsyncSession, monkeypatch):
    from app.services.send_limiter import get_send_limiter

    limiter = get_send_limiter()
    # Config says warm-up started long ago (fully warmed → max limit)...
    monkeypatch.setattr(limiter.settings, "warmup_enabled", True)
    monkeypatch.setattr(limiter.settings, "warmup_start_date", "2020-01-01")
    monkeypatch.setattr(limiter.settings, "max_emails_per_day", 50)
    monkeypatch.setattr(limiter.settings, "warmup_daily_limit_week1", 10)

    # ...but a reset today persists a fresh start → week-1 cap wins.
    await system_settings.reset_warmup_start_date(db_session)
    limit = await limiter._get_daily_limit_db(db_session)
    assert limit == 10
