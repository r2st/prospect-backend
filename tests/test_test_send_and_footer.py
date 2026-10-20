"""Tests for the compliant test-send endpoint, footer address, and warm-up reset."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.services import system_settings


# --- Footer: unsubscribe + DoAide attribution only ---


def test_footer_is_unsubscribe_and_postal_address_only():
    """CAN-SPAM needs the sender's name and postal address; nothing else goes in.
    In particular no "Powered by" link — a cross-domain link in every message."""
    from app.services.email_dispatch import _unsubscribe_footer_text, _unsubscribe_footer_html

    settings = get_settings()
    text = _unsubscribe_footer_text("prospect@example.com")
    html = _unsubscribe_footer_html("prospect@example.com")

    for footer in (text, html):
        assert "unsubscribe" in footer.lower()
        assert settings.company_name in footer
        assert settings.company_postal_address in footer
        assert "Powered by" not in footer
        assert 'href="https://prospect.doaide.com"' not in footer


def test_footer_uses_the_org_sender_identity_when_set():
    from app.services.email_dispatch import SenderIdentity, _unsubscribe_footer_text

    footer = _unsubscribe_footer_text(
        "p@example.com", SenderIdentity("DevHeal Labs AI Pvt Ltd", "1 MG Road, Bengaluru 560001, India")
    )
    assert footer.rstrip().endswith("DevHeal Labs AI Pvt Ltd, 1 MG Road, Bengaluru 560001, India")


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
        assert "Powered by" not in part
    settings = get_settings()
    assert call["body_text"].rstrip().endswith(
        f"{settings.company_name}, {settings.company_postal_address}"
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


# --- Deliverability: headers, encoding, mailbox choice ---


def test_gmail_mime_is_well_formed_for_spam_filters():
    """Date + Message-ID on the sender's domain, one-click unsubscribe headers,
    and quoted-printable (not base64) text parts."""
    import base64
    import email
    from types import SimpleNamespace

    from app.services.gmail_sender import GmailSender

    sender = GmailSender(SimpleNamespace(email="rep@devheallabs.com", display_name="Rep"))
    raw = sender._build_mime(
        "p@example.com", "Hello", "<p>Hi — there</p>", "Hi — there", None,
        list_unsubscribe_url="https://prospect.doaide.com/api/v1/track/unsubscribe?token=t",
    )["raw"]
    msg = email.message_from_bytes(base64.urlsafe_b64decode(raw))

    assert msg["Date"]
    assert msg["Message-ID"].endswith("@devheallabs.com>")
    assert msg["From"] == "Rep <rep@devheallabs.com>"
    assert "https://prospect.doaide.com/api/v1/track/unsubscribe?token=t" in msg["List-Unsubscribe"]
    assert msg["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"
    parts = [p for p in msg.walk() if p.get_content_maintype() == "text"]
    assert [p.get_content_type() for p in parts] == ["text/plain", "text/html"]
    for p in parts:
        assert p["Content-Transfer-Encoding"] == "quoted-printable"
        assert p.get_content_charset() == "utf-8"


@pytest.mark.asyncio
async def test_rotation_prefers_a_business_mailbox_over_a_personal_one(db_session: AsyncSession):
    import uuid
    from datetime import datetime, timezone

    from app.models.gmail_account import GmailAccount
    from app.services import email_dispatch

    def acct(addr):
        return GmailAccount(
            id=uuid.uuid4(), email=addr, google_sub=str(uuid.uuid4()), display_name="R",
            refresh_token_encrypted="x", status="connected", sending_enabled=True,
            connected_at=datetime.now(timezone.utc),
        )

    personal, business = acct("someone@gmail.com"), acct("rep@devheallabs.com")
    # The personal one is "least recently used" — rotation would pick it first.
    business.last_used_at = datetime.now(timezone.utc)
    db_session.add_all([personal, business])
    await db_session.flush()

    for _ in range(3):
        picked = await email_dispatch.get_next_rotation_account(db_session)
        assert picked.email == "rep@devheallabs.com"

    assert email_dispatch.is_personal_mailbox("someone@gmail.com")
    assert not email_dispatch.is_personal_mailbox("rep@devheallabs.com")


def test_low_text_to_link_ratio_is_flagged():
    from app.services.deliverability_optimizer import get_deliverability_optimizer

    opt = get_deliverability_optimizer()
    thin = opt.scan_content_heuristic("Quick look", "<p>See https://example.com/x</p>")
    assert any("text-to-link" in i for i in thin["issues"])

    body = "<p>" + "We help infrastructure teams verify recovery outcomes. " * 6 + "https://example.com</p>"
    ok = opt.scan_content_heuristic("Infrastructure operations", body)
    assert not any("text-to-link" in i for i in ok["issues"])


def test_wrapping_never_gives_a_tag_two_style_attributes():
    import re

    from app.services.email_dispatch import _unsubscribe_footer_html, wrap_html_email

    html = wrap_html_email("<p>Hello <a href='https://example.com'>site</a></p>" + _unsubscribe_footer_html("p@example.com"))
    for tag in re.findall(r"<[a-z][^>]*>", html):
        assert len(re.findall(r"\bstyle\s*=", tag)) <= 1, tag
    assert '<p style="margin:0 0 10px 0">Hello' in html
    assert '<a style="color:#1a73e8;text-decoration:underline" href=' in html
