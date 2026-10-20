"""One-click unsubscribe (RFC 8058) + SMTP List-Unsubscribe headers.

Covers the P0 audit fix: Gmail/Yahoo one-click unsubscribe POSTs to the
``List-Unsubscribe`` URL, so ``/track/unsubscribe`` must accept POST (not only
GET), and the SMTP transport must set the same List-Unsubscribe headers Gmail
already does.
"""

import uuid

import pytest

from app.models.company import Company
from app.models.contact import Contact
from app.routers.tracking import _get_unsubscribe_serializer


def _unsub_token(email: str) -> str:
    return _get_unsubscribe_serializer().dumps(email, salt="unsubscribe")


async def _make_contact(db_session, email: str) -> Contact:
    company = Company(id=uuid.uuid4(), name="Acme", country="US", source="test")
    db_session.add(company)
    await db_session.flush()
    contact = Contact(
        id=uuid.uuid4(), company_id=company.id, email=email,
        first_name="Test", last_name="Lead",
    )
    db_session.add(contact)
    await db_session.commit()
    return contact


@pytest.mark.asyncio
async def test_post_unsubscribe_one_click_marks_contact(client, db_session):
    """RFC 8058 one-click: a POST with a valid token unsubscribes the contact."""
    await _make_contact(db_session, "oneclick@example.com")
    token = _unsub_token("oneclick@example.com")

    resp = await client.post(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 200

    row = (
        await db_session.execute(
            Contact.__table__.select().where(Contact.email == "oneclick@example.com")
        )
    ).first()
    assert row is not None
    assert row.unsubscribed is True


@pytest.mark.asyncio
async def test_get_unsubscribe_renders_a_page_and_the_button_unsubscribes(client, db_session):
    """GET only shows the page; the page's button POST does the opt-out."""
    await _make_contact(db_session, "getunsub@example.com")
    token = _unsub_token("getunsub@example.com")

    resp = await client.get(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 200
    assert "getunsub@example.com" in resp.text

    resp = await client.post(
        f"/api/v1/track/unsubscribe?token={token}", data={"source": "page"}
    )
    assert resp.status_code == 200
    assert "unsubscribed" in resp.text.lower()

    row = (
        await db_session.execute(
            Contact.__table__.select().where(Contact.email == "getunsub@example.com")
        )
    ).first()
    assert row.unsubscribed is True


@pytest.mark.asyncio
async def test_post_unsubscribe_bad_token_returns_400(client):
    """A tampered/expired token gets a 400 — the address stays subscribed."""
    resp = await client.post("/api/v1/track/unsubscribe?token=not-a-real-token")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_post_unsubscribe_is_public(unauth_client, db_session):
    """The one-click POST must work with no credentials (mailbox providers send
    no auth) — it lives under the public /track/ prefix."""
    await _make_contact(db_session, "publicunsub@example.com")
    token = _unsub_token("publicunsub@example.com")
    resp = await unauth_client.post(f"/api/v1/track/unsubscribe?token={token}")
    assert resp.status_code == 200


# --- SMTP List-Unsubscribe headers -----------------------------------------


class _FakeSmtpAccount:
    email = "sender@example.com"
    display_name = "Sender"
    password_encrypted = "x"
    host = "smtp.example.com"
    port = 587
    username = "sender@example.com"
    use_tls = True
    use_ssl = False


def test_smtp_sender_sets_list_unsubscribe_headers():
    from app.services.smtp_sender import SmtpSender

    sender = SmtpSender(_FakeSmtpAccount())
    msg = sender._build_message(
        to_email="lead@example.com",
        subject="Hi",
        body_html="<p>Hi</p>",
        body_text="Hi",
        reply_to=None,
        list_unsubscribe_url="https://track.example.com/api/v1/track/unsubscribe?token=abc",
    )
    assert msg["List-Unsubscribe"] is not None
    assert "https://track.example.com/api/v1/track/unsubscribe?token=abc" in msg["List-Unsubscribe"]
    assert msg["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"


def test_smtp_sender_no_unsubscribe_header_when_url_absent():
    from app.services.smtp_sender import SmtpSender

    sender = SmtpSender(_FakeSmtpAccount())
    msg = sender._build_message(
        to_email="lead@example.com",
        subject="Hi",
        body_html="<p>Hi</p>",
        body_text="Hi",
        reply_to=None,
    )
    assert msg["List-Unsubscribe"] is None


def test_smtp_sender_sets_threading_headers():
    """In-Reply-To / References must be carried so SMTP replies thread."""
    from app.services.smtp_sender import SmtpSender

    sender = SmtpSender(_FakeSmtpAccount())
    msg = sender._build_message(
        to_email="lead@example.com",
        subject="Re: Hi",
        body_html="<p>Hi</p>",
        body_text="Hi",
        reply_to=None,
        in_reply_to="<abc@mail>",
        references="<abc@mail>",
    )
    assert msg["In-Reply-To"] == "<abc@mail>"
    assert msg["References"] == "<abc@mail>"
