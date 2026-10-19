"""Email template attachments: stored per template, sent with every email from it.

Three layers, tested separately because each fails differently:

* **the endpoints** — upload, download, delete, and the limits that have to be
  enforced at upload, since at send time an oversized or blocked file fails
  every recipient in the batch;
* **the send paths** — compose, send-test and the scheduled sweep each have to
  hand the template's files to the dispatcher, and a path that forgets sends a
  perfectly good email without the deck it promises;
* **the wire format** — each transport has to produce ``multipart/mixed`` with
  the body first, and must produce exactly the old ``multipart/alternative``
  when there is nothing attached.
"""

import base64
import email
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.email_template import EmailSend, EmailTemplate, EmailTemplateAttachment
from app.models.tenant import Organization
from app.models.user import UserRole
from app.org_context import APPREND_ORG_ID
from app.routers import email_templates as router_mod
from app.services import email_dispatch, email_sender as ses_mod
from app.services import template_attachments as ta
from app.services.gmail_sender import GmailSender
from app.services.smtp_sender import SmtpSender
from app.tasks import scheduled_sends

PDF = b"%PDF-1.4\n" + b"x" * 2048


async def _template(client, name="Deck send"):
    resp = await client.post("/api/v1/email-templates", json={
        "name": name, "subject": "Our deck", "body": "<p>Hi {{first_name}}</p>",
    })
    assert resp.status_code == 201, resp.text
    return resp.json()


async def _upload(client, template_id, filename="deck.pdf", data=PDF, ctype="application/pdf"):
    return await client.post(
        f"/api/v1/email-templates/{template_id}/attachments",
        files={"file": (filename, data, ctype)},
    )


# --- Endpoints ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_lists_metadata_on_the_template(client):
    tpl = await _template(client)
    resp = await _upload(client, tpl["id"])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["filename"] == "deck.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["size_bytes"] == len(PDF)

    listing = (await client.get("/api/v1/email-templates")).json()
    [found] = [t for t in listing if t["id"] == tpl["id"]]
    assert [a["id"] for a in found["attachments"]] == [body["id"]]
    # Metadata only: the picker must not ship every file's bytes.
    assert "content" not in found["attachments"][0]


@pytest.mark.asyncio
async def test_a_new_template_has_no_attachments(client):
    tpl = await _template(client)
    assert tpl["attachments"] == []


@pytest.mark.asyncio
async def test_download_returns_the_bytes_as_a_download(client):
    tpl = await _template(client)
    att = (await _upload(client, tpl["id"], filename="Price sheet.pdf")).json()

    resp = await client.get(f"/api/v1/email-templates/{tpl['id']}/attachments/{att['id']}")
    assert resp.status_code == 200
    assert resp.content == PDF
    # Never inline: the type is whatever the uploader's browser declared.
    assert resp.headers["content-disposition"].startswith("attachment;")
    assert "Price%20sheet.pdf" in resp.headers["content-disposition"]
    assert resp.headers["x-content-type-options"] == "nosniff"


@pytest.mark.asyncio
async def test_delete_removes_it(client):
    tpl = await _template(client)
    att = (await _upload(client, tpl["id"])).json()

    resp = await client.delete(f"/api/v1/email-templates/{tpl['id']}/attachments/{att['id']}")
    assert resp.status_code == 204

    listing = (await client.get("/api/v1/email-templates")).json()
    [found] = [t for t in listing if t["id"] == tpl["id"]]
    assert found["attachments"] == []
    gone = await client.get(f"/api/v1/email-templates/{tpl['id']}/attachments/{att['id']}")
    assert gone.status_code == 404


@pytest.mark.asyncio
async def test_an_attachment_is_only_reachable_through_its_own_template(client):
    a = await _template(client, "A")
    b = await _template(client, "B")
    att = (await _upload(client, a["id"])).json()

    url = f"/api/v1/email-templates/{b['id']}/attachments/{att['id']}"
    assert (await client.get(url)).status_code == 404
    assert (await client.delete(url)).status_code == 404


@pytest.mark.asyncio
async def test_upload_to_a_missing_template_is_404(client):
    resp = await _upload(client, uuid.uuid4())
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_blocked_file_types_are_refused_at_upload(client):
    tpl = await _template(client)
    resp = await _upload(client, tpl["id"], filename="setup.EXE", ctype="application/octet-stream")
    assert resp.status_code == 400
    assert ".exe" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_an_empty_file_is_refused(client):
    tpl = await _template(client)
    resp = await _upload(client, tpl["id"], data=b"")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_a_file_over_the_per_file_cap_is_refused(client, monkeypatch):
    monkeypatch.setattr(ta, "MAX_FILE_BYTES", 1024)
    tpl = await _template(client)
    resp = await _upload(client, tpl["id"], data=b"x" * 1025)
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_the_template_total_is_capped(client, monkeypatch):
    monkeypatch.setattr(ta, "MAX_TOTAL_BYTES", 3000)
    tpl = await _template(client)
    assert (await _upload(client, tpl["id"], data=b"a" * 2000)).status_code == 201
    resp = await _upload(client, tpl["id"], filename="second.pdf", data=b"b" * 1500)
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_the_number_of_files_is_capped(client, monkeypatch):
    monkeypatch.setattr(ta, "MAX_FILES", 2)
    tpl = await _template(client)
    for n in range(2):
        assert (await _upload(client, tpl["id"], filename=f"f{n}.pdf")).status_code == 201
    assert (await _upload(client, tpl["id"], filename="f2.pdf")).status_code == 400


@pytest.mark.asyncio
async def test_a_hostile_filename_is_reduced_to_a_safe_basename(client):
    tpl = await _template(client)
    # Control characters are covered by ``test_sanitize_filename`` below: an
    # HTTP client percent-encodes them in a multipart filename before they
    # could reach the server, so only the helper can be handed them raw.
    resp = await _upload(client, tpl["id"], filename="../../etc/passwd.pdf")
    assert resp.status_code == 201
    assert resp.json()["filename"] == "passwd.pdf"


@pytest.mark.asyncio
async def test_a_malformed_content_type_is_replaced_by_a_guess(client):
    tpl = await _template(client)
    resp = await _upload(client, tpl["id"], filename="notes.txt", data=b"hello",
                         ctype="text/plain\r\nX-Evil: 1")
    assert resp.status_code == 201
    assert resp.json()["content_type"] == "text/plain"


@pytest.mark.asyncio
async def test_another_orgs_template_cannot_take_attachments(
    client, db_session: AsyncSession, make_user, auth_headers
):
    other = Organization(
        id=uuid.uuid4(), name="Other Co", slug="other-co", is_active=True, billing_plan="pro",
    )
    db_session.add(other)
    foreign = EmailTemplate(
        id=uuid.uuid4(), name="Theirs", subject="s", body="b", org_id=other.id,
    )
    db_session.add(foreign)
    await db_session.commit()

    me = await make_user("rep@apprend.tech", role=UserRole.ADMIN.value, org_id=APPREND_ORG_ID)
    resp = await client.post(
        f"/api/v1/email-templates/{foreign.id}/attachments",
        files={"file": ("deck.pdf", PDF, "application/pdf")},
        headers=auth_headers(me),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_clone_copies_the_attachments(client):
    tpl = await _template(client)
    await _upload(client, tpl["id"], filename="deck.pdf")

    resp = await client.post(f"/api/v1/email-templates/{tpl['id']}/clone")
    assert resp.status_code == 201, resp.text
    copy = resp.json()
    assert [a["filename"] for a in copy["attachments"]] == ["deck.pdf"]
    att_id = copy["attachments"][0]["id"]
    dl = await client.get(f"/api/v1/email-templates/{copy['id']}/attachments/{att_id}")
    assert dl.content == PDF


@pytest.mark.asyncio
async def test_deleting_a_template_deletes_its_attachments(client, db_session: AsyncSession):
    await _template(client, "Default")  # first one becomes default; delete the second
    tpl = await _template(client, "Draft")
    await _upload(client, tpl["id"])

    assert (await client.delete(f"/api/v1/email-templates/{tpl['id']}")).status_code == 204
    left = (await db_session.execute(select(EmailTemplateAttachment))).scalars().all()
    assert left == []


# --- Send paths --------------------------------------------------------------


@pytest.fixture
def router_sends(monkeypatch):
    calls = []

    async def _fake(db, **kwargs):
        calls.append(kwargs)
        return {"status": "sent", "message_id": "m-1"}

    monkeypatch.setattr(router_mod, "send_outreach_email", _fake)
    return calls


@pytest.mark.asyncio
async def test_compose_sends_the_templates_files(client, sample_contact, router_sends):
    tpl = await _template(client)
    await _upload(client, tpl["id"], filename="deck.pdf")

    resp = await client.post("/api/v1/email-compose", json={
        "contact_ids": [str(sample_contact.id)], "template_id": tpl["id"],
    })
    assert resp.status_code == 200, resp.text
    [call] = router_sends
    assert [(a.filename, a.content_type, a.data) for a in call["attachments"]] == [
        ("deck.pdf", "application/pdf", PDF)
    ]


@pytest.mark.asyncio
async def test_compose_without_attachments_sends_none(client, sample_contact, router_sends):
    tpl = await _template(client)
    await client.post("/api/v1/email-compose", json={
        "contact_ids": [str(sample_contact.id)], "template_id": tpl["id"],
    })
    [call] = router_sends
    assert call["attachments"] == []


@pytest.mark.asyncio
async def test_send_test_includes_the_files(client, router_sends):
    tpl = await _template(client)
    await _upload(client, tpl["id"], filename="deck.pdf")

    resp = await client.post("/api/v1/email-templates/send-test", json={
        "template_id": tpl["id"], "test_email": "me@apprend.tech",
    })
    assert resp.json()["status"] == "sent", resp.text
    [call] = router_sends
    assert [a.filename for a in call["attachments"]] == ["deck.pdf"]


@pytest.mark.asyncio
async def test_the_scheduled_sweep_sends_the_templates_files(
    db_session: AsyncSession, monkeypatch
):
    calls = []

    async def _fake(db, **kwargs):
        calls.append(kwargs)
        return {"status": "sent", "message_id": "m-1"}

    monkeypatch.setattr(scheduled_sends, "send_outreach_email", _fake)

    tpl = EmailTemplate(id=uuid.uuid4(), name="Later", subject="s", body="b")
    db_session.add(tpl)
    await db_session.flush()
    db_session.add(EmailTemplateAttachment(
        template_id=tpl.id, filename="deck.pdf", content_type="application/pdf",
        size_bytes=len(PDF), content=PDF,
    ))
    now = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)
    db_session.add(EmailSend(
        id=uuid.uuid4(), template_id=tpl.id, to_email="p@prospect.com", subject="s",
        status="scheduled", scheduled_at=now - timedelta(minutes=1), body_html="<p>Hi</p>",
    ))
    await db_session.commit()

    await scheduled_sends.send_due_scheduled_for_session(db_session, now=now)

    [call] = calls
    assert [(a.filename, a.data) for a in call["attachments"]] == [("deck.pdf", PDF)]


@pytest.mark.asyncio
async def test_the_dispatcher_hands_attachments_to_the_transport(
    db_session: AsyncSession, monkeypatch
):
    seen = []

    class _Spy:
        def send_email(self, **kwargs):
            seen.append(kwargs)
            return {"status": "sent"}

    account = SimpleNamespace(id=uuid.uuid4(), email="rep@apprend.tech", status="connected")

    async def _allowed(*a, **k):
        return None

    async def _account(db, account_id):
        return account

    async def _claim(db, acct, **k):
        return email_dispatch.VolumeSlot(account=acct)

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(email_dispatch, "check_send_allowed", _allowed)
    monkeypatch.setattr(email_dispatch, "get_account", _account)
    monkeypatch.setattr(email_dispatch, "claim_send_volume", _claim)
    monkeypatch.setattr(email_dispatch, "settle_send_volume", _noop)
    monkeypatch.setattr(email_dispatch, "record_account_send", _noop)
    monkeypatch.setattr(email_dispatch, "_make_sender", lambda a: _Spy())

    files = [ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF)]
    await email_dispatch.send_outreach_email(
        db_session, to_email="p@prospect.com", subject="s", body_html="<p>b</p>",
        sender_account_id=account.id, skip_spam_scan=True, attachments=files,
    )
    await email_dispatch.send_outreach_email(
        db_session, to_email="p@prospect.com", subject="s", body_html="<p>b</p>",
        sender_account_id=account.id, skip_spam_scan=True,
    )

    assert seen[0]["attachments"] == files
    # A plain email calls the transport exactly as it always did.
    assert "attachments" not in seen[1]


# --- Wire format -------------------------------------------------------------


def _parts(message):
    return [p.get_content_type() for p in message.get_payload()]


def test_no_attachments_keeps_the_plain_alternative_body():
    msg = ta.build_mime_body("text", "<p>html</p>")
    assert msg.get_content_type() == "multipart/alternative"
    assert _parts(msg) == ["text/plain", "text/html"]


def test_no_html_sends_a_single_plain_text_part():
    msg = ta.build_mime_body("Hello there", "")
    assert msg.get_content_type() == "text/plain"
    assert msg.get_payload(decode=True).decode() == "Hello there"

    with_file = ta.build_mime_body("Hello", None, [ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF)])
    assert _parts(with_file) == ["text/plain", "application/pdf"]


def test_attachments_wrap_the_body_in_multipart_mixed():
    files = [
        ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF),
        ta.OutgoingAttachment("Résumé – 2026.txt", "text/plain", b"caf\xc3\xa9"),
    ]
    raw = ta.build_mime_body("text", "<p>html</p>", files).as_bytes()
    msg = email.message_from_bytes(raw)

    assert msg.get_content_type() == "multipart/mixed"
    body, pdf, txt = msg.get_payload()
    assert body.get_content_type() == "multipart/alternative"
    assert pdf.get_content_type() == "application/pdf"
    assert pdf.get_content_disposition() == "attachment"
    assert pdf.get_filename() == "deck.pdf"
    assert pdf.get_payload(decode=True) == PDF
    # Non-ASCII names survive the header encoding.
    assert txt.get_filename() == "Résumé – 2026.txt"
    assert txt.get_payload(decode=True) == b"caf\xc3\xa9"


def _account():
    return SimpleNamespace(email="rep@apprend.tech", display_name="Rep")


def test_gmail_message_carries_the_attachment_and_its_headers():
    files = [ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF)]
    body = GmailSender(_account())._build_mime(
        "p@prospect.com", "Our deck", "<p>Hi</p>", "Hi", None,
        list_unsubscribe_url="https://example.test/u", attachments=files,
    )
    msg = email.message_from_bytes(base64.urlsafe_b64decode(body["raw"]))
    assert msg.get_content_type() == "multipart/mixed"
    assert msg["To"] == "p@prospect.com"
    assert msg["List-Unsubscribe"]
    assert msg.get_payload()[1].get_filename() == "deck.pdf"


def test_gmail_threaded_reply_carries_the_attachment():
    files = [ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF)]
    body = GmailSender(_account())._build_mime_threaded(
        "p@prospect.com", "Re: Our deck", "<p>Hi</p>", "Hi", None,
        in_reply_to="<a@b>", references="<a@b>", attachments=files,
    )
    msg = email.message_from_bytes(base64.urlsafe_b64decode(body["raw"]))
    assert msg["In-Reply-To"] == "<a@b>"
    assert msg.get_payload()[1].get_filename() == "deck.pdf"


def test_smtp_message_carries_the_attachment():
    files = [ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF)]
    msg = SmtpSender(_account())._build_message(
        "p@prospect.com", "Our deck", "<p>Hi</p>", "Hi", None, attachments=files,
    )
    parsed = email.message_from_string(msg.as_string())
    assert parsed.get_content_type() == "multipart/mixed"
    assert parsed.get_payload()[1].get_payload(decode=True) == PDF


class _FakeSes:
    def __init__(self, raw_error=None):
        self.raw = []
        self.structured = []
        self.raw_error = raw_error

    def send_raw_email(self, **kwargs):
        if self.raw_error:
            raise self.raw_error
        self.raw.append(kwargs)
        return {"MessageId": "ses-1"}

    def send_email(self, **kwargs):
        self.structured.append(kwargs)
        return {"MessageId": "ses-2"}


@pytest.fixture
def ses(monkeypatch):
    monkeypatch.setattr(ses_mod.settings, "aws_access_key_id", "AKIA-test")
    monkeypatch.setattr(ses_mod.settings, "aws_secret_access_key", "secret")
    sender = ses_mod.EmailSender()
    return sender


def test_ses_sends_attachments_raw_even_without_an_unsubscribe_url(ses):
    fake = _FakeSes()
    ses._client = fake
    files = [ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF)]

    result = ses.send_email(
        to_email="p@prospect.com", subject="s", body_html="<p>b</p>",
        include_unsubscribe=False, attachments=files,
    )
    assert result["status"] == "sent"
    assert fake.structured == []
    msg = email.message_from_string(fake.raw[0]["RawMessage"]["Data"])
    assert msg.get_payload()[1].get_filename() == "deck.pdf"


def test_ses_does_not_drop_attachments_when_raw_is_not_permitted(ses):
    denied = Exception("denied")
    denied.response = {"Error": {"Code": "AccessDenied", "Message": "no raw"}}
    fake = _FakeSes(raw_error=denied)
    ses._client = fake
    files = [ta.OutgoingAttachment("deck.pdf", "application/pdf", PDF)]

    result = ses.send_email(
        to_email="p@prospect.com", subject="s", body_html="<p>b</p>",
        include_unsubscribe=False, list_unsubscribe_url="https://example.test/u",
        attachments=files,
    )
    # The structured fallback would send the email without its file.
    assert result["status"] == "error"
    assert fake.structured == []


# --- Helpers -----------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("deck.pdf", "deck.pdf"),
    ("evil\r\nBcc: x@y.z.pdf", "evilBcc: x@y.z.pdf"),
    ("tab\there\x00.pdf", "tabhere.pdf"),
    ("C:\\Users\\me\\deck.pdf", "deck.pdf"),
    ("", "attachment"),
    (None, "attachment"),
    ("...", "attachment"),
    ("a" * 300 + ".pdf", "a" * 251 + ".pdf"),
])
def test_sanitize_filename(raw, expected):
    assert ta.sanitize_filename(raw) == expected
