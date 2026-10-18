"""Campaign emails carry the campaign's documents, and can be drafted from email templates.

The draft to Fedir at Gart Solutions went out (to review) without the
SteadyState_Partner_Brief.pdf the campaign was built from: ``attach_documents``
defaulted off and nothing in the UI could turn it on. And drafts couldn't come
from — or become — the email templates the rest of the app uses.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from app.config import get_settings
from app.models.campaign import Campaign, CampaignProspect
from app.models.company import Company
from app.models.contact import Contact
from app.models.discovery_document import STATUS_READY, DiscoveryDocument
from app.models.email_template import EmailTemplate, EmailTemplateAttachment
from app.models.gmail_account import GmailAccount
from app.models.org_llm_key import OrgLlmKey
from app.models.user import User
from app.org_context import APPREND_ORG_ID
from app.services import campaigns as svc
from app.services import crypto, discovery_documents, email_dispatch, org_gemini

MIDWEEK = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc)
PDF = b"%PDF-1.4 brief"


@pytest.fixture(autouse=True)
def _crypto_key():
    settings = get_settings()
    prev = settings.token_encryption_key
    settings.token_encryption_key = Fernet.generate_key().decode()
    crypto._fernet.cache_clear()
    yield
    settings.token_encryption_key = prev
    crypto._fernet.cache_clear()


def _gmail(**kw):
    defaults = dict(id=uuid.uuid4(), email=f"sender-{uuid.uuid4().hex[:6]}@example.com",
                    google_sub=uuid.uuid4().hex, refresh_token_encrypted="x",
                    status="connected", sending_enabled=True, is_active=True)
    defaults.update(kw)
    return GmailAccount(**defaults)


async def _doc(db, filename="Partner_Brief.pdf", file_type="pdf", content=PDF, **kw):
    d = DiscoveryDocument(filename=filename, file_type=file_type, size_bytes=len(content or b""),
                          status=STATUS_READY, content=content, extracted={"candidates": []}, **kw)
    db.add(d)
    await db.commit()
    return d


async def _campaign(db, **kw):
    acct = _gmail()
    db.add(acct)
    await db.flush()
    c = Campaign(name="T", status="active", document_ids=[], icp_json={"summary": "Incident remediation"},
                 gmail_account_id=acct.id, daily_send_limit=20, send_window_start="09:00",
                 send_window_end="18:00", timezone="America/Toronto", send_on_weekends=False,
                 review_before_send=True, attach_documents=False, warmup_day=1, template_mode="ai")
    for k, v in kw.items():
        setattr(c, k, v)
    db.add(c)
    await db.commit()
    return c


async def _template(db, name="Intro", subject="{{first_name}}, quick one about {{company}}",
                    body="<p>Hi {{first_name}},</p><p>Saw {{company}} is scaling SRE.<br>Worth a chat?</p>"):
    t = EmailTemplate(name=name, subject=subject, body=body)
    db.add(t)
    await db.commit()
    return t


async def _prospect(db, c, *, status="queued", first="Fedir", last="Kompaniiets", title="CEO",
                    company="Gart Solutions", domain="gartsolutions.com", **kw):
    co = Company(name=company, domain=domain, source="tavily+gemini")
    db.add(co)
    await db.flush()
    ct = Contact(company_id=co.id, first_name=first, last_name=last, title=title,
                 email=f"{first.lower()}@{domain}", email_verification_result="deliverable",
                 email_verified_at=MIDWEEK)
    db.add(ct)
    await db.flush()
    p = CampaignProspect(campaign_id=c.id, company_id=co.id, contact_id=ct.id, to_email=ct.email,
                         status=status, **kw)
    db.add(p)
    await db.commit()
    return p


# ── attachments ──────────────────────────────────────────────────────────────

async def _create(client, db_session, monkeypatch, docs, **body):
    from app.tasks import campaigns as campaign_tasks

    admin = (await db_session.execute(select(User.id).where(
        User.email == "_default_test_admin@landline.app"))).scalar_one()
    db_session.add_all([_gmail(user_id=admin), OrgLlmKey(org_id=APPREND_ORG_ID, provider="gemini",
                                                         key_encrypted=crypto.encrypt("k"), key_hint="••••")])
    await db_session.commit()

    async def fake_icp(db, org_id, d):
        return svc.Icp(summary="x"), "x"

    monkeypatch.setattr(svc, "build_icp", fake_icp)
    monkeypatch.setattr(campaign_tasks.run_campaign_now, "delay", lambda *a: None)
    return await client.post("/api/v1/campaigns", json={
        "name": "C", "document_ids": [str(d.id) for d in docs], **body})


@pytest.mark.asyncio
async def test_a_campaign_built_from_a_pdf_attaches_it_by_default(client, db_session, monkeypatch):
    r = await _create(client, db_session, monkeypatch, [await _doc(db_session)])
    assert r.status_code == 201, r.text
    assert r.json()["attach_documents"] is True
    detail = (await client.get(f"/api/v1/campaigns/{r.json()['id']}")).json()
    assert detail["attachments"] == ["Partner_Brief.pdf"]


@pytest.mark.asyncio
async def test_a_spreadsheet_is_not_attached(client, db_session, monkeypatch):
    csv = await _doc(db_session, filename="leads.csv", file_type="csv", content=b"a,b")
    r = await _create(client, db_session, monkeypatch, [csv])
    assert r.json()["attach_documents"] is False


@pytest.mark.asyncio
async def test_attaching_can_be_turned_off_at_creation(client, db_session, monkeypatch):
    r = await _create(client, db_session, monkeypatch, [await _doc(db_session)], attach_documents=False)
    assert r.json()["attach_documents"] is False


@pytest.mark.asyncio
async def test_documents_and_template_files_both_ride_along_once_each(db_session):
    doc = await _doc(db_session)
    t = await _template(db_session)
    db_session.add_all([
        EmailTemplateAttachment(template_id=t.id, filename="pricing.pdf", content_type="application/pdf",
                                content=b"%PDF p", size_bytes=6),
        EmailTemplateAttachment(template_id=t.id, filename="Partner_Brief.pdf", content_type="application/pdf",
                                content=b"%PDF dup", size_bytes=8),
    ])
    c = await _campaign(db_session, document_ids=[str(doc.id)], attach_documents=True,
                        template_id=t.id, template_mode="template")
    files = await svc._attachments(db_session, c)
    assert [a.filename for a in files] == ["Partner_Brief.pdf", "pricing.pdf"]
    assert files[0].data == PDF   # the campaign's own document wins the duplicate name
    assert await svc.attachment_names(db_session, c) == ["Partner_Brief.pdf", "pricing.pdf"]


@pytest.mark.asyncio
async def test_the_send_carries_the_attachments(db_session, monkeypatch):
    doc = await _doc(db_session)
    c = await _campaign(db_session, document_ids=[str(doc.id)], attach_documents=True)
    p = await _prospect(db_session, c, status="approved", email_subject="Hi", email_body="Hello")
    seen = {}

    async def fake_send(db, **kw):
        seen.update(kw)
        return {"status": "sent", "message_id": "m"}

    monkeypatch.setattr(email_dispatch, "send_outreach_email", fake_send)
    assert (await svc.send_prospect(db_session, p.id, MIDWEEK)).status == "sent"
    assert [a.filename for a in seen["attachments"]] == ["Partner_Brief.pdf"]


@pytest.mark.asyncio
async def test_a_live_campaigns_attached_document_survives_the_30_day_purge(db_session):
    old = datetime.now(timezone.utc) - timedelta(days=45)
    kept = await _doc(db_session, filename="kept.pdf", created_at=old)
    gone = await _doc(db_session, filename="gone.pdf", created_at=old)
    stopped = await _doc(db_session, filename="stopped.pdf", created_at=old)
    await _campaign(db_session, document_ids=[str(kept.id)], attach_documents=True)
    await _campaign(db_session, document_ids=[str(gone.id)], attach_documents=False)
    await _campaign(db_session, document_ids=[str(stopped.id)], attach_documents=True, status="stopped")
    assert await discovery_documents.purge_expired_files(db_session) == 2
    await db_session.commit()
    content = dict((await db_session.execute(
        select(DiscoveryDocument.filename, DiscoveryDocument.content)
        .execution_options(populate_existing=True))).all())
    assert content == {"kept.pdf": PDF, "gone.pdf": None, "stopped.pdf": None}


# ── drafting from a template ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_template_mode_fills_the_template_without_calling_the_ai(db_session, monkeypatch):
    t = await _template(db_session)
    c = await _campaign(db_session, template_id=t.id, template_mode="template", review_before_send=False)
    p = await _prospect(db_session, c)

    async def no_ai(*a, **kw):
        raise AssertionError("template mode must not call the AI")

    monkeypatch.setattr(org_gemini, "generate_json", no_ai)
    assert await svc.generate_drafts(db_session, c, MIDWEEK) == 1
    await db_session.refresh(p)
    assert p.email_subject == "Fedir, quick one about Gart Solutions"
    assert p.email_body == "Hi Fedir,\n\nSaw Gart Solutions is scaling SRE.\nWorth a chat?"
    assert p.status == "approved"   # review off and nothing left unfilled


@pytest.mark.asyncio
async def test_a_template_with_a_bracket_placeholder_still_goes_to_review(db_session):
    t = await _template(db_session, body="<p>Hi {{first_name}}, [Insert case study]</p>")
    c = await _campaign(db_session, template_id=t.id, template_mode="template", review_before_send=False)
    p = await _prospect(db_session, c)
    await svc.generate_drafts(db_session, c, MIDWEEK)
    await db_session.refresh(p)
    assert p.status == "draft_created"


@pytest.mark.asyncio
async def test_ai_template_mode_gives_the_ai_the_template_to_personalise(db_session, monkeypatch):
    t = await _template(db_session)
    c = await _campaign(db_session, template_id=t.id, template_mode="ai_template")
    await _prospect(db_session, c)
    prompts = []

    async def fake_json(db, org_id, *, prompt, schema, **kw):
        prompts.append(prompt)
        return schema(subject="S", body="B")

    monkeypatch.setattr(org_gemini, "generate_json", fake_json)
    await svc.generate_drafts(db_session, c, MIDWEEK)
    assert "Base the email on this template" in prompts[0]
    assert "Saw {{company}} is scaling SRE." in prompts[0]


@pytest.mark.asyncio
async def test_an_archived_template_falls_back_to_ai_drafting(db_session, monkeypatch):
    t = await _template(db_session)
    t.is_archived = True
    c = await _campaign(db_session, template_id=t.id, template_mode="template")
    await _prospect(db_session, c)
    prompts = []

    async def fake_json(db, org_id, *, prompt, schema, **kw):
        prompts.append(prompt)
        return schema(subject="S", body="B")

    monkeypatch.setattr(org_gemini, "generate_json", fake_json)
    await svc.generate_drafts(db_session, c, MIDWEEK)
    assert len(prompts) == 1 and "Base the email on this template" not in prompts[0]


@pytest.mark.asyncio
async def test_choosing_a_template_on_the_campaign(client, db_session):
    t = await _template(db_session)
    c = await _campaign(db_session)
    url = f"/api/v1/campaigns/{c.id}"
    r = await client.patch(url, json={"template_id": str(t.id), "template_mode": "ai_template"})
    assert r.status_code == 200 and r.json()["template_mode"] == "ai_template"
    assert (await client.patch(url, json={"template_id": str(uuid.uuid4())})).status_code == 404
    # A template mode with no template is refused, not silently ignored.
    r = await client.patch(url, json={"template_id": None, "template_mode": "template"})
    assert r.status_code == 422
    r = await client.patch(url, json={"template_id": None, "template_mode": "ai"})
    assert r.status_code == 200 and r.json()["template_id"] is None


def test_html_to_text_keeps_paragraphs_and_line_breaks():
    assert svc.html_to_text("<p>Hi &amp; hello,</p><p>Line one<br/>line two</p>") == \
        "Hi & hello,\n\nLine one\nline two"


# ── saving a draft as a template ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_draft_saved_as_a_template_turns_the_prospect_into_merge_fields(client, db_session):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c, status="draft_created",
                        email_subject="Gart Solutions and SRE automation",
                        email_body="Hi Fedir,\n\nAs CEO of Gart Solutions you know on-call.\n\nBest, Subhendu")
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/save-template",
                          json={"name": "  SRE intro  ", "use_for_campaign": "template"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "SRE intro"
    assert body["subject"] == "{{company}} and SRE automation"
    assert "Hi {{first_name}}," in body["body"]
    assert "As {{contact_title}} of {{company}} you know on-call." in body["body"]
    assert body["campaign_template_mode"] == "template"
    t = (await db_session.execute(select(EmailTemplate).where(EmailTemplate.name == "SRE intro"))).scalars().one()
    await db_session.refresh(c)
    assert c.template_id == t.id
    # The saved template renders back into the original for that prospect.
    contact = await db_session.get(Contact, p.contact_id)
    company = await db_session.get(Company, p.company_id)
    subject, text = svc.render_template_draft(t, contact, company)
    assert subject == "Gart Solutions and SRE automation"
    assert text.startswith("Hi Fedir,")


@pytest.mark.asyncio
async def test_saving_refuses_a_duplicate_name_and_an_empty_draft(client, db_session):
    await _template(db_session, name="Taken")
    c = await _campaign(db_session)
    drafted = await _prospect(db_session, c, status="draft_created", email_subject="S", email_body="B")
    empty = await _prospect(db_session, c, first="Ana", domain="other.com", company="Other")
    url = f"/api/v1/campaigns/{c.id}/prospects"
    assert (await client.post(f"{url}/{drafted.id}/save-template", json={"name": "taken"})).status_code == 409
    assert (await client.post(f"{url}/{empty.id}/save-template", json={"name": "New"})).status_code == 409


def test_generalising_matches_whole_words_only():
    contact = Contact(first_name="Ana", last_name="Li", title=None)
    company = Company(name="Analytics Co")
    assert svc.generalize_draft("Hi Ana, Analytics Co loves analytics.", contact, company) == \
        "Hi {{first_name}}, {{company}} loves analytics."


# ── the preview can open what the email carries ──────────────────────────────

@pytest.mark.asyncio
async def test_the_preview_can_fetch_an_attached_pdf(client, db_session):
    doc = await _doc(db_session, filename="SteadyState Brief.pdf")
    c = await _campaign(db_session, document_ids=[str(doc.id)], attach_documents=True)
    r = await client.get(f"/api/v1/campaigns/{c.id}/attachments/file", params={"name": "SteadyState Brief.pdf"})
    assert r.status_code == 200
    assert r.content == PDF
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline;")
    assert "SteadyState%20Brief.pdf" in r.headers["content-disposition"]
    assert r.headers["cache-control"] == "private, no-store"


@pytest.mark.asyncio
async def test_only_files_the_email_actually_carries_can_be_fetched(client, db_session):
    doc = await _doc(db_session)
    off = await _campaign(db_session, document_ids=[str(doc.id)], attach_documents=False)
    url = f"/api/v1/campaigns/{off.id}/attachments/file"
    # Attaching is off: the document exists but the email won't carry it.
    assert (await client.get(url, params={"name": "Partner_Brief.pdf"})).status_code == 404
    on = await _campaign(db_session, document_ids=[str(doc.id)], attach_documents=True)
    url = f"/api/v1/campaigns/{on.id}/attachments/file"
    assert (await client.get(url, params={"name": "other.pdf"})).status_code == 404
    assert (await client.get(f"/api/v1/campaigns/{uuid.uuid4()}/attachments/file",
                             params={"name": "Partner_Brief.pdf"})).status_code == 404


@pytest.mark.asyncio
async def test_a_word_attachment_downloads_rather_than_renders(client, db_session):
    doc = await _doc(db_session, filename="brief.docx", file_type="docx", content=b"PK\x03\x04docx")
    c = await _campaign(db_session, document_ids=[str(doc.id)], attach_documents=True)
    r = await client.get(f"/api/v1/campaigns/{c.id}/attachments/file", params={"name": "brief.docx"})
    assert r.status_code == 200 and r.headers["content-disposition"].startswith("attachment;")


# ── switching templates on one draft ─────────────────────────────────────────
# The draft preview couldn't switch templates, didn't say which template a
# draft came from, and so never showed that template's merge fields.

@pytest.mark.asyncio
async def test_drafting_from_a_template_records_it_on_the_draft(db_session):
    t = await _template(db_session)
    c = await _campaign(db_session, template_id=t.id, template_mode="template")
    p = await _prospect(db_session, c)
    await svc.generate_drafts(db_session, c, MIDWEEK)
    await db_session.refresh(p)
    assert p.template_id == t.id
    svc.reset_for_redraft(p)
    assert p.template_id is None


@pytest.mark.asyncio
async def test_one_draft_can_switch_to_another_template(client, db_session, monkeypatch):
    first = await _template(db_session)
    other = await _template(db_session, name="Follow-up", subject="Still on {{company}}'s radar?",
                            body="<p>Hi {{first_name}}, following up.</p>")
    c = await _campaign(db_session, template_id=first.id, template_mode="template")
    p = await _prospect(db_session, c, status="approved", email_subject="Old", email_body="Old",
                        template_id=first.id, approved_at=MIDWEEK)

    async def no_ai(*a, **kw):
        raise AssertionError("template mode must not call the AI")

    monkeypatch.setattr(org_gemini, "generate_json", no_ai)
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/redraft",
                          json={"template_id": str(other.id), "template_mode": "template"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["email_subject"] == "Still on Gart Solutions's radar?"
    assert body["email_body"] == "Hi Fedir, following up."
    assert body["template_id"] == str(other.id) and body["template_name"] == "Follow-up"
    # New copy goes back to review; the campaign keeps its own template.
    assert body["status"] == "draft_created"
    await db_session.refresh(c)
    assert c.template_id == first.id

    listed = (await client.get(f"/api/v1/campaigns/{c.id}/prospects")).json()["items"][0]
    assert listed["template_id"] == str(other.id) and listed["template_name"] == "Follow-up"


@pytest.mark.asyncio
async def test_a_draft_can_go_back_to_the_ai(client, db_session, monkeypatch):
    t = await _template(db_session)
    c = await _campaign(db_session)
    p = await _prospect(db_session, c, status="draft_created", email_subject="S", email_body="B",
                        template_id=t.id)
    prompts = []

    async def fake_json(db, org_id, *, prompt, schema, **kw):
        prompts.append(prompt)
        return schema(subject="AI subject", body="AI body")

    monkeypatch.setattr(org_gemini, "generate_json", fake_json)
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/redraft", json={"template_id": None})
    assert r.status_code == 200, r.text
    assert r.json()["email_subject"] == "AI subject" and r.json()["template_id"] is None
    assert "Base the email on this template" not in prompts[0]


@pytest.mark.asyncio
async def test_redrafting_one_refuses_sent_mail_and_unknown_templates(client, db_session):
    t = await _template(db_session)
    c = await _campaign(db_session)
    sent = await _prospect(db_session, c, status="sent", email_subject="S", email_body="B")
    draft = await _prospect(db_session, c, status="draft_created", email_subject="S", email_body="B",
                            first="Ana", domain="other.com", company="Other")
    url = f"/api/v1/campaigns/{c.id}/prospects"
    assert (await client.post(f"{url}/{sent.id}/redraft", json={"template_id": str(t.id)})).status_code == 409
    r = await client.post(f"{url}/{draft.id}/redraft", json={"template_id": str(uuid.uuid4())})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_a_draft_saved_as_a_template_is_linked_to_it(client, db_session):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c, status="draft_created", email_subject="Hi Fedir", email_body="Hello Fedir")
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/save-template", json={"name": "Mine"})
    assert r.status_code == 201
    await db_session.refresh(p)
    assert str(p.template_id) == r.json()["id"]
    listed = (await client.get("/api/v1/email-templates")).json()
    assert any(t["id"] == r.json()["id"] and "{{first_name}}" in t["body"] for t in listed)


# ── the campaign's default template ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_new_campaign_drafts_from_the_orgs_default_template(client, db_session, monkeypatch):
    t = await _template(db_session)
    t.is_default = True
    await db_session.commit()
    r = await _create(client, db_session, monkeypatch, [await _doc(db_session)])
    assert r.status_code == 201, r.text
    assert r.json()["template_id"] == str(t.id)
    assert r.json()["template_mode"] == "ai_template"


@pytest.mark.asyncio
async def test_a_template_chosen_at_creation_beats_the_default(client, db_session, monkeypatch):
    default = await _template(db_session, name="Default")
    default.is_default = True
    chosen = await _template(db_session, name="Chosen")
    await db_session.commit()
    r = await _create(client, db_session, monkeypatch, [await _doc(db_session)],
                      template_id=str(chosen.id), template_mode="template")
    assert (r.json()["template_id"], r.json()["template_mode"]) == (str(chosen.id), "template")
    # Choosing "the AI writes them" at creation is honoured too.
    r = await _create(client, db_session, monkeypatch, [await _doc(db_session)],
                      template_id=None, template_mode="ai")
    assert (r.json()["template_id"], r.json()["template_mode"]) == (None, "ai")


@pytest.mark.asyncio
async def test_without_a_default_template_a_new_campaign_is_ai_written(client, db_session, monkeypatch):
    r = await _create(client, db_session, monkeypatch, [await _doc(db_session)])
    assert (r.json()["template_id"], r.json()["template_mode"]) == (None, "ai")


@pytest.mark.asyncio
async def test_an_archived_default_template_is_not_used(client, db_session, monkeypatch):
    t = await _template(db_session)
    t.is_default, t.is_archived = True, True
    await db_session.commit()
    r = await _create(client, db_session, monkeypatch, [await _doc(db_session)])
    assert r.json()["template_id"] is None and r.json()["template_mode"] == "ai"


@pytest.mark.asyncio
async def test_picking_a_template_alone_switches_drafting_to_it(client, db_session, monkeypatch):
    t = await _template(db_session)
    c = await _campaign(db_session)
    url = f"/api/v1/campaigns/{c.id}"
    r = await client.patch(url, json={"template_id": str(t.id)})
    assert (r.json()["template_id"], r.json()["template_mode"]) == (str(t.id), "ai_template")
    # A mode already chosen is kept when only the template changes.
    await client.patch(url, json={"template_mode": "template"})
    other = await _template(db_session, name="Other")
    r = await client.patch(url, json={"template_id": str(other.id)})
    assert r.json()["template_mode"] == "template"
    # Clearing the template hands drafting back to the AI rather than a 422.
    r = await client.patch(url, json={"template_id": None})
    assert r.status_code == 200
    assert (r.json()["template_id"], r.json()["template_mode"]) == (None, "ai")

    # New drafts follow the campaign's template.
    await client.patch(url, json={"template_id": str(t.id), "template_mode": "template"})
    await db_session.refresh(c)
    assert (await svc.campaign_template(db_session, c)).id == t.id
