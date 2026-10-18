"""The draft preview's sender, re-drafting after a template change, and editing contacts.

* The preview shows which mailbox an email goes (or went) out from.
* Choosing a template for a running campaign only helps if its existing drafts
  are rewritten — ``POST /campaigns/{id}/redraft``.
* Discovery found companies with nobody to email, and people with the wrong
  details; ``PUT …/prospects/{pid}/contact`` adds or edits the person, and a
  changed address is verified and gated again before anything can be sent.
"""

import uuid
from datetime import datetime, timezone

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from app.config import get_settings
from app.models.campaign import Campaign, CampaignProspect
from app.models.company import Company
from app.models.contact import Contact
from app.models.email_template import EmailSend
from app.models.gmail_account import GmailAccount
from app.models.smtp_account import SmtpAccount
from app.services import crypto

MIDWEEK = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _crypto_key():
    settings = get_settings()
    prev = settings.token_encryption_key
    settings.token_encryption_key = Fernet.generate_key().decode()
    crypto._fernet.cache_clear()
    yield
    settings.token_encryption_key = prev
    crypto._fernet.cache_clear()


@pytest.fixture
def dispatched(monkeypatch):
    from app.tasks import campaigns as campaign_tasks

    calls = []
    monkeypatch.setattr(campaign_tasks.process_new_prospects, "delay", lambda *a: calls.append(a))
    return calls


async def _campaign(db, *, account=None, **kw):
    if account is None:
        account = GmailAccount(id=uuid.uuid4(), email="me@sender.io", display_name="Subhendu Das",
                               google_sub=uuid.uuid4().hex, refresh_token_encrypted="x",
                               status="connected", sending_enabled=True, is_active=True)
    db.add(account)
    await db.flush()
    fields = dict(name="T", status="active", document_ids=[], icp_json={"summary": "x"},
                  daily_send_limit=20, send_window_start="09:00", send_window_end="18:00",
                  timezone="America/Toronto", send_on_weekends=False, review_before_send=True,
                  attach_documents=False, warmup_day=1, template_mode="ai")
    if isinstance(account, SmtpAccount):
        fields["smtp_account_id"] = account.id
    else:
        fields["gmail_account_id"] = account.id
    fields.update(kw)
    c = Campaign(**fields)
    db.add(c)
    await db.commit()
    return c


async def _prospect(db, c, *, status="draft_created", email="fedir@gart.io", first="Fedir",
                    domain="gart.io", **kw):
    co = (await db.execute(select(Company).where(Company.domain == domain))).scalars().first()
    if co is None:
        co = Company(name=f"Co {domain}", domain=domain, source="tavily+gemini")
        db.add(co)
        await db.flush()
    ct = None
    if first or email:
        ct = Contact(company_id=co.id, first_name=first, last_name="K", title="CEO", email=email,
                     email_verification_result="deliverable" if email else None,
                     email_verified_at=MIDWEEK if email else None)
        db.add(ct)
        await db.flush()
    p = CampaignProspect(campaign_id=c.id, company_id=co.id, contact_id=ct.id if ct else None,
                         to_email=email, status=status,
                         email_subject="Hi" if status in ("draft_created", "approved") else None,
                         email_body="Hello Fedir" if status in ("draft_created", "approved") else None, **kw)
    db.add(p)
    await db.commit()
    return p


# ── the sender ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_campaign_reports_which_mailbox_its_drafts_go_out_from(client, db_session):
    c = await _campaign(db_session)
    sender = (await client.get(f"/api/v1/campaigns/{c.id}")).json()["sender"]
    assert sender["email"] == "me@sender.io" and sender["kind"] == "gmail"
    assert sender["display_name"] == "Subhendu Das" and sender["ready"] is True


@pytest.mark.asyncio
async def test_a_mailbox_that_cannot_send_is_named_with_the_reason(client, db_session):
    smtp = SmtpAccount(id=uuid.uuid4(), email="out@mailer.io", host="h", username="u",
                       password_encrypted="x", status="error", sending_enabled=True)
    c = await _campaign(db_session, account=smtp)
    sender = (await client.get(f"/api/v1/campaigns/{c.id}")).json()["sender"]
    assert (sender["email"], sender["kind"], sender["ready"], sender["problem"]) == \
        ("out@mailer.io", "smtp", False, "disconnected")


@pytest.mark.asyncio
async def test_a_sent_email_shows_the_address_it_actually_went_from(client, db_session):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c, status="sent")
    send = EmailSend(to_email=p.to_email, subject="Hi", body_html="<p>x</p>", status="sent",
                     sent_at=MIDWEEK, from_email="old-mailbox@sender.io")
    db_session.add(send)
    await db_session.flush()
    p.email_send_id = send.id
    await db_session.commit()
    item = (await client.get(f"/api/v1/campaigns/{c.id}/prospects")).json()["items"][0]
    assert item["sent_from"] == "old-mailbox@sender.io"
    assert item["company_domain"] == "gart.io" and item["contact_first_name"] == "Fedir"


# ── re-drafting after a template change ─────────────────────────────────────

@pytest.mark.asyncio
async def test_redraft_rewrites_unapproved_drafts_and_leaves_approved_ones(client, db_session, dispatched):
    c = await _campaign(db_session)
    draft = await _prospect(db_session, c, status="draft_created")
    approved = await _prospect(db_session, c, status="approved", email="ana@other.io", first="Ana",
                               domain="other.io", approved_at=MIDWEEK)
    r = await client.post(f"/api/v1/campaigns/{c.id}/redraft")
    assert r.json() == {"redrafting": 1, "processing": "dispatched"}
    await db_session.refresh(draft)
    await db_session.refresh(approved)
    assert (draft.status, draft.email_subject) == ("queued", None)
    assert (approved.status, approved.email_subject) == ("approved", "Hi")
    assert dispatched


# ── editing and adding contacts ──────────────────────────────────────────────

URL = "/api/v1/campaigns/{c}/prospects/{p}/contact"


@pytest.mark.asyncio
async def test_a_changed_email_is_verified_again_and_redrafted(client, db_session, dispatched):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c)
    r = await client.put(URL.format(c=c.id, p=p.id), json={
        "first_name": "Fedir", "last_name": "Kompaniiets", "title": "Co-founder", "email": "F.K@Gart.io"})
    assert r.status_code == 200, r.text
    assert r.json()["queued"] == 1 and dispatched
    await db_session.refresh(p)
    ct = await db_session.get(Contact, p.contact_id)
    assert (p.to_email, p.status, p.email_body) == ("f.k@gart.io", "queued", None)
    # A verdict is about an address: the new one hasn't been checked.
    assert ct.email_verification_result is None and ct.email_verified_at is None
    assert (ct.last_name, ct.title) == ("Kompaniiets", "Co-founder")


@pytest.mark.asyncio
async def test_a_title_only_edit_keeps_the_draft(client, db_session, dispatched):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c)
    r = await client.put(URL.format(c=c.id, p=p.id), json={
        "first_name": "Fedir", "last_name": "K", "title": "Founder", "email": "fedir@gart.io"})
    assert r.json()["queued"] == 0 and dispatched == []
    await db_session.refresh(p)
    assert (p.status, p.email_body) == ("draft_created", "Hello Fedir")
    assert (await db_session.get(Contact, p.contact_id)).email_verification_result == "deliverable"


@pytest.mark.asyncio
async def test_adding_a_contact_where_discovery_found_nobody(client, db_session, dispatched):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c, status="no_email", email=None, first=None, domain="obsium.io",
                        skip_reason="No email address found for this company yet")
    r = await client.put(URL.format(c=c.id, p=p.id), json={
        "first_name": "Rahul", "title": "Head of Platform", "email": "rahul@obsium.io"})
    assert r.status_code == 200 and r.json()["queued"] == 1
    await db_session.refresh(p)
    ct = await db_session.get(Contact, p.contact_id)
    assert (ct.first_name, ct.company_id, ct.email) == ("Rahul", p.company_id, "rahul@obsium.io")
    assert (p.status, p.to_email, p.skip_reason) == ("queued", "rahul@obsium.io", None)


@pytest.mark.asyncio
async def test_an_edited_address_still_has_to_clear_the_do_not_contact_rules(client, db_session, dispatched):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c)
    r = await client.put(URL.format(c=c.id, p=p.id), json={"first_name": "Fedir", "email": "boss@sender.io"})
    body = r.json()
    assert body["queued"] == 0 and body["skipped"][0]["reason"] == "own_org" and dispatched == []
    await db_session.refresh(p)
    assert p.status == "skipped" and p.email_body is None


@pytest.mark.asyncio
async def test_an_address_another_contact_holds_is_refused(client, db_session, dispatched):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c)
    await _prospect(db_session, c, email="taken@else.io", first="Other", domain="else.io")
    r = await client.put(URL.format(c=c.id, p=p.id), json={"first_name": "Fedir", "email": "taken@else.io"})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_removing_the_email_leaves_a_no_email_prospect(client, db_session, dispatched):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c)
    r = await client.put(URL.format(c=c.id, p=p.id), json={"first_name": "Fedir", "email": None})
    assert r.json()["skipped"][0]["reason"] == "no_email"
    await db_session.refresh(p)
    assert (p.status, p.to_email) == ("no_email", None)


@pytest.mark.asyncio
async def test_a_sent_prospect_cannot_be_edited(client, db_session, dispatched):
    c = await _campaign(db_session)
    p = await _prospect(db_session, c, status="sent")
    r = await client.put(URL.format(c=c.id, p=p.id), json={"first_name": "X", "email": "x@gart.io"})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_a_company_can_have_several_contacts_in_the_campaign(client, db_session, dispatched):
    """Adding another person at the same company: the manual add merges into the
    company, so both are prospects of it."""
    c = await _campaign(db_session)
    p = await _prospect(db_session, c)
    co = await db_session.get(Company, p.company_id)
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/manual", json={
        "company_name": co.name, "domain": co.domain, "first_name": "Olena", "email": "olena@gart.io"})
    assert r.json()["queued"] == 1
    rows = (await db_session.execute(select(CampaignProspect).where(
        CampaignProspect.campaign_id == c.id))).scalars().all()
    assert {r.company_id for r in rows} == {co.id} and len(rows) == 2
