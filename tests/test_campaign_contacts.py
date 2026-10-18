"""The contact database behind campaigns, adding people to a campaign, and the sender.

* ``GET /contacts/facets`` and the new list filters group the org-wide contact
  table by Hunter verdict, source, industry and campaign.
* ``POST /campaigns/{id}/prospects/add`` (from the database) and ``…/manual``
  (typed in) both say *why* anyone was passed over, and hand the rest to
  verification — nothing is sent until Hunter calls the address deliverable.
* A campaign can send from an SMTP account as well as Gmail (migration 125).
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
from app.models.discovery_document import STATUS_READY, DiscoveryDocument
from app.models.gmail_account import GmailAccount
from app.models.org_llm_key import OrgLlmKey
from app.models.smtp_account import SmtpAccount
from app.models.user import User
from app.org_context import APPREND_ORG_ID
from app.services import campaigns as svc
from app.services import crypto, email_dispatch

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


@pytest.fixture(autouse=True)
def _mx_valid(monkeypatch):
    from app.services import email_validation

    monkeypatch.setattr(email_validation, "check_mx", lambda domain: "valid")


@pytest.fixture
def dispatched(monkeypatch):
    from app.tasks import campaigns as campaign_tasks

    calls = []
    monkeypatch.setattr(campaign_tasks.process_new_prospects, "delay", lambda *a: calls.append(a))
    return calls


class _FakeHunter:
    def __init__(self, finds=None):
        self.enabled = True
        self.finds = finds or {}
        self.find_calls = []

    async def make_budget(self, db):
        return None

    async def find_email(self, db, company, first, last, *, budget=None):
        self.find_calls.append(company.domain)
        return self.finds.get(company.domain)

    async def verify_contact(self, db, contact, *, budget=None):
        contact.email_verification_result = "deliverable"
        contact.email_verified_at = MIDWEEK
        contact.email_verification_meta = {"hunter_result": "deliverable"}
        return {"status": "ok", "result": "deliverable"}


def _gmail(**kw):
    defaults = dict(id=uuid.uuid4(), email=f"sender-{uuid.uuid4().hex[:6]}@example.com",
                    google_sub=uuid.uuid4().hex, refresh_token_encrypted="x",
                    status="connected", sending_enabled=True, is_active=True)
    defaults.update(kw)
    return GmailAccount(**defaults)


def _smtp(**kw):
    defaults = dict(id=uuid.uuid4(), email=f"smtp-{uuid.uuid4().hex[:6]}@mailer.io",
                    host="smtp.mailer.io", username="u", password_encrypted="x",
                    status="connected", sending_enabled=True)
    defaults.update(kw)
    return SmtpAccount(**defaults)


async def _campaign(db, **kw):
    acct = _gmail()
    db.add(acct)
    await db.flush()
    c = Campaign(name="Test", status="active", document_ids=[], icp_json={"summary": "x"},
                 gmail_account_id=acct.id, daily_send_limit=20, send_window_start="09:00",
                 send_window_end="18:00", timezone="America/Toronto", send_on_weekends=False,
                 review_before_send=False, attach_documents=False, warmup_day=1)
    for k, v in kw.items():
        setattr(c, k, v)
    db.add(c)
    await db.commit()
    return c


async def _person(db, email, *, domain=None, industry=None, employees=None, verdict=None,
                  first="Jane", last: "str | None" = "Doe", source="manual"):
    domain = domain or (email.split("@")[1] if email else f"co-{uuid.uuid4().hex[:6]}.com")
    co = (await db.execute(select(Company).where(Company.domain == domain))).scalars().first()
    if co is None:
        co = Company(name=f"Co {domain}", domain=domain, source="manual",
                     industry_segment=industry, employee_count=employees)
        db.add(co)
        await db.flush()
    ct = Contact(company_id=co.id, first_name=first, last_name=last, email=email, source=source,
                 email_verification_result=verdict, email_verified_at=MIDWEEK if verdict else None)
    db.add(ct)
    await db.commit()
    return ct


# ── the contact database ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_facets_group_contacts_by_verdict_source_industry_and_campaign(client, db_session):
    c = await _campaign(db_session, name="Q4 push")
    a = await _person(db_session, "a@one.com", verdict="deliverable", industry="Fintech", source="groq")
    await _person(db_session, "b@two.com", verdict="undeliverable", industry="Fintech", source="groq")
    await _person(db_session, "c@three.com", industry="Retail")
    db_session.add(CampaignProspect(campaign_id=c.id, contact_id=a.id, to_email=a.email, status="queued"))
    await db_session.commit()

    f = (await client.get("/api/v1/contacts/facets")).json()
    assert f["total"] == 3
    verdicts = {v["value"]: v["count"] for v in f["verification"]}
    assert verdicts == {"deliverable": 1, "undeliverable": 1, "unverified": 1}
    assert [v["value"] for v in f["verification"]] == ["deliverable", "undeliverable", "unverified"]
    assert {v["label"]: v["count"] for v in f["industry"]} == {"Fintech": 2, "Retail": 1}
    assert {v["value"]: v["count"] for v in f["source"]}["groq"] == 2
    assert f["campaign"] == [{"value": str(c.id), "label": "Q4 push", "count": 1}]


@pytest.mark.asyncio
async def test_contacts_filter_by_verdict_industry_size_and_campaign(client, db_session):
    c = await _campaign(db_session)
    a = await _person(db_session, "a@big.com", verdict="deliverable", industry="Fintech", employees=500)
    await _person(db_session, "b@small.com", verdict="risky", industry="Retail", employees=20)
    await _person(db_session, "c@none.com")
    db_session.add(CampaignProspect(campaign_id=c.id, contact_id=a.id, to_email=a.email, status="queued"))
    await db_session.commit()

    async def emails(**params):
        r = await client.get("/api/v1/contacts/", params=params)
        assert r.status_code == 200, r.text
        return sorted(i["email"] for i in r.json()["items"])

    assert await emails(verification="deliverable") == ["a@big.com"]
    assert await emails(verification="unverified") == ["c@none.com"]
    assert await emails(industry="tail") == ["b@small.com"]
    assert await emails(min_employees=100) == ["a@big.com"]
    assert await emails(max_employees=100) == ["b@small.com"]
    assert await emails(campaign_id=str(c.id)) == ["a@big.com"]
    item = (await client.get("/api/v1/contacts/", params={"verification": "risky"})).json()["items"][0]
    assert item["email_verification_result"] == "risky"


# ── adding from the database ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_adding_contacts_queues_them_and_says_why_others_were_skipped(client, db_session, dispatched):
    c = await _campaign(db_session)
    other = await _campaign(db_session)
    fresh = await _person(db_session, "fresh@lead.com")
    taken = await _person(db_session, "taken@lead2.com")
    db_session.add(CampaignProspect(campaign_id=other.id, contact_id=taken.id, to_email=taken.email,
                                    status="queued"))
    blocked = await _person(db_session, "gone@lead3.com")
    blocked.suppressed = True
    db_session.add(_gmail(email="me@ourco.com"))
    ours = await _person(db_session, "colleague@ourco.com")
    await db_session.commit()

    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/add",
                          json={"contact_ids": [str(x.id) for x in (fresh, taken, blocked, ours)]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] == 1 and body["processing"] == "dispatched"
    reasons = {s["email"]: s["reason"] for s in body["skipped"]}
    assert reasons == {"taken@lead2.com": "already_prospect", "gone@lead3.com": "blocked",
                       "colleague@ourco.com": "own_org"}
    assert all(s["label"] for s in body["skipped"])
    assert dispatched == [(str(c.org_id), str(c.id))]
    p = (await db_session.execute(select(CampaignProspect).where(
        CampaignProspect.campaign_id == c.id))).scalars().one()
    assert (p.to_email, p.status, p.source) == ("fresh@lead.com", "queued", "contact_db")


@pytest.mark.asyncio
async def test_a_database_contact_without_an_email_gets_one_from_hunter(client, db_session, dispatched, monkeypatch):
    from app.services import hunter_verification

    c = await _campaign(db_session)
    named = await _person(db_session, None, domain="found.com", first="Priya", last="Shah")
    hunter = _FakeHunter(finds={"found.com": "priya@found.com"})
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: hunter)

    body = (await client.post(f"/api/v1/campaigns/{c.id}/prospects/add",
                              json={"contact_ids": [str(named.id)]})).json()
    assert body["queued"] == 1 and hunter.find_calls == ["found.com"]
    await db_session.refresh(named)
    assert named.email == "priya@found.com"


@pytest.mark.asyncio
async def test_nothing_queued_dispatches_nothing(client, db_session, dispatched):
    c = await _campaign(db_session)
    nobody = await _person(db_session, None, first="X", last=None)
    body = (await client.post(f"/api/v1/campaigns/{c.id}/prospects/add",
                              json={"contact_ids": [str(nobody.id)]})).json()
    assert body["queued"] == 0 and body["processing"] == "nothing_queued"
    assert body["skipped"][0]["reason"] == "no_email" and dispatched == []


@pytest.mark.asyncio
async def test_add_refuses_a_stopped_campaign_and_unknown_contacts(client, db_session, dispatched):
    c = await _campaign(db_session, status="stopped")
    ct = await _person(db_session, "x@y.com")
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/add", json={"contact_ids": [str(ct.id)]})
    assert r.status_code == 409
    live = await _campaign(db_session)
    r = await client.post(f"/api/v1/campaigns/{live.id}/prospects/add", json={"contact_ids": [str(uuid.uuid4())]})
    assert r.status_code == 404


# ── adding by hand ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_manual_contact_joins_the_database_and_the_campaign(client, db_session, dispatched):
    c = await _campaign(db_session)
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/manual", json={
        "company_name": "Acme Robotics", "first_name": "Ana", "last_name": "Li",
        "title": "COO", "email": "Ana.Li@AcmeRobotics.com"})
    assert r.status_code == 200, r.text
    assert r.json()["queued"] == 1 and dispatched
    ct = (await db_session.execute(select(Contact).where(Contact.email == "ana.li@acmerobotics.com"))).scalars().one()
    co = await db_session.get(Company, ct.company_id)
    assert (co.name, co.domain, ct.title, ct.source) == ("Acme Robotics", "acmerobotics.com", "COO", "manual")
    # The same person again: one contact, and the second add says why it did nothing.
    again = (await client.post(f"/api/v1/campaigns/{c.id}/prospects/manual", json={
        "company_name": "Acme Robotics", "first_name": "Ana", "email": "ana.li@acmerobotics.com"})).json()
    assert again["queued"] == 0 and again["skipped"][0]["reason"] == "already_prospect"
    assert len((await db_session.execute(select(Contact).where(
        Contact.email == "ana.li@acmerobotics.com"))).scalars().all()) == 1


@pytest.mark.asyncio
async def test_a_manual_contact_without_an_email_is_found_by_hunter(client, db_session, dispatched, monkeypatch):
    from app.services import hunter_verification

    c = await _campaign(db_session)
    hunter = _FakeHunter(finds={"beta.dev": "sam@beta.dev"})
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: hunter)
    body = (await client.post(f"/api/v1/campaigns/{c.id}/prospects/manual", json={
        "company_name": "Beta", "domain": "https://www.beta.dev/", "first_name": "Sam", "last_name": "Ray"})).json()
    assert body["queued"] == 1 and hunter.find_calls == ["beta.dev"]


@pytest.mark.asyncio
async def test_a_malformed_manual_email_is_refused(client, db_session, dispatched):
    c = await _campaign(db_session)
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/manual", json={
        "company_name": "X", "first_name": "Y", "email": "not-an-email"})
    assert r.status_code == 422 and dispatched == []


@pytest.mark.asyncio
async def test_processing_verifies_drafts_and_schedules_without_discovering(db_session, monkeypatch):
    from app.services import hunter_verification, org_gemini

    c = await _campaign(db_session)
    ct = await _person(db_session, "new@added.com")
    db_session.add(CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email=ct.email, status="queued"))
    await db_session.commit()
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: _FakeHunter())

    async def fake_json(db, org_id, *, schema, **kw):
        return schema(subject="Hello", body="A short note.")

    async def no_discovery(*a, **kw):
        raise AssertionError("adding contacts must not run discovery")

    monkeypatch.setattr(org_gemini, "generate_json", fake_json)
    monkeypatch.setattr(svc, "discover_prospects", no_discovery)
    calls = []
    out = await svc.process_new_prospects(db_session, c.id, now=MIDWEEK, schedule=lambda *a: calls.append(a))
    assert out["verification"]["verified"] == 1 and out["drafted"] == 1 and out["scheduled"] == 1
    assert len(calls) == 1


# ── the sender ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_senders_list_gmail_and_smtp_accounts(client, db_session):
    db_session.add_all([_gmail(email="g@ourco.com"), _smtp(email="s@ourco.com"),
                        _smtp(email="off@ourco.com", sending_enabled=False)])
    await db_session.commit()
    senders = (await client.get("/api/v1/campaigns/senders")).json()
    assert {(s["kind"], s["email"]) for s in senders} == {("gmail", "g@ourco.com"), ("smtp", "s@ourco.com")}


async def _create(client, db_session, monkeypatch, **body):
    doc = DiscoveryDocument(filename="p.txt", file_type="txt", size_bytes=1, status=STATUS_READY,
                            content=b"x", extracted={"candidates": []})
    db_session.add_all([doc, OrgLlmKey(org_id=APPREND_ORG_ID, provider="gemini",
                                       key_encrypted=crypto.encrypt("k"), key_hint="••••")])
    await db_session.commit()

    async def fake_icp(db, org_id, docs):
        return svc.Icp(summary="Widgets"), "Widgets"

    from app.tasks import campaigns as campaign_tasks

    monkeypatch.setattr(svc, "build_icp", fake_icp)
    monkeypatch.setattr(campaign_tasks.run_campaign_now, "delay", lambda *a: None)
    return await client.post("/api/v1/campaigns", json={"name": "S", "document_ids": [str(doc.id)], **body})


@pytest.mark.asyncio
async def test_a_campaign_can_be_created_to_send_from_smtp(client, db_session, monkeypatch):
    smtp = _smtp(email="outbound@ourco.com")
    db_session.add(smtp)
    await db_session.commit()
    r = await _create(client, db_session, monkeypatch, sender_account_id=str(smtp.id))
    assert r.status_code == 201, r.text
    assert r.json()["smtp_account_id"] == str(smtp.id) and r.json()["gmail_account_id"] is None
    c = await db_session.get(Campaign, uuid.UUID(r.json()["id"]))
    assert (await svc.sending_account(db_session, c)).id == smtp.id


@pytest.mark.asyncio
async def test_switching_the_sender_clears_the_other_kind(client, db_session, monkeypatch):
    smtp, gm = _smtp(), _gmail()
    db_session.add_all([smtp, gm])
    await db_session.commit()
    created = (await _create(client, db_session, monkeypatch, sender_account_id=str(smtp.id))).json()
    r = await client.patch(f"/api/v1/campaigns/{created['id']}", json={"sender_account_id": str(gm.id)})
    assert r.status_code == 200, r.text
    assert r.json()["gmail_account_id"] == str(gm.id) and r.json()["smtp_account_id"] is None


@pytest.mark.asyncio
async def test_a_disconnected_or_unknown_sender_is_refused(client, db_session, monkeypatch):
    off = _smtp(status="error")
    db_session.add(off)
    await db_session.commit()
    r = await _create(client, db_session, monkeypatch, sender_account_id=str(off.id))
    assert r.status_code == 409 and r.json()["detail"]["code"] == "sender_unavailable"


@pytest.mark.asyncio
async def test_an_smtp_campaign_sends_through_its_smtp_account(db_session, monkeypatch):
    smtp = _smtp()
    db_session.add(smtp)
    await db_session.flush()
    c = await _campaign(db_session, gmail_account_id=None, smtp_account_id=smtp.id)
    ct = await _person(db_session, "ok@acme.com", verdict="deliverable")
    p = CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email=ct.email, status="approved",
                         email_subject="Hi", email_body="Hello there")
    db_session.add(p)
    await db_session.commit()
    seen = {}

    async def fake_send(db, **kw):
        seen.update(kw)
        return {"status": "sent", "message_id": "m1", "from_email": smtp.email}

    monkeypatch.setattr(email_dispatch, "send_outreach_email", fake_send)
    assert (await svc.send_prospect(db_session, p.id, MIDWEEK)).status == "sent"
    assert seen["sender_account_id"] == smtp.id


@pytest.mark.asyncio
async def test_a_campaign_created_without_settings_waits_for_review(client, db_session, monkeypatch):
    """The campaign is live and drafting the moment it exists; a client that
    sends no settings must get review-before-send, not auto-approved drafts."""
    db_session.add(_gmail(user_id=(await db_session.execute(
        select(User.id).where(User.email == "_default_test_admin@landline.app"))).scalar_one()))
    await db_session.commit()
    r = await _create(client, db_session, monkeypatch)
    assert r.status_code == 201, r.text
    assert r.json()["review_before_send"] is True


@pytest.mark.asyncio
async def test_settings_in_the_api_field_names_are_saved(client, db_session, monkeypatch):
    smtp = _smtp()
    db_session.add(smtp)
    await db_session.commit()
    created = (await _create(client, db_session, monkeypatch, sender_account_id=str(smtp.id),
                             review_before_send=False, daily_send_limit=40)).json()
    assert created["review_before_send"] is False and created["daily_send_limit"] == 40
    r = await client.patch(f"/api/v1/campaigns/{created['id']}", json={
        "review_before_send": True, "send_window_start": "08:00", "send_window_end": "16:00",
        "icp_json": {"summary": "Widgets", "target_locations": ["Ontario"],
                     "company_size": {"min": 10, "max": 50}}})
    body = r.json()
    assert body["review_before_send"] is True and body["send_window_start"] == "08:00"
    assert body["icp_json"]["target_locations"] == ["Ontario"]
