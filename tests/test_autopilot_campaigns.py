"""Autopilot Outreach: router, warm-up, window, dedup, sending and the bounce guard."""

import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.campaign import Campaign, CampaignDailyLog, CampaignProspect
from app.models.company import Company
from app.models.contact import Contact
from app.models.discovery_document import STATUS_READY, DiscoveryDocument
from app.models.email_template import EmailSend
from app.models.gmail_account import GmailAccount
from app.models.org_llm_key import OrgLlmKey
from app.org_context import APPREND_ORG_ID
from app.services import campaigns as svc
from app.services import crypto, email_dispatch

# Wednesday 2026-10-14, 14:00 in Toronto (18:00 UTC): inside a 09:00–18:00 window.
MIDWEEK = datetime(2026, 10, 14, 18, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2026, 10, 17, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _mx_valid(monkeypatch):
    """Discovery checks a domain's MX before spending Hunter credits; never ask
    real DNS from a test — the outcome would be a property of the network."""
    from app.services import email_validation

    monkeypatch.setattr(email_validation, "check_mx", lambda domain: "valid")


@pytest.fixture(autouse=True)
def _crypto_key():
    from cryptography.fernet import Fernet

    from app.config import get_settings

    settings = get_settings()
    prev = settings.token_encryption_key
    settings.token_encryption_key = Fernet.generate_key().decode()
    crypto._fernet.cache_clear()
    yield
    settings.token_encryption_key = prev
    crypto._fernet.cache_clear()


def _gmail(**kw) -> GmailAccount:
    defaults = dict(
        id=uuid.uuid4(), email=f"sender-{uuid.uuid4().hex[:6]}@example.com",
        google_sub=uuid.uuid4().hex, refresh_token_encrypted="x",
        status="connected", sending_enabled=True, is_active=True,
    )
    defaults.update(kw)
    return GmailAccount(**defaults)


async def _campaign(db, **kw) -> Campaign:
    acct = _gmail()
    db.add(acct)
    await db.flush()
    c = Campaign(
        name="Test", status="active", document_ids=[], icp_json={"summary": "We sell widgets"},
        gmail_account_id=acct.id, daily_send_limit=20, send_window_start="09:00",
        send_window_end="18:00", timezone="America/Toronto", send_on_weekends=False,
        review_before_send=False, attach_documents=False, warmup_day=1,
    )
    for k, v in kw.items():
        setattr(c, k, v)
    db.add(c)
    await db.commit()
    return c


async def _contact(db, email: str, *, verified: bool = False) -> Contact:
    co = Company(name=f"Co {email}", domain=email.split("@")[1], source="manual")
    db.add(co)
    await db.flush()
    ct = Contact(company_id=co.id, first_name="Jane", last_name="Doe", email=email)
    if verified:
        ct.email_verification_result = "deliverable"
        ct.email_verified_at = MIDWEEK
    db.add(ct)
    await db.commit()
    return ct


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_ramp_is_five_plus_five_a_day_capped_by_the_limit():
    c = Campaign(daily_send_limit=20, warmup_day=1)
    assert svc.ramp_cap(c) == 5
    c.warmup_day = 3
    assert svc.ramp_cap(c) == 15
    c.warmup_day = 30
    assert svc.ramp_cap(c) == 20


def test_window_is_business_hours_in_the_campaign_timezone():
    c = Campaign(send_window_start="09:00", send_window_end="18:00",
                 timezone="America/Toronto", send_on_weekends=False)
    assert svc.window_open(c, MIDWEEK)
    assert not svc.window_open(c, MIDWEEK.replace(hour=23))  # 19:00 Toronto
    assert not svc.window_open(c, SATURDAY)
    c.send_on_weekends = True
    assert svc.window_open(c, SATURDAY)


# ── router ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_campaign_builds_icp_and_picks_the_callers_mailbox(client, db_session, monkeypatch):
    from app.models.user import User

    admin = (await db_session.execute(
        select(User).where(User.email == "_default_test_admin@landline.app")
    )).scalars().first()
    acct = _gmail(user_id=admin.id)
    doc = DiscoveryDocument(
        filename="pitch.txt", file_type="txt", size_bytes=10, status=STATUS_READY,
        content=b"We sell widgets", extracted={"candidates": []},
    )
    db_session.add_all([acct, doc, OrgLlmKey(
        org_id=APPREND_ORG_ID, provider="gemini",
        key_encrypted=crypto.encrypt("k"), key_hint="••••",
    )])
    await db_session.commit()

    async def fake_icp(db, org_id, docs):
        return svc.Icp(summary="Widgets for factories", target_industries=["Manufacturing"]), "Widgets"

    from app.tasks import campaigns as campaign_tasks

    dispatched = []
    monkeypatch.setattr(svc, "build_icp", fake_icp)
    monkeypatch.setattr(campaign_tasks.run_campaign_now, "delay", lambda *a: dispatched.append(a))
    resp = await client.post("/api/v1/campaigns", json={
        "name": "Q4 push", "document_ids": [str(doc.id)], "daily_send_limit": 40,
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # The first search is dispatched now, not left for the next 15-minute tick.
    assert dispatched == [(str(APPREND_ORG_ID), body["id"])]
    assert body["icp_json"]["target_industries"] == ["Manufacturing"]
    assert body["gmail_account_id"] == str(acct.id)
    assert body["todays_cap"] == 5 and body["status"] == "active"

    listed = await client.get("/api/v1/campaigns")
    assert [c["id"] for c in listed.json()] == [body["id"]]
    detail = await client.get(f"/api/v1/campaigns/{body['id']}")
    assert detail.json()["metrics"]["total_prospects"] == 0


@pytest.mark.asyncio
async def test_create_refuses_without_an_ai_key(client, db_session):
    doc = DiscoveryDocument(filename="a.txt", file_type="txt", size_bytes=1, status=STATUS_READY)
    db_session.add(doc)
    await db_session.commit()
    resp = await client.post("/api/v1/campaigns", json={"name": "x", "document_ids": [str(doc.id)]})
    assert resp.status_code == 409
    assert resp.json()["detail"]["code"] == "gemini_key_required"


@pytest.mark.asyncio
async def test_limit_over_100_is_rejected(client, db_session):
    c = await _campaign(db_session)
    resp = await client.patch(f"/api/v1/campaigns/{c.id}", json={"daily_send_limit": 101})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_pause_resume_stop_and_review_flow(client, db_session):
    c = await _campaign(db_session, review_before_send=True)
    ct = await _contact(db_session, "lead@acme.com")
    p = CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email="lead@acme.com",
                         status="draft_created", email_subject="Hi", email_body="Hello")
    other = CampaignProspect(campaign_id=c.id, to_email="b@acme.com", status="queued")
    db_session.add_all([p, other])
    await db_session.commit()

    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/approve",
                          json={"email_subject": "Quick question"})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert r.json()["email_subject"] == "Quick question"

    page = (await client.get(f"/api/v1/campaigns/{c.id}/prospects?page_size=1")).json()
    assert page["total"] == 2 and len(page["items"]) == 1

    assert (await client.patch(f"/api/v1/campaigns/{c.id}", json={"status": "paused"})).json()["status"] == "paused"
    assert (await client.patch(f"/api/v1/campaigns/{c.id}", json={"status": "active"})).json()["status"] == "active"
    assert (await client.patch(f"/api/v1/campaigns/{c.id}", json={"status": "stopped"})).json()["status"] == "stopped"
    await db_session.refresh(other)
    assert other.status == "skipped"
    assert (await client.patch(f"/api/v1/campaigns/{c.id}", json={"status": "active"})).status_code == 409

    assert (await client.delete(f"/api/v1/campaigns/{c.id}")).status_code == 204
    assert (await client.get(f"/api/v1/campaigns/{c.id}")).status_code == 404


@pytest.mark.asyncio
async def test_skip_prospect(client, db_session):
    c = await _campaign(db_session)
    p = CampaignProspect(campaign_id=c.id, to_email="x@acme.com", status="queued")
    db_session.add(p)
    await db_session.commit()
    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/skip", json={"reason": "competitor"})
    assert r.status_code == 200 and r.json()["skip_reason"] == "competitor"
    again = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/approve")
    assert again.status_code == 409


# ── dedup ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_address_is_never_queued_twice_across_campaigns(db_session):
    a = await _campaign(db_session)
    b = await _campaign(db_session)
    ct = await _contact(db_session, "Dup@Acme.com")
    assert await svc.queue_contacts(db_session, a, [ct], source="document") == 1
    await db_session.commit()
    assert await svc.queue_contacts(db_session, b, [ct], source="document") == 0

    # Mailed by another path entirely → also excluded.
    ct2 = await _contact(db_session, "mailed@beta.com")
    db_session.add(EmailSend(to_email="mailed@beta.com", subject="s", body_html="b", status="sent"))
    await db_session.commit()
    assert await svc.queue_contacts(db_session, b, [ct2], source="document") == 0


# ── sending ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_schedule_spaces_sends_two_to_five_minutes_and_respects_the_ramp(db_session):
    c = await _campaign(db_session)
    for i in range(8):
        ct = await _contact(db_session, f"p{i}@acme{i}.com", verified=True)
        db_session.add(CampaignProspect(
            campaign_id=c.id, contact_id=ct.id, to_email=f"p{i}@acme{i}.com", status="approved",
            email_subject="s", email_body="b", approved_at=MIDWEEK - timedelta(minutes=8 - i),
        ))
    await db_session.commit()
    calls = []
    n = await svc.schedule_sends(db_session, c, MIDWEEK,
                                 schedule=lambda org, pid, cd: calls.append(cd))
    # 15-minute tick, 120–300s gaps → at most 8 fit, ramp day 1 caps at 5.
    assert 1 <= n <= 5 and len(calls) == n
    assert calls[0] == 0
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert all(120 <= g <= 300 for g in gaps)


@pytest.mark.asyncio
async def test_nothing_is_scheduled_outside_the_window(db_session):
    c = await _campaign(db_session)
    db_session.add(CampaignProspect(campaign_id=c.id, to_email="p@acme.com", status="approved",
                                    email_subject="s", email_body="b"))
    await db_session.commit()
    assert await svc.schedule_sends(db_session, c, SATURDAY, schedule=lambda *a: None) == 0


@pytest.mark.asyncio
async def test_send_prospect_goes_through_the_dispatcher_and_records_the_send(db_session, monkeypatch):
    c = await _campaign(db_session)
    ct = await _contact(db_session, "ok@acme.com")
    p = CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email="ok@acme.com",
                         status="approved", email_subject="Hi", email_body="Hello there")
    db_session.add(p)
    await db_session.commit()
    seen = {}

    async def fake_send(db, **kw):
        seen.update(kw)
        return {"status": "sent", "gmail_message_id": "m1", "gmail_thread_id": "t1",
                "from_email": "me@example.com"}

    monkeypatch.setattr(email_dispatch, "send_outreach_email", fake_send)
    out = await svc.send_prospect(db_session, p.id, MIDWEEK)
    assert out.status == "sent"
    assert seen["to_email"] == "ok@acme.com" and seen["sender_account_id"] == c.gmail_account_id
    await db_session.refresh(p)
    assert p.status == "sent" and p.gmail_thread_id == "t1" and p.email_send_id is not None
    log = (await db_session.execute(select(CampaignDailyLog))).scalars().first()
    assert log.emails_sent == 1

    # A second delivery of the same task is harmless.
    assert (await svc.send_prospect(db_session, p.id, MIDWEEK)).status == "not_approved"


@pytest.mark.asyncio
async def test_a_dispatcher_hold_puts_the_prospect_back(db_session, monkeypatch):
    c = await _campaign(db_session)
    p = CampaignProspect(campaign_id=c.id, to_email="hold@acme.com", status="approved",
                         email_subject="Hi", email_body="Hello")
    db_session.add(p)
    await db_session.commit()

    async def capped(db, **kw):
        return {"status": "skipped", "reason": "daily cap"}

    monkeypatch.setattr(email_dispatch, "send_outreach_email", capped)
    assert (await svc.send_prospect(db_session, p.id, MIDWEEK)).status == "held"
    await db_session.refresh(p)
    assert p.status == "approved" and p.sent_at is None


@pytest.mark.asyncio
async def test_an_exception_mid_send_fails_rather_than_retrying(db_session, monkeypatch):
    c = await _campaign(db_session)
    p = CampaignProspect(campaign_id=c.id, to_email="boom@acme.com", status="approved",
                         email_subject="Hi", email_body="Hello")
    db_session.add(p)
    await db_session.commit()

    async def boom(db, **kw):
        raise RuntimeError("socket closed")

    monkeypatch.setattr(email_dispatch, "send_outreach_email", boom)
    assert (await svc.send_prospect(db_session, p.id, MIDWEEK)).status == "failed"
    await db_session.refresh(p)
    assert p.status == "failed"


# ── bounce guard, warm-up ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bounce_rate_over_five_percent_pauses_the_campaign(db_session):
    c = await _campaign(db_session)
    for i in range(10):
        db_session.add(CampaignProspect(
            campaign_id=c.id, to_email=f"b{i}@acme.com",
            status="bounced" if i == 0 else "sent", sent_at=MIDWEEK - timedelta(hours=1),
            bounced_at=MIDWEEK if i == 0 else None,
        ))
    await db_session.commit()
    assert await svc.enforce_bounce_rate(db_session, c, MIDWEEK)
    assert c.status == "paused" and c.pause_reason == "bounce_rate"


@pytest.mark.asyncio
async def test_warmup_advances_only_after_a_day_that_sent(db_session):
    c = await _campaign(db_session)
    await svc.advance_warmup(db_session, c, MIDWEEK)
    assert c.warmup_day == 1 and c.warmup_on == date(2026, 10, 14)
    # Next day, nothing was sent yesterday → ramp holds.
    await svc.advance_warmup(db_session, c, MIDWEEK + timedelta(days=1))
    assert c.warmup_day == 1
    db_session.add(CampaignDailyLog(campaign_id=c.id, date=date(2026, 10, 15), emails_sent=5,
                                    prospects_found=0, emails_bounced=0, errors=[]))
    await db_session.flush()
    await svc.advance_warmup(db_session, c, MIDWEEK + timedelta(days=2))
    assert c.warmup_day == 2


# ── regressions ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_icp_reads_the_deferred_document_bytes_on_a_fresh_session(
    db_session, db_engine, monkeypatch
):
    """``content`` is deferred: a lazy load from an async session raises
    MissingGreenlet, which surfaced in production as POST /campaigns failing."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.services import org_gemini

    doc = DiscoveryDocument(filename="pitch.txt", file_type="txt", size_bytes=15,
                            status=STATUS_READY, content=b"We sell widgets", extracted={})
    db_session.add(doc)
    await db_session.commit()
    prompts = []

    async def fake_json(db, org_id, *, prompt, schema, **kw):
        prompts.append(prompt)
        return schema(summary="Widgets")

    monkeypatch.setattr(org_gemini, "generate_json", fake_json)
    fresh = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with fresh() as s:
        loaded = (await s.execute(select(DiscoveryDocument).where(DiscoveryDocument.id == doc.id))).scalars().one()
        icp, summary = await svc.build_icp(s, APPREND_ORG_ID, [loaded])
    assert summary == "Widgets"
    assert "We sell widgets" in prompts[0]


@pytest.mark.asyncio
async def test_a_route_crash_is_not_reported_as_bad_credentials(client, db_session, monkeypatch):
    """The auth middleware used to wrap call_next in the token try/except, so any
    exception from an authenticated route came back as a 401."""
    from app.models.user import User

    admin = (await db_session.execute(
        select(User).where(User.email == "_default_test_admin@landline.app")
    )).scalars().first()
    doc = DiscoveryDocument(filename="a.txt", file_type="txt", size_bytes=1, status=STATUS_READY)
    db_session.add_all([doc, _gmail(user_id=admin.id), OrgLlmKey(
        org_id=APPREND_ORG_ID, provider="gemini", key_encrypted=crypto.encrypt("k"), key_hint="••••",
    )])
    await db_session.commit()

    async def boom(*a, **kw):
        raise RuntimeError("route bug")

    monkeypatch.setattr(svc, "build_icp", boom)
    try:
        resp = await client.post("/api/v1/campaigns", json={"name": "x", "document_ids": [str(doc.id)]})
    except RuntimeError:
        return  # propagated as a server error — correct
    assert resp.status_code != 401, resp.text


# ── guessed emails for LLM-discovered contacts ───────────────────────────────


@pytest.mark.parametrize(
    "first, domain, expected",
    [
        ("Jane", "acme.com", "jane@acme.com"),
        ("  JANE ", "ACME.com", "jane@acme.com"),
        ("Mary Ann", "acme.com", "mary@acme.com"),
        ("José", "acme.io", "jose@acme.io"),
        ("O'Neil", "acme.com", "oneil@acme.com"),
        ("Jane", "https://www.acme.com/about", "jane@acme.com"),
        ("Jane", "www.acme.co.uk:443", "jane@acme.co.uk"),
        (None, "acme.com", None),
        ("", "acme.com", None),
        ("Jane", None, None),
        ("Jane", "localhost", None),
        ("李", "acme.com", None),
    ],
)
def test_guess_email_is_firstname_at_domain(first, domain, expected):
    assert svc.guess_email(first, domain) == expected


class _FakeHunter:
    """Stands in for HunterVerificationService: canned finder hits and verdicts."""

    def __init__(self, *, enabled=True, finds=None, verdicts=None, people=None):
        self.enabled = enabled
        self.finds = finds or {}          # domain -> email
        self.verdicts = verdicts or {}    # email -> result
        self.people = people or {}        # domain -> domain-search entries
        self.find_calls, self.verify_calls, self.domain_calls = [], [], []

    async def make_budget(self, db):
        return None

    async def find_email(self, db, company, first, last, *, budget=None):
        self.find_calls.append(company.domain)
        return self.finds.get(company.domain)

    async def domain_search(self, db, domain, *, budget=None):
        self.domain_calls.append(domain)
        people = self.people.get(domain, [])
        return {"status": "ok" if people else "not_found",
                "email": people[0]["email"] if people else None, "emails": people}

    async def verify_contact(self, db, contact, *, budget=None):
        self.verify_calls.append(contact.email)
        result = self.verdicts.get(contact.email, "deliverable")
        contact.email_verification_result = result
        contact.email_verified_at = MIDWEEK
        contact.email_verification_meta = {"hunter_result": result}
        return {"status": "ok", "result": result}


async def _run_search(db_session, monkeypatch, hunter, companies, *, grounded=True):
    from types import SimpleNamespace

    from app.services import gemini_discovery, hunter_verification, ingestion

    _run_search.grounded = grounded

    async def fake_search(*a, **kw):
        _run_search.last_query = kw.get("query")
        _run_search.last_exclude = kw.get("exclude_domains")
        return SimpleNamespace(records=[], provider="groq", grounded=_run_search.grounded)

    class FakeIngest:
        async def ingest(self, db, records):
            return SimpleNamespace(outcomes=[SimpleNamespace(company_id=c.id) for c in companies])

    monkeypatch.setattr(gemini_discovery, "search_companies", fake_search)
    monkeypatch.setattr(ingestion, "get_ingestion_service", lambda: FakeIngest())
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: hunter)
    campaign = await _campaign(db_session)
    return campaign, await svc._search_companies(db_session, campaign)


async def _named(db, domain, first="Priya", last="Shah"):
    co = Company(name=f"Co {domain or 'none'}", domain=domain, source="manual")
    db.add(co)
    await db.flush()
    ct = Contact(company_id=co.id, first_name=first, last_name=last)
    db.add(ct)
    await db.commit()
    return co, ct


@pytest.mark.asyncio
async def test_hunter_finder_supplies_the_real_address_then_it_is_verified(db_session, monkeypatch):
    co, ct = await _named(db_session, "finder.com")
    hunter = _FakeHunter(finds={"finder.com": "P.Shah@finder.com"})
    campaign, queued = await _run_search(db_session, monkeypatch, hunter, [co])
    assert queued == 1
    await db_session.refresh(ct)
    assert ct.email == "p.shah@finder.com"
    assert hunter.verify_calls == ["p.shah@finder.com"]
    assert ct.email_verification_result == "deliverable"
    assert ct.email_verification_meta["email_origin"] == "hunter_finder"


@pytest.mark.asyncio
async def test_pattern_guess_is_the_fallback_when_the_finder_misses(db_session, monkeypatch):
    co, ct = await _named(db_session, "guessco.com")
    hunter = _FakeHunter(verdicts={"priya@guessco.com": "risky"})
    _, queued = await _run_search(db_session, monkeypatch, hunter, [co])
    assert queued == 1
    await db_session.refresh(ct)
    assert hunter.find_calls == ["guessco.com"]
    assert ct.email == "priya@guessco.com"
    assert ct.email_verification_result == "risky"
    assert ct.email_verification_meta["email_origin"] == "pattern"


@pytest.mark.asyncio
async def test_without_a_hunter_key_the_guess_is_queued_unverified(db_session, monkeypatch):
    co, ct = await _named(db_session, "nokey.com")
    keep = await _contact(db_session, "real@kept.com")
    hunter = _FakeHunter(enabled=False)
    campaign, queued = await _run_search(
        db_session, monkeypatch, hunter, [co, await db_session.get(Company, keep.company_id)]
    )
    assert queued == 2
    await db_session.refresh(ct)
    await db_session.refresh(keep)
    assert ct.email == "priya@nokey.com" and ct.email_verified_at is None
    assert keep.email == "real@kept.com"  # a real address is never overwritten
    assert hunter.find_calls == [] and hunter.verify_calls == []


@pytest.mark.asyncio
async def test_a_company_with_no_address_is_recorded_not_dropped(db_session, monkeypatch):
    co, ct = await _named(db_session, None)
    campaign, queued = await _run_search(db_session, monkeypatch, _FakeHunter(enabled=False), [co])
    assert queued == 0
    rows = (await db_session.execute(
        select(CampaignProspect).where(CampaignProspect.campaign_id == campaign.id)
    )).scalars().all()
    assert [(r.status, r.company_id, r.to_email) for r in rows] == [("no_email", co.id, None)]
    # Not pending: it never reaches drafting and never holds the discovery gate.
    assert await svc._pending_count(db_session, campaign) == 0


@pytest.mark.asyncio
async def test_unverified_prospects_are_held_and_rejected_ones_retired(db_session, monkeypatch):
    from app.services import hunter_verification

    c = await _campaign(db_session)
    good = await _contact(db_session, "good@v1.com", verified=True)
    pending = await _contact(db_session, "pending@v2.com")
    bad = await _contact(db_session, "bad@v3.com")
    bad.email_verification_result, bad.email_verified_at = "undeliverable", MIDWEEK
    for ct in (good, pending, bad):
        db_session.add(CampaignProspect(
            campaign_id=c.id, contact_id=ct.id, to_email=ct.email, status="approved",
            email_subject="s", email_body="b", approved_at=MIDWEEK,
        ))
    await db_session.commit()

    # No key: nothing is verified, the bad one is retired, the pending one waits.
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: _FakeHunter(enabled=False))
    assert await svc.verify_prospects(db_session, c) == {"verified": 0, "rejected": 1}
    calls = []
    assert await svc.schedule_sends(db_session, c, MIDWEEK,
                                    schedule=lambda org, pid, cd: calls.append(pid)) == 1
    by_contact = {p.contact_id: p for p in (await db_session.execute(
        select(CampaignProspect).where(CampaignProspect.campaign_id == c.id))).scalars()}
    assert calls == [by_contact[good.id].id]
    assert by_contact[pending.id].scheduled_for is None
    assert by_contact[bad.id].status == "skipped"

    # With a key the held one is verified and becomes sendable.
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: _FakeHunter())
    assert (await svc.verify_prospects(db_session, c))["verified"] == 1
    await db_session.refresh(pending)
    assert pending.email_verification_result == "deliverable"


@pytest.mark.asyncio
async def test_metrics_report_discovery_queue_and_verification(db_session):
    from app.routers.campaigns import _metrics

    c = await _campaign(db_session)
    good = await _contact(db_session, "good@m1.com", verified=True)
    waiting = await _contact(db_session, "wait@m2.com")
    co, _ = await _named(db_session, None)
    db_session.add_all([
        CampaignProspect(campaign_id=c.id, company_id=good.company_id, contact_id=good.id,
                         to_email=good.email, status="approved", email_subject="s", email_body="b"),
        CampaignProspect(campaign_id=c.id, company_id=waiting.company_id, contact_id=waiting.id,
                         to_email=waiting.email, status="draft_created", email_subject="s", email_body="b"),
        CampaignProspect(campaign_id=c.id, company_id=co.id, status="no_email"),
    ])
    await db_session.commit()
    m = await _metrics(db_session, c)
    assert m["companies_discovered"] == 3
    assert m["companies_without_email"] == 1
    assert m["queued_to_send"] == 1
    assert (m["pending_verified"], m["pending_unverified"]) == (1, 1)


# ── tick ordering and the first run on create ────────────────────────────────


@pytest.mark.asyncio
async def test_verification_runs_before_drafting(db_session, monkeypatch):
    c = await _campaign(db_session, last_discovery_on=svc.local_today(
        Campaign(timezone="America/Toronto"), MIDWEEK))
    order = []

    async def fake_verify(db, campaign):
        order.append("verify")
        return {"verified": 0, "rejected": 0}

    async def fake_drafts(db, campaign, now=None):
        order.append("draft")
        return 0

    monkeypatch.setattr(svc, "verify_prospects", fake_verify)
    monkeypatch.setattr(svc, "generate_drafts", fake_drafts)
    await svc.run_campaign(db_session, c, MIDWEEK, schedule=lambda *a: None)
    assert order == ["verify", "draft"]


@pytest.mark.asyncio
async def test_a_rejected_address_is_never_drafted(db_session, monkeypatch):
    from app.services import hunter_verification

    c = await _campaign(db_session, last_discovery_on=svc.local_today(
        Campaign(timezone="America/Toronto"), MIDWEEK))
    bad = await _contact(db_session, "gone@nope.com")
    db_session.add(CampaignProspect(campaign_id=c.id, contact_id=bad.id, to_email=bad.email, status="queued"))
    await db_session.commit()
    monkeypatch.setattr(hunter_verification, "get_hunter_service",
                        lambda: _FakeHunter(verdicts={"gone@nope.com": "undeliverable"}))
    drafted_for = []

    async def fake_drafts(db, campaign, now=None):
        rows = (await db.execute(select(CampaignProspect).where(
            CampaignProspect.campaign_id == campaign.id, CampaignProspect.status == "queued"))).scalars().all()
        drafted_for.extend(p.to_email for p in rows)
        return len(rows)

    monkeypatch.setattr(svc, "generate_drafts", fake_drafts)
    out = await svc.run_campaign(db_session, c, MIDWEEK, schedule=lambda *a: None)
    assert out["verification"] == {"verified": 1, "rejected": 1}
    assert drafted_for == []


async def _drafting(db_session, monkeypatch, *, hunter_enabled):
    """A campaign with one verified and one not-yet-verified queued prospect."""
    from app.services import hunter_verification, org_gemini

    c = await _campaign(db_session)
    ok = await _contact(db_session, "ok@verified.com", verified=True)
    pending = await _contact(db_session, "guess@unchecked.com")
    for ct in (pending, ok):  # the unverified one is older, so it would be picked first
        db_session.add(CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email=ct.email, status="queued"))
        await db_session.commit()
    monkeypatch.setattr(hunter_verification, "get_hunter_service",
                        lambda: _FakeHunter(enabled=hunter_enabled))
    prompted = []

    async def fake_json(db, org_id, *, prompt, schema, **kw):
        prompted.append(prompt)
        return schema(subject="Hello", body="A short note.")

    monkeypatch.setattr(org_gemini, "generate_json", fake_json)
    drafted = await svc.generate_drafts(db_session, c, MIDWEEK)
    rows = (await db_session.execute(select(CampaignProspect).where(
        CampaignProspect.campaign_id == c.id))).scalars().all()
    return drafted, {p.to_email: p.status for p in rows}


@pytest.mark.asyncio
async def test_with_hunter_on_only_verified_prospects_are_drafted(db_session, monkeypatch):
    """Verify and draft each take a few rows per tick and pick different ones, so
    ordering alone let the E2E campaign draft guesses Hunter later rejected."""
    drafted, statuses = await _drafting(db_session, monkeypatch, hunter_enabled=True)
    assert drafted == 1
    assert statuses == {"ok@verified.com": "approved", "guess@unchecked.com": "queued"}


@pytest.mark.asyncio
async def test_without_a_hunter_key_drafting_does_not_wait_for_verification(db_session, monkeypatch):
    drafted, statuses = await _drafting(db_session, monkeypatch, hunter_enabled=False)
    assert drafted == 2
    assert set(statuses.values()) == {"approved"}


@pytest.mark.asyncio
async def test_run_campaign_now_runs_one_active_campaign(db_session, monkeypatch):
    c = await _campaign(db_session)
    ran = []

    async def fake_run(db, campaign, now=None, *, schedule):
        ran.append(campaign.id)
        return {"campaign_id": str(campaign.id), "discovered": 3}

    monkeypatch.setattr(svc, "run_campaign", fake_run)
    assert (await svc.run_campaign_now(db_session, c.id, schedule=lambda *a: None))["discovered"] == 3
    assert ran == [c.id]

    c.status = "paused"
    await db_session.commit()
    out = await svc.run_campaign_now(db_session, c.id, schedule=lambda *a: None)
    assert out["skipped"] == "not_active" and ran == [c.id]


@pytest.mark.asyncio
async def test_run_campaign_now_logs_a_failure_instead_of_raising(db_session, monkeypatch):
    c = await _campaign(db_session)

    async def boom(db, campaign, now=None, *, schedule):
        raise RuntimeError("search exploded")

    monkeypatch.setattr(svc, "run_campaign", boom)
    out = await svc.run_campaign_now(db_session, c.id, now=MIDWEEK, schedule=lambda *a: None)
    assert out["error"] == "search exploded"
    log = (await db_session.execute(select(CampaignDailyLog).where(
        CampaignDailyLog.campaign_id == c.id))).scalars().one()
    assert "search exploded" in str(log.errors)


@pytest.mark.asyncio
async def test_deleting_an_active_campaign_stops_it_and_keeps_the_sent_record(client, db_session):
    c = await _campaign(db_session)
    pending = CampaignProspect(campaign_id=c.id, to_email="later@acme.com", status="approved",
                               email_subject="s", email_body="b")
    sent = CampaignProspect(campaign_id=c.id, to_email="done@acme.com", status="sent")
    db_session.add_all([pending, sent])
    await db_session.commit()

    assert (await client.delete(f"/api/v1/campaigns/{c.id}")).status_code == 204
    assert c.id not in [x["id"] for x in (await client.get("/api/v1/campaigns")).json()]
    await db_session.refresh(c)
    await db_session.refresh(pending)
    await db_session.refresh(sent)
    assert c.status == "stopped" and c.deleted_at is not None
    assert pending.status == "skipped" and pending.skip_reason == "Campaign deleted"
    # Kept, so the address is never mailed again by a later campaign.
    assert sent.status == "sent"
    assert await svc.run_campaigns_tick(db_session, now=MIDWEEK, schedule=lambda *a: None) == []


@pytest.mark.parametrize(
    "first, last, ok",
    [
        ("Tobi", "Lütke", True),
        ("Mary-Jane", "O'Neil", True),
        ("Priya", None, True),
        ("John", "K. (Placeholder for CTO)", False),  # seen live, cost a Hunter search
        ("CTO", None, False),
        ("Head", "of Engineering", False),
        ("J", None, False),
        ("John", "Doe", False),
        ("Sales", "Team", False),
        ("Ana", "Li2", False),
        ("info@acme.com", None, False),
    ],
)
def test_plausible_person_name(first, last, ok):
    assert svc.plausible_person_name(first, last) is ok


@pytest.mark.asyncio
async def test_a_placeholder_contact_spends_no_hunter_credit(db_session, monkeypatch):
    co, ct = await _named(db_session, "veeva.com", first="John", last="K. (Placeholder for CTO)")
    hunter = _FakeHunter()
    campaign, queued = await _run_search(db_session, monkeypatch, hunter, [co])
    assert queued == 0
    assert hunter.find_calls == [] and hunter.verify_calls == []
    await db_session.refresh(ct)
    assert ct.email is None
    rows = (await db_session.execute(
        select(CampaignProspect).where(CampaignProspect.campaign_id == campaign.id)
    )).scalars().all()
    assert [r.status for r in rows] == ["no_email"]


@pytest.mark.asyncio
async def test_a_real_name_is_preferred_over_an_earlier_placeholder(db_session, monkeypatch):
    co, _ = await _named(db_session, "mixed.com", first="CTO", last=None)
    real = Contact(company_id=co.id, first_name="Ravi", last_name="Iyer")
    db_session.add(real)
    await db_session.commit()
    hunter = _FakeHunter(finds={"mixed.com": "ravi.iyer@mixed.com"})
    _, queued = await _run_search(db_session, monkeypatch, hunter, [co])
    assert queued == 1
    await db_session.refresh(real)
    assert real.email == "ravi.iyer@mixed.com"


@pytest.mark.asyncio
async def test_discovery_tells_the_model_which_companies_it_already_found(db_session, monkeypatch):
    old = await _campaign(db_session)
    seen = await _contact(db_session, "x@already.com")
    db_session.add(CampaignProspect(campaign_id=old.id, company_id=seen.company_id,
                                    contact_id=seen.id, to_email=seen.email, status="skipped"))
    await db_session.commit()
    co, _ = await _named(db_session, "fresh.com")
    await _run_search(db_session, monkeypatch, _FakeHunter(enabled=False), [co])
    assert _run_search.last_query == "We sell widgets"
    assert "already.com" in _run_search.last_exclude


@pytest.mark.asyncio
async def test_a_discovered_company_with_no_prospect_row_is_still_excluded(db_session, monkeypatch):
    """Early runs left 23 companies with no prospect row; never excluded, Groq
    returned the same household names every day."""
    db_session.add(Company(name="Stripe", domain="stripe.com", source="groq"))
    db_session.add(Company(name="Web Co", domain="webco.io", source="tavily+gemini"))
    db_session.add(Company(name="Mine", domain="manual.com", source="manual"))
    await db_session.commit()
    co, _ = await _named(db_session, "fresh2.com")
    await _run_search(db_session, monkeypatch, _FakeHunter(enabled=False), [co])
    assert {"stripe.com", "webco.io"} <= set(_run_search.last_exclude)
    assert "manual.com" not in _run_search.last_exclude


@pytest.mark.asyncio
async def test_a_first_search_sends_the_bare_icp(db_session, monkeypatch):
    co, _ = await _named(db_session, "first.com")
    await _run_search(db_session, monkeypatch, _FakeHunter(enabled=False), [co])
    assert _run_search.last_query == "We sell widgets"


# ── instant send on approve ──────────────────────────────────────────────────

async def _approved(db, campaign, email, *, verified=True):
    ct = await _contact(db, email, verified=verified)
    p = CampaignProspect(campaign_id=campaign.id, contact_id=ct.id, to_email=email,
                         status="approved", approved_at=MIDWEEK,
                         email_subject="Hi", email_body="Hello")
    db.add(p)
    await db.commit()
    return p


@pytest.mark.asyncio
async def test_send_approved_now_schedules_the_approved_email_immediately(db_session):
    c = await _campaign(db_session)
    p = await _approved(db_session, c, "now@acme.com")
    calls = []
    out = await svc.send_approved_now(db_session, c.id, now=MIDWEEK, schedule=lambda *a: calls.append(a))
    assert out["scheduled"] == 1
    assert calls == [(c.org_id, p.id, 0)]  # no countdown: on the wire now


@pytest.mark.asyncio
async def test_a_burst_of_approvals_is_still_paced(db_session):
    """Instant means the first one goes now — not that the 120–300s gap is gone."""
    c = await _campaign(db_session)
    await _approved(db_session, c, "one@first.com")
    calls = []
    await svc.send_approved_now(db_session, c.id, now=MIDWEEK, schedule=lambda *a: calls.append(a))
    await _approved(db_session, c, "two@second.com")
    await svc.send_approved_now(db_session, c.id, now=MIDWEEK, schedule=lambda *a: calls.append(a))
    assert [cd for *_, cd in calls][0] == 0
    assert svc.GAP_SECONDS[0] <= calls[1][2] <= svc.GAP_SECONDS[1]


@pytest.mark.asyncio
async def test_an_unverified_approval_is_verified_then_sent_in_the_same_pass(db_session, monkeypatch):
    from app.services import hunter_verification

    c = await _campaign(db_session)
    p = await _approved(db_session, c, "fresh@acme.com", verified=False)
    hunter = _FakeHunter()
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: hunter)
    calls = []
    out = await svc.send_approved_now(db_session, c.id, now=MIDWEEK, schedule=lambda *a: calls.append(a))
    assert hunter.verify_calls == ["fresh@acme.com"]
    assert out["scheduled"] == 1 and calls[0][1] == p.id


@pytest.mark.asyncio
async def test_instant_send_neither_discovers_nor_drafts(db_session, monkeypatch):
    c = await _campaign(db_session)

    async def boom(*a, **k):
        raise AssertionError("instant send must not spend discovery/drafting budget")

    monkeypatch.setattr(svc, "discover_prospects", boom)
    monkeypatch.setattr(svc, "generate_drafts", boom)
    out = await svc.send_approved_now(db_session, c.id, now=MIDWEEK, schedule=lambda *a: None)
    assert out["scheduled"] == 0 and "error" not in out


@pytest.mark.asyncio
async def test_instant_send_respects_the_window_and_campaign_status(db_session):
    c = await _campaign(db_session)
    await _approved(db_session, c, "later@acme.com")
    calls = []
    out = await svc.send_approved_now(db_session, c.id, now=SATURDAY, schedule=lambda *a: calls.append(a))
    assert out["scheduled"] == 0 and calls == []  # weekend, weekends off

    c.status = "paused"
    await db_session.commit()
    out = await svc.send_approved_now(db_session, c.id, now=MIDWEEK, schedule=lambda *a: calls.append(a))
    assert out["skipped"] == "not_active" and calls == []


@pytest.mark.asyncio
async def test_approving_dispatches_the_send_instead_of_waiting_for_the_tick(client, db_session, monkeypatch):
    from app.tasks import campaigns as campaign_tasks

    c = await _campaign(db_session, review_before_send=True)
    ct = await _contact(db_session, "lead@acme.com", verified=True)
    p = CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email=ct.email,
                         status="draft_created", email_subject="Hi", email_body="Hello")
    db_session.add(p)
    await db_session.commit()
    dispatched = []
    monkeypatch.setattr(campaign_tasks.send_approved_now, "delay", lambda *a: dispatched.append(a))
    monkeypatch.setattr(svc, "window_open", lambda campaign, now=None: True)

    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/approve")
    assert r.status_code == 200
    assert r.json()["instant_send"] == "dispatched"
    assert dispatched == [(str(c.org_id), str(c.id))]


@pytest.mark.asyncio
async def test_approving_outside_the_window_says_so_and_dispatches_nothing(client, db_session, monkeypatch):
    from app.tasks import campaigns as campaign_tasks

    c = await _campaign(db_session, review_before_send=True)
    p = CampaignProspect(campaign_id=c.id, to_email="x@acme.com", status="draft_created",
                         email_subject="Hi", email_body="Hello")
    db_session.add(p)
    await db_session.commit()
    dispatched = []
    monkeypatch.setattr(campaign_tasks.send_approved_now, "delay", lambda *a: dispatched.append(a))
    monkeypatch.setattr(svc, "window_open", lambda campaign, now=None: False)

    r = await client.post(f"/api/v1/campaigns/{c.id}/prospects/{p.id}/approve")
    assert r.json()["instant_send"] == "window_closed"
    assert dispatched == []


@pytest.mark.asyncio
async def test_resuming_a_campaign_sends_what_was_approved_while_paused(client, db_session, monkeypatch):
    from app.tasks import campaigns as campaign_tasks

    c = await _campaign(db_session, status="paused")
    dispatched = []
    monkeypatch.setattr(campaign_tasks.send_approved_now, "delay", lambda *a: dispatched.append(a))
    monkeypatch.setattr(svc, "window_open", lambda campaign, now=None: True)

    r = await client.patch(f"/api/v1/campaigns/{c.id}", json={"status": "active"})
    assert r.status_code == 200
    assert dispatched == [(str(c.org_id), str(c.id))]
    # Pausing doesn't.
    await client.patch(f"/api/v1/campaigns/{c.id}", json={"status": "paused"})
    assert len(dispatched) == 1


# ── where addresses come from: documents, own org, no-web search, MX ─────────

async def _doc_campaign(db, candidates):
    doc = DiscoveryDocument(filename="brief.txt", file_type="txt", size_bytes=10,
                            status=STATUS_READY, content=b"brief",
                            extracted={"candidates": candidates})
    db.add(doc)
    await db.commit()
    return await _campaign(db, document_ids=[str(doc.id)])


def _cand(cid, name, domain, *people):
    return {"id": cid, "name": name, "domain": domain, "confidence": 1.0, "contacts": [
        {"id": f"{cid}-p{i}", "first_name": f, "last_name": l, "email": e, "title": None}
        for i, (f, l, e) in enumerate(people, 1)
    ]}


@pytest.mark.asyncio
async def test_a_named_document_contact_without_an_email_goes_to_hunter(db_session, monkeypatch):
    from app.services import hunter_verification

    c = await _doc_campaign(db_session, [_cand("c1", "Named Co", "named.com", ("Priya", "Shah", None))])
    hunter = _FakeHunter(finds={"named.com": "priya.shah@named.com"})
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: hunter)

    assert await svc._seed_from_documents(db_session, c) == 1
    assert hunter.find_calls == ["named.com"]
    assert hunter.verify_calls == ["priya.shah@named.com"]
    p = (await db_session.execute(select(CampaignProspect).where(
        CampaignProspect.campaign_id == c.id))).scalars().one()
    assert (p.to_email, p.source) == ("priya.shah@named.com", "document")


@pytest.mark.asyncio
async def test_document_emails_are_still_queued_as_given(db_session, monkeypatch):
    from app.services import hunter_verification

    c = await _doc_campaign(db_session, [
        _cand("c1", "Mailed Co", "mailed.com", ("Ana", "Li", "ana@mailed.com"), ("Bo", "Ng", None)),
    ])
    hunter = _FakeHunter()
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: hunter)
    assert await svc._seed_from_documents(db_session, c) == 1
    # The company already has an address from the document: no credit spent on Bo.
    assert hunter.find_calls == []


@pytest.mark.asyncio
async def test_the_document_named_person_is_resolved_not_an_older_colleague(db_session, monkeypatch):
    from app.services import hunter_verification

    co = Company(name="Old Co", domain="old.com", source="manual")
    db_session.add(co)
    await db_session.flush()
    older = Contact(company_id=co.id, first_name="Zed", last_name="Older")
    db_session.add(older)
    await db_session.commit()
    c = await _doc_campaign(db_session, [_cand("c1", "Old Co", "old.com", ("Priya", "Shah", None))])
    hunter = _FakeHunter(finds={"old.com": "priya@old.com"})
    monkeypatch.setattr(hunter_verification, "get_hunter_service", lambda: hunter)

    await svc._seed_from_documents(db_session, c)
    await db_session.refresh(older)
    assert older.email is None
    named = (await db_session.execute(select(Contact).where(
        Contact.company_id == co.id, Contact.first_name == "Priya"))).scalars().one()
    assert named.email == "priya@old.com"


@pytest.mark.parametrize("address, own", [
    ("sumaninster18@gmail.com", True),
    ("suman.inster18@gmail.com", True),          # Gmail ignores dots
    ("Suman.Inster18+test@googlemail.com", True),  # …and +tags, and the alias domain
    ("someone.else@gmail.com", False),           # a free-mail sender doesn't own gmail.com
])
def test_canonical_address_matches_the_sending_mailbox(address, own):
    assert (svc.canonical_address(address) == svc.canonical_address("sumaninster18@gmail.com")) is own


@pytest.mark.asyncio
async def test_the_orgs_own_addresses_are_never_queued(db_session):
    c = await _campaign(db_session)
    db_session.add(_gmail(email="sumaninster18@gmail.com"))
    db_session.add(_gmail(email="founder@ourco.com"))
    await db_session.commit()
    # Free-mail contacts share a domain-less company: gmail.com is not one firm.
    freemail = Company(name="Personal addresses", source="manual")
    db_session.add(freemail)
    await db_session.flush()
    self_gmail = Contact(company_id=freemail.id, first_name="Suman", email="suman.inster18@gmail.com")
    other_gmail = Contact(company_id=freemail.id, first_name="Lead", email="lead@gmail.com")
    db_session.add_all([self_gmail, other_gmail])
    await db_session.commit()
    colleague = await _contact(db_session, "colleague@ourco.com")
    stranger = await _contact(db_session, "lead@prospect.com")

    queued = await svc.queue_contacts(
        db_session, c, [self_gmail, colleague, stranger, other_gmail], source="document")
    assert queued == 2
    rows = (await db_session.execute(select(CampaignProspect.to_email).where(
        CampaignProspect.campaign_id == c.id))).scalars().all()
    assert sorted(rows) == ["lead@gmail.com", "lead@prospect.com"]


@pytest.mark.asyncio
async def test_a_users_work_domain_is_excluded_too(db_session, make_user):
    c = await _campaign(db_session)
    await make_user("rep@ourteam.io")
    teammate = await _contact(db_session, "teammate@ourteam.io")
    assert await svc.queue_contacts(db_session, c, [teammate], source="document") == 0


@pytest.mark.asyncio
async def test_no_web_first_name_only_uses_domain_search_and_replaces_the_invented_name(
    db_session, monkeypatch
):
    co, ct = await _named(db_session, "real.com", first="David", last=None)
    hunter = _FakeHunter(people={"real.com": [
        {"email": "maria.lopez@real.com", "type": "personal", "score": 92,
         "first_name": "Maria", "last_name": "Lopez", "position": "CEO"},
    ]})
    _, queued = await _run_search(db_session, monkeypatch, hunter, [co], grounded=False)
    assert queued == 1
    assert hunter.domain_calls == ["real.com"] and hunter.find_calls == []
    await db_session.refresh(ct)
    assert (ct.first_name, ct.last_name, ct.title, ct.email) == ("Maria", "Lopez", "CEO", "maria.lopez@real.com")
    assert ct.email_verification_meta["email_origin"] == "hunter_domain_search"


@pytest.mark.asyncio
async def test_no_web_domain_search_finding_nobody_guesses_nothing(db_session, monkeypatch):
    co, ct = await _named(db_session, "empty.com", first="Sarah", last=None)
    hunter = _FakeHunter()
    _, queued = await _run_search(db_session, monkeypatch, hunter, [co], grounded=False)
    assert queued == 0
    await db_session.refresh(ct)
    assert ct.email is None  # no sarah@empty.com off an invented first name
    assert hunter.verify_calls == []


@pytest.mark.asyncio
async def test_grounded_or_full_name_results_still_use_the_finder(db_session, monkeypatch):
    co1, _ = await _named(db_session, "web.com", first="David", last=None)
    hunter = _FakeHunter(finds={"web.com": "david@web.com"})
    await _run_search(db_session, monkeypatch, hunter, [co1], grounded=True)
    assert hunter.domain_calls == []

    co2, _ = await _named(db_session, "full.com", first="Priya", last="Shah")
    hunter = _FakeHunter(finds={"full.com": "priya@full.com"})
    await _run_search(db_session, monkeypatch, hunter, [co2], grounded=False)
    assert hunter.domain_calls == [] and hunter.find_calls == ["full.com"]


@pytest.mark.asyncio
async def test_a_domain_with_no_mail_server_spends_no_hunter_credit(db_session, monkeypatch):
    from app.services import email_validation

    monkeypatch.setattr(email_validation, "check_mx",
                        lambda domain: "invalid" if domain == "dead.com" else "valid")
    dead, dead_ct = await _named(db_session, "dead.com")
    live, _ = await _named(db_session, "live.com", first="Ana", last="Li")
    hunter = _FakeHunter(finds={"live.com": "ana@live.com"})
    _, queued = await _run_search(db_session, monkeypatch, hunter, [dead, live])
    assert hunter.find_calls == ["live.com"]
    assert queued == 1
    await db_session.refresh(dead_ct)
    assert dead_ct.email is None  # no guess either: it would be a certain bounce


@pytest.mark.asyncio
async def test_an_mx_lookup_that_cant_complete_does_not_skip_the_domain(db_session, monkeypatch):
    from app.services import email_validation

    monkeypatch.setattr(email_validation, "check_mx", lambda domain: "unknown")
    co, _ = await _named(db_session, "maybe.com")
    hunter = _FakeHunter(finds={"maybe.com": "priya@maybe.com"})
    await _run_search(db_session, monkeypatch, hunter, [co])
    assert hunter.find_calls == ["maybe.com"]


@pytest.mark.parametrize("to_email", [
    "ceo@ourco.com",              # the org's work domain — the Partner Brief case
    "suman.inster18@gmail.com",   # the free-mail sending mailbox, under Gmail's dots
])
@pytest.mark.asyncio
async def test_an_own_address_queued_before_the_filter_is_refused_at_send_time(
    db_session, monkeypatch, to_email
):
    """queue_contacts refuses these now; a row approved before it did — the org's
    own CEO, due Monday 09:00 — must still never reach the wire."""
    c = await _campaign(db_session)
    db_session.add_all([_gmail(email="founder@ourco.com"), _gmail(email="sumaninster18@gmail.com")])
    freemail = Company(name="Personal addresses", source="manual")
    db_session.add(freemail)
    await db_session.flush()
    ct = Contact(company_id=freemail.id, first_name="Own", email=to_email,
                 email_verification_result="deliverable", email_verified_at=MIDWEEK)
    db_session.add(ct)
    await db_session.flush()
    p = CampaignProspect(campaign_id=c.id, contact_id=ct.id, to_email=to_email,
                         status="approved", email_subject="Hi", email_body="Hello")
    db_session.add(p)
    await db_session.commit()
    sent = []

    async def fake_send(db, **kw):
        sent.append(kw)
        return {"status": "sent"}

    monkeypatch.setattr(email_dispatch, "send_outreach_email", fake_send)
    out = await svc.send_prospect(db_session, p.id, MIDWEEK)
    assert out.status == "skipped" and sent == []
    await db_session.refresh(p)
    assert p.status == "skipped" and "own address" in p.skip_reason
