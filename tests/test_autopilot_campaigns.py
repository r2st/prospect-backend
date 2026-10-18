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

    monkeypatch.setattr(svc, "build_icp", fake_icp)
    resp = await client.post("/api/v1/campaigns", json={
        "name": "Q4 push", "document_ids": [str(doc.id)], "daily_send_limit": 40,
    })
    assert resp.status_code == 201, resp.text
    body = resp.json()
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

    def __init__(self, *, enabled=True, finds=None, verdicts=None):
        self.enabled = enabled
        self.finds = finds or {}          # domain -> email
        self.verdicts = verdicts or {}    # email -> result
        self.find_calls, self.verify_calls = [], []

    async def make_budget(self, db):
        return None

    async def find_email(self, db, company, first, last, *, budget=None):
        self.find_calls.append(company.domain)
        return self.finds.get(company.domain)

    async def verify_contact(self, db, contact, *, budget=None):
        self.verify_calls.append(contact.email)
        result = self.verdicts.get(contact.email, "deliverable")
        contact.email_verification_result = result
        contact.email_verified_at = MIDWEEK
        contact.email_verification_meta = {"hunter_result": result}
        return {"status": "ok", "result": result}


async def _run_search(db_session, monkeypatch, hunter, companies):
    from types import SimpleNamespace

    from app.services import gemini_discovery, hunter_verification, ingestion

    async def fake_search(*a, **kw):
        return SimpleNamespace(records=[], provider="groq")

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
