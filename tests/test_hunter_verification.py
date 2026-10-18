"""Tests for P3 — Hunter.io email verification.

Covers the properties that actually protect the sender:

* **Layering** — syntax and MX failures short-circuit without ever costing a
  Hunter credit; an MX result of *unknown* must NOT short-circuit (dnspython is
  an optional dep, and treating its absence as a stop would silently disable the
  whole capability).
* **Result mapping** — including the ``accept_all`` catch-all case, which Hunter
  cannot actually resolve and which must never reach auto-enroll.
* **Fail-open** — a missing key, an API error, a timeout, or an exhausted 429
  retry leaves the contact untouched and sendable. Never ``invalid``.
* **The credit gate blocks before the call**, not after.
* **Sweep selection** — idempotency, and never spending credits on contacts we
  could never email anyway.

All network is faked; no test touches Hunter or DNS.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.contact import Contact
from app.rate_limit import limiter
from app.services import hunter_verification
from app.services.hunter_verification import (
    _Budget,
    get_hunter_service,
)


# ─── fakes ───────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code=200, data=None, headers=None):
        self.status_code = status_code
        self._data = data if data is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._data


class _FakeClient:
    """Stands in for httpx.AsyncClient: records calls, replays scripted responses.

    ``responses`` is consumed left-to-right; the final entry repeats, so a single
    scripted response serves any number of calls.
    """

    calls: list = []
    responses: list = []
    raise_exc: Exception | None = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params=None, headers=None):
        _FakeClient.calls.append({"url": url, "params": params or {}, "headers": headers or {}})
        if _FakeClient.raise_exc is not None:
            raise _FakeClient.raise_exc
        if not _FakeClient.responses:
            return _FakeResponse(200, {"data": {}})
        if len(_FakeClient.responses) > 1:
            return _FakeClient.responses.pop(0)
        return _FakeClient.responses[0]


def _payload(result="deliverable", score=95, **flags):
    data = {
        "email": "ceo@acme.com",
        "result": result,
        "score": score,
        "accept_all": False,
        "disposable": False,
        "webmail": False,
        "gibberish": False,
        "smtp_check": True,
        "mx_records": True,
        "block": False,
        "status": "valid",
    }
    data.update(flags)
    return {"data": data}


@pytest.fixture(autouse=True)
def _fake_http(monkeypatch):
    _FakeClient.calls = []
    _FakeClient.responses = []
    _FakeClient.raise_exc = None
    monkeypatch.setattr(hunter_verification.httpx, "AsyncClient", _FakeClient)
    # Keep the retry backoff from making the suite slow.
    monkeypatch.setattr(hunter_verification, "_BACKOFF_BASE", 0.001)
    monkeypatch.setattr(hunter_verification, "_MAX_RETRY_AFTER", 0.01)
    yield
    _FakeClient.calls = []
    _FakeClient.responses = []
    _FakeClient.raise_exc = None


@pytest.fixture(autouse=True)
def _mx_valid(monkeypatch):
    """Default: MX passes, so tests reach the Hunter layer unless they say not to."""
    monkeypatch.setattr(hunter_verification, "check_mx", lambda d: "valid")


@pytest.fixture
def hunter_on(monkeypatch):
    """Configure the capability on. Endpoints build their own service instance
    from the cached Settings singleton, so patch that rather than an instance."""
    settings = get_settings()
    monkeypatch.setattr(settings, "hunter_api_key", "test-key")
    monkeypatch.setattr(settings, "hunter_base_url", "https://api.hunter.io/v2")
    monkeypatch.setattr(settings, "hunter_daily_credit_budget", 100)
    monkeypatch.setattr(settings, "hunter_min_score", 80)
    monkeypatch.setattr(settings, "hunter_concurrency", 5)
    monkeypatch.setattr(settings, "hunter_reverify_days", 90)
    monkeypatch.setattr(settings, "hunter_max_batch", 100)
    return settings


async def _mk_contact(db, company, **kw):
    contact = Contact(
        id=uuid.uuid4(),
        company_id=company.id,
        email=kw.pop("email", f"{uuid.uuid4().hex[:8]}@acme.com"),
        source="test",
        suppressed=kw.pop("suppressed", False),
        unsubscribed=kw.pop("unsubscribed", False),
        **kw,
    )
    db.add(contact)
    await db.commit()
    return contact


# ─── layering: the free checks must short-circuit before spending ────────────

@pytest.mark.asyncio
async def test_bad_syntax_never_calls_hunter(hunter_on):
    verdict = await get_hunter_service().verify_email("not-an-email")
    assert verdict["status"] == "ok"
    assert verdict["result"] == "undeliverable"
    assert verdict["source"] == "mx"
    assert verdict["email_status"] == "invalid"
    assert _FakeClient.calls == []  # the whole point: no credit spent


@pytest.mark.asyncio
async def test_mx_invalid_never_calls_hunter(hunter_on, monkeypatch):
    monkeypatch.setattr(hunter_verification, "check_mx", lambda d: "invalid")
    verdict = await get_hunter_service().verify_email("ceo@nomx.example")
    assert verdict["result"] == "undeliverable"
    assert verdict["source"] == "mx"
    assert verdict["email_status"] == "invalid"
    assert _FakeClient.calls == []


@pytest.mark.asyncio
async def test_mx_unknown_still_calls_hunter(hunter_on, monkeypatch):
    """An MX lookup we couldn't complete says nothing about the mailbox.

    dnspython is an optional dependency — if its absence short-circuited here,
    Hunter would never run at all on a host without it.
    """
    monkeypatch.setattr(hunter_verification, "check_mx", lambda d: "unknown")
    _FakeClient.responses = [_FakeResponse(200, _payload())]
    verdict = await get_hunter_service().verify_email("ceo@acme.com")
    assert len(_FakeClient.calls) == 1
    assert verdict["result"] == "deliverable"


# ─── result mapping ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deliverable_maps_to_valid(hunter_on):
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 97))]
    verdict = await get_hunter_service().verify_email("ceo@acme.com")
    assert (verdict["result"], verdict["score"]) == ("deliverable", 97)
    assert verdict["email_status"] == "valid"
    assert verdict["source"] == "hunter"


@pytest.mark.asyncio
async def test_undeliverable_maps_to_invalid(hunter_on):
    _FakeClient.responses = [_FakeResponse(200, _payload("undeliverable", 12))]
    verdict = await get_hunter_service().verify_email("nobody@acme.com")
    assert verdict["result"] == "undeliverable"
    assert verdict["email_status"] == "invalid"


@pytest.mark.asyncio
async def test_risky_is_sendable_never_invalid(hunter_on):
    _FakeClient.responses = [_FakeResponse(200, _payload("risky", 60))]
    verdict = await get_hunter_service().verify_email("maybe@acme.com")
    assert verdict["result"] == "risky"
    # Sendable by a human — 'unknown' is honest, 'invalid' would be a lie.
    assert verdict["email_status"] == "unknown"


@pytest.mark.asyncio
async def test_accept_all_downgrades_deliverable_to_risky(hunter_on):
    """A catch-all domain accepts everything at SMTP time and bounces later.

    Hunter genuinely cannot tell a real mailbox from a typo there, so a
    'deliverable' carrying accept_all must not be trusted as deliverable.
    """
    _FakeClient.responses = [
        _FakeResponse(200, _payload("deliverable", 90, accept_all=True))
    ]
    verdict = await get_hunter_service().verify_email("anything@catchall.com")
    assert verdict["result"] == "risky"
    assert verdict["meta"]["accept_all"] is True
    assert verdict["meta"]["hunter_result"] == "deliverable"  # raw answer preserved
    assert verdict["email_status"] == "unknown"


@pytest.mark.asyncio
async def test_unrecognized_result_is_unknown_not_invalid(hunter_on):
    _FakeClient.responses = [_FakeResponse(200, _payload("something-new", 50))]
    verdict = await get_hunter_service().verify_email("ceo@acme.com")
    assert verdict["result"] == "unknown"
    assert verdict["email_status"] == "unknown"


# ─── auto-enroll eligibility ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_accept_all_contact_is_not_auto_enrollable(
    db_session: AsyncSession, sample_company, hunter_on
):
    contact = await _mk_contact(db_session, sample_company, email="x@catchall.com")
    _FakeClient.responses = [
        _FakeResponse(200, _payload("deliverable", 95, accept_all=True))
    ]
    service = get_hunter_service()
    await service.verify_contact(db_session, contact)
    await db_session.commit()

    assert contact.email_verification_result == "risky"
    assert contact.email_status == "unknown"       # still sendable...
    assert service.is_auto_enroll_eligible(contact) is False  # ...but never automatically


@pytest.mark.asyncio
async def test_low_score_deliverable_sends_but_is_not_auto_enrollable(
    db_session: AsyncSession, sample_company, hunter_on
):
    contact = await _mk_contact(db_session, sample_company)
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 55))]
    service = get_hunter_service()
    await service.verify_contact(db_session, contact)

    assert contact.email_status == "valid"                     # sendable
    assert service.is_auto_enroll_eligible(contact) is False   # below hunter_min_score


@pytest.mark.asyncio
async def test_high_score_deliverable_is_auto_enrollable(
    db_session: AsyncSession, sample_company, hunter_on
):
    contact = await _mk_contact(db_session, sample_company)
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 95))]
    service = get_hunter_service()
    await service.verify_contact(db_session, contact)
    assert service.is_auto_enroll_eligible(contact) is True


@pytest.mark.asyncio
async def test_suppressed_contact_is_never_auto_enrollable(
    db_session: AsyncSession, sample_company, hunter_on
):
    contact = await _mk_contact(db_session, sample_company, suppressed=True)
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 99))]
    service = get_hunter_service()
    await service.verify_contact(db_session, contact)
    assert service.is_auto_enroll_eligible(contact) is False


# ─── fail-open ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_api_key_persists_nothing_and_leaves_contact_sendable(
    db_session: AsyncSession, sample_company, monkeypatch
):
    monkeypatch.setattr(get_settings(), "hunter_api_key", "")
    contact = await _mk_contact(db_session, sample_company, email_status="valid")

    verdict = await get_hunter_service().verify_contact(db_session, contact)

    assert verdict["status"] == "skipped"
    assert verdict["reason"] == "no_api_key"
    assert _FakeClient.calls == []
    assert contact.email_status == "valid"          # untouched
    assert contact.email_verified_at is None        # so the sweep retries it


@pytest.mark.asyncio
async def test_api_error_is_fail_open(db_session: AsyncSession, sample_company, hunter_on):
    _FakeClient.responses = [_FakeResponse(400, {"errors": [{"details": "bad"}]})]
    contact = await _mk_contact(db_session, sample_company, email_status="valid")

    verdict = await get_hunter_service().verify_contact(db_session, contact)

    assert verdict["status"] == "error"
    assert contact.email_status == "valid"       # NEVER downgraded to invalid
    assert contact.email_verified_at is None
    assert contact.email_verification_result is None


@pytest.mark.asyncio
async def test_timeout_is_fail_open(db_session: AsyncSession, sample_company, hunter_on):
    _FakeClient.raise_exc = TimeoutError("timed out")
    contact = await _mk_contact(db_session, sample_company, email_status="valid")

    verdict = await get_hunter_service().verify_contact(db_session, contact)

    assert verdict["status"] == "error"
    assert contact.email_status == "valid"
    assert contact.email_verified_at is None


@pytest.mark.asyncio
async def test_rate_limit_retries_then_succeeds(hunter_on):
    _FakeClient.responses = [
        _FakeResponse(429, {}, {"Retry-After": "0.001"}),
        _FakeResponse(200, _payload("deliverable", 91)),
    ]
    verdict = await get_hunter_service().verify_email("ceo@acme.com")
    assert len(_FakeClient.calls) == 2
    assert verdict["result"] == "deliverable"


@pytest.mark.asyncio
async def test_rate_limit_exhausted_is_error_not_invalid(hunter_on):
    _FakeClient.responses = [_FakeResponse(429, {}, {"Retry-After": "0.001"})]
    verdict = await get_hunter_service().verify_email("ceo@acme.com")
    assert len(_FakeClient.calls) == hunter_verification._MAX_ATTEMPTS
    assert verdict["status"] == "error"
    assert verdict["reason"] == "rate_limited"   # distinct from a real verdict
    assert verdict["email_status"] is None       # nothing to persist


@pytest.mark.asyncio
async def test_server_error_retries(hunter_on):
    _FakeClient.responses = [
        _FakeResponse(503, {}),
        _FakeResponse(200, _payload("deliverable", 88)),
    ]
    verdict = await get_hunter_service().verify_email("ceo@acme.com")
    assert len(_FakeClient.calls) == 2
    assert verdict["result"] == "deliverable"


@pytest.mark.asyncio
async def test_client_error_does_not_retry(hunter_on):
    """A 401/404 won't improve on retry — don't spend the attempts."""
    _FakeClient.responses = [_FakeResponse(401, {})]
    verdict = await get_hunter_service().verify_email("ceo@acme.com")
    assert len(_FakeClient.calls) == 1
    assert verdict["status"] == "error"
    assert verdict["reason"] == "http_401"


# ─── credit gate ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_budget_gate_blocks_before_the_call(hunter_on):
    """The single most important test here: an exhausted budget must not spend."""
    budget = _Budget(0)
    verdict = await get_hunter_service().verify_email("ceo@acme.com", budget=budget)
    assert verdict["status"] == "skipped"
    assert verdict["reason"] == "budget_exhausted"
    assert _FakeClient.calls == []  # no HTTP happened at all


@pytest.mark.asyncio
async def test_budget_is_shared_and_drawn_down_across_a_batch(
    db_session: AsyncSession, sample_company, hunter_on
):
    contacts = [await _mk_contact(db_session, sample_company) for _ in range(5)]
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 90))]

    budget = _Budget(2)
    results = await get_hunter_service().verify_batch(
        db_session, contacts, budget=budget
    )

    assert len(_FakeClient.calls) == 2  # only what the budget could pay for
    assert sum(1 for r in results if r["status"] == "ok") == 2
    assert sum(1 for r in results if r["reason"] == "budget_exhausted") == 3


@pytest.mark.asyncio
async def test_free_layers_do_not_consume_budget(hunter_on, monkeypatch):
    """A syntax failure costs nothing, so it must not draw down the allowance."""
    budget = _Budget(1)
    await get_hunter_service().verify_email("garbage", budget=budget)
    assert budget.remaining == 1


@pytest.mark.asyncio
async def test_budget_cap_handles_unlimited():
    assert _Budget(float("inf")).cap(200) == 200   # int(inf) would raise
    assert _Budget(5).cap(200) == 5
    assert _Budget(0).cap(200) == 0


@pytest.mark.asyncio
async def test_make_budget_counts_only_todays_hunter_calls(
    db_session: AsyncSession, sample_company, hunter_on
):
    now = datetime.now(timezone.utc)
    # Counts: verified by Hunter today.
    await _mk_contact(
        db_session, sample_company,
        email_verification_source="hunter", email_verified_at=now,
    )
    # Doesn't count: yesterday.
    await _mk_contact(
        db_session, sample_company,
        email_verification_source="hunter", email_verified_at=now - timedelta(days=1),
    )
    # Doesn't count: free MX layer, no credit spent.
    await _mk_contact(
        db_session, sample_company,
        email_verification_source="mx", email_verified_at=now,
    )

    service = get_hunter_service()
    assert await service.hunter_calls_today(db_session) == 1
    budget = await service.make_budget(db_session)
    assert budget.remaining == 99  # 100 - 1


@pytest.mark.asyncio
async def test_zero_budget_setting_means_unlimited(
    db_session: AsyncSession, hunter_on, monkeypatch
):
    monkeypatch.setattr(get_settings(), "hunter_daily_credit_budget", 0)
    budget = await get_hunter_service().make_budget(db_session)
    assert budget.remaining == float("inf")
    assert budget.exhausted is False


# ─── persistence + the send gate ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_undeliverable_sets_the_gate_every_send_path_reads(
    db_session: AsyncSession, sample_company, hunter_on
):
    """``email_status='invalid'`` is the single sendability gate.

    Audited call sites that skip on it: ``app/tasks/outreach.py`` (both
    ``send_sequence_step`` and the ``process_outreach_queue`` dispatch filter)
    and ``app/routers/sequences.py``. Hunter's job is to put the right value
    here; those paths already honour it.
    """
    contact = await _mk_contact(db_session, sample_company)
    _FakeClient.responses = [_FakeResponse(200, _payload("undeliverable", 8))]

    await get_hunter_service().verify_contact(db_session, contact)
    await db_session.commit()

    assert contact.email_status == "invalid"
    assert contact.email_verification_result == "undeliverable"
    assert contact.email_verification_source == "hunter"
    assert contact.email_verified_at is not None
    assert contact.email_checked_at is not None


@pytest.mark.asyncio
async def test_verification_meta_is_recorded(
    db_session: AsyncSession, sample_company, hunter_on
):
    contact = await _mk_contact(db_session, sample_company)
    _FakeClient.responses = [
        _FakeResponse(200, _payload("risky", 40, disposable=True, webmail=True))
    ]
    await get_hunter_service().verify_contact(db_session, contact)

    meta = contact.email_verification_meta
    assert meta["disposable"] is True
    assert meta["webmail"] is True
    assert meta["smtp_check"] is True


@pytest.mark.asyncio
async def test_one_failure_does_not_sink_the_batch(
    db_session: AsyncSession, sample_company, hunter_on, monkeypatch
):
    good = await _mk_contact(db_session, sample_company, email="good@acme.com")
    bad = await _mk_contact(db_session, sample_company, email="garbage")  # syntax fail
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 92))]

    results = await get_hunter_service().verify_batch(db_session, [good, bad])

    assert len(results) == 2
    assert good.email_status == "valid"
    assert bad.email_status == "invalid"


# ─── sweep selection ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_select_pending_is_idempotent(
    db_session: AsyncSession, sample_company, hunter_on
):
    """Re-running the sweep must not re-verify what it already verified."""
    fresh = await _mk_contact(db_session, sample_company)
    await _mk_contact(
        db_session, sample_company, email_verified_at=datetime.now(timezone.utc)
    )

    pending = await get_hunter_service().select_pending(db_session, 50)
    assert [c.id for c in pending] == [fresh.id]


@pytest.mark.asyncio
async def test_select_pending_skips_contacts_we_could_never_email(
    db_session: AsyncSession, sample_company, hunter_on
):
    keep = await _mk_contact(db_session, sample_company)
    await _mk_contact(db_session, sample_company, suppressed=True)
    await _mk_contact(db_session, sample_company, unsubscribed=True)
    await _mk_contact(db_session, sample_company, email=None)

    pending = await get_hunter_service().select_pending(db_session, 50)
    assert [c.id for c in pending] == [keep.id]


@pytest.mark.asyncio
async def test_select_stale_respects_the_reverify_window(
    db_session: AsyncSession, sample_company, hunter_on
):
    now = datetime.now(timezone.utc)
    stale = await _mk_contact(
        db_session, sample_company,
        email_verified_at=now - timedelta(days=120), email_status="valid",
    )
    await _mk_contact(  # verified recently — not due
        db_session, sample_company,
        email_verified_at=now - timedelta(days=10), email_status="valid",
    )

    due = await get_hunter_service().select_stale(db_session, 50)
    assert [c.id for c in due] == [stale.id]


@pytest.mark.asyncio
async def test_select_stale_skips_known_bad_addresses(
    db_session: AsyncSession, sample_company, hunter_on
):
    """We will never email an invalid address, so re-verifying it is pure burn."""
    now = datetime.now(timezone.utc)
    await _mk_contact(
        db_session, sample_company,
        email_verified_at=now - timedelta(days=200), email_status="invalid",
    )
    keep = await _mk_contact(
        db_session, sample_company,
        email_verified_at=now - timedelta(days=200), email_status="valid",
    )

    due = await get_hunter_service().select_stale(db_session, 50)
    assert [c.id for c in due] == [keep.id]


@pytest.mark.asyncio
async def test_select_stale_is_oldest_first(
    db_session: AsyncSession, sample_company, hunter_on
):
    """A capped run should spend its credits on the rottenest addresses."""
    now = datetime.now(timezone.utc)
    newer = await _mk_contact(
        db_session, sample_company,
        email_verified_at=now - timedelta(days=100), email_status="valid",
    )
    oldest = await _mk_contact(
        db_session, sample_company,
        email_verified_at=now - timedelta(days=365), email_status="valid",
    )

    due = await get_hunter_service().select_stale(db_session, 50)
    assert [c.id for c in due] == [oldest.id, newer.id]


# ─── email finder ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_find_email_recovers_a_null_email_contact(
    db_session: AsyncSession, sample_company, hunter_on
):
    """Apollo routinely returns a person with email=None, which find_contacts
    inserts un-emailable. Hunter's finder turns some of those back into leads."""
    _FakeClient.responses = [
        _FakeResponse(200, {"data": {"email": "jane.doe@acme.com", "score": 92}})
    ]
    found = await get_hunter_service().find_email(
        db_session, sample_company, "Jane", "Doe"
    )
    assert found == "jane.doe@acme.com"
    assert _FakeClient.calls[0]["params"]["first_name"] == "Jane"


@pytest.mark.asyncio
async def test_find_email_needs_a_domain_and_a_name(
    db_session: AsyncSession, sample_company, hunter_on
):
    assert await get_hunter_service().find_email(db_session, sample_company, "Jane", None) is None
    assert _FakeClient.calls == []


@pytest.mark.asyncio
async def test_find_email_is_fail_open(
    db_session: AsyncSession, sample_company, hunter_on
):
    _FakeClient.raise_exc = TimeoutError("nope")
    found = await get_hunter_service().find_email(
        db_session, sample_company, "Jane", "Doe"
    )
    assert found is None


# ─── endpoints ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_endpoint(
    client: AsyncClient, db_session: AsyncSession, sample_company, hunter_on
):
    contact = await _mk_contact(db_session, sample_company)
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 96))]

    resp = await client.post(f"/api/v1/dialtone/contacts/{contact.id}/verify")

    assert resp.status_code == 200
    body = resp.json()
    assert body["result"] == "deliverable"
    assert body["score"] == 96
    assert body["email_status"] == "valid"
    assert body["auto_enroll_eligible"] is True


@pytest.mark.asyncio
async def test_verify_endpoint_404_for_unknown_contact(client: AsyncClient, hunter_on):
    resp = await client.post(f"/api/v1/dialtone/contacts/{uuid.uuid4()}/verify")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_verify_endpoint_is_200_when_hunter_is_down(
    client: AsyncClient, db_session: AsyncSession, sample_company, hunter_on
):
    """An upstream outage is not the contact's fault — fail open, don't 5xx."""
    contact = await _mk_contact(db_session, sample_company, email_status="valid")
    _FakeClient.raise_exc = TimeoutError("hunter is down")

    resp = await client.post(f"/api/v1/dialtone/contacts/{contact.id}/verify")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["email_status"] == "valid"  # still sendable


@pytest.mark.asyncio
async def test_bulk_verify_endpoint(
    client: AsyncClient, db_session: AsyncSession, sample_company, hunter_on
):
    a = await _mk_contact(db_session, sample_company, email="a@acme.com")
    b = await _mk_contact(db_session, sample_company, email="b@acme.com")
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 90))]

    resp = await client.post(
        "/api/v1/dialtone/contacts/bulk-verify",
        json={"contact_ids": [str(a.id), str(b.id), str(uuid.uuid4())]},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["requested"] == 3
    assert body["found"] == 2       # the unknown id is ignored, not fatal
    assert body["verified"] == 2
    assert body["breakdown"]["deliverable"] == 2


@pytest.mark.asyncio
async def test_bulk_verify_rejects_an_oversized_batch(
    client: AsyncClient, hunter_on, monkeypatch
):
    monkeypatch.setattr(get_settings(), "hunter_max_batch", 2)
    resp = await client.post(
        "/api/v1/dialtone/contacts/bulk-verify",
        json={"contact_ids": [str(uuid.uuid4()) for _ in range(3)]},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_verify_endpoint_succeeds_with_the_limiter_live(
    client: AsyncClient, db_session: AsyncSession, sample_company, hunter_on
):
    """Both POSTs here are @limiter.limit'd, and conftest disables the limiter
    globally — so without this the success path is never exercised with it on.

    That exact blind spot shipped a 500 on successful /auth/login (fixed in
    0cf90cd): slowapi's header injection reaches for a ``response: Response``
    kwarg these endpoints don't declare, so it broke only the *successful* calls
    while failures kept returning cleanly.
    """
    contact = await _mk_contact(db_session, sample_company)
    _FakeClient.responses = [_FakeResponse(200, _payload("deliverable", 93))]

    limiter.reset()
    limiter.enabled = True
    try:
        resp = await client.post(f"/api/v1/dialtone/contacts/{contact.id}/verify")
        assert resp.status_code == 200, resp.text
        assert resp.json()["result"] == "deliverable"

        bulk = await client.post(
            "/api/v1/dialtone/contacts/bulk-verify",
            json={"contact_ids": [str(contact.id)]},
        )
        assert bulk.status_code == 200, bulk.text
    finally:
        limiter.enabled = False
        limiter.reset()


@pytest.mark.asyncio
async def test_verification_stats_endpoint(
    client: AsyncClient, db_session: AsyncSession, sample_company, hunter_on
):
    now = datetime.now(timezone.utc)
    await _mk_contact(
        db_session, sample_company, email="good@acme.com", email_status="valid",
        email_verification_result="deliverable", email_verification_score=95,
        email_verification_source="hunter", email_verified_at=now,
    )
    await _mk_contact(
        db_session, sample_company, email="dead@acme.com", email_status="invalid",
        email_verification_result="undeliverable", email_verification_score=5,
        email_verification_source="hunter", email_verified_at=now,
    )
    await _mk_contact(db_session, sample_company, email="new@acme.com")  # unverified

    resp = await client.get("/api/v1/dialtone/contacts/verification-stats")

    assert resp.status_code == 200
    body = resp.json()
    assert body["emailable"] == 3
    assert body["verified"] == 2
    assert body["unverified"] == 1
    assert body["by_result"]["deliverable"] == 1
    assert body["by_result"]["undeliverable"] == 1
    assert body["by_result"]["unchecked"] == 1
    assert body["auto_enroll_eligible"] == 1
    assert body["bounces_prevented"] == 1
    assert body["verified_undeliverable_rate"] == 0.5  # 1 of 2 verified
    assert body["hunter_enabled"] is True


@pytest.mark.asyncio
async def test_background_sweeps_can_be_switched_off(monkeypatch):
    """A free plan's allowance must not go to the backlog before anything mails it."""
    from app.config import Settings, get_settings
    from app.tasks import verification

    assert Settings.model_fields["hunter_background_sweeps"].default is False

    def _boom(*a, **kw):
        raise AssertionError("sweep must not touch Hunter when switched off")

    monkeypatch.setattr(get_settings(), "hunter_background_sweeps", False)
    monkeypatch.setattr(verification, "get_hunter_service", _boom)
    out = await verification._sweep(_boom, 200, "verify_pending_contacts")
    assert out == {"status": "skipped", "reason": "sweeps_disabled", "checked": 0}


@pytest.mark.asyncio
async def test_the_api_key_travels_in_a_header_never_the_url(hunter_on):
    """httpx logs request URLs at INFO; a key in the query string is a key in the logs."""
    _FakeClient.responses = [_FakeResponse(200, _payload())]
    await get_hunter_service().verify_email("ana@acme.com")
    (call,) = _FakeClient.calls
    assert call["headers"] == {"X-API-KEY": "test-key"}
    assert "test-key" not in call["url"]
    assert "test-key" not in str(call["params"]) and "api_key" not in call["params"]
