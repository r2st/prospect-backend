"""Jev lead temperature (hot / warm / cold) on the lead scoring path."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

pytest.importorskip("typesafe_sdk")

from app.models.company import Company  # noqa: E402
from app.models.lead_score import LeadScore  # noqa: E402
from app.services import jev_service  # noqa: E402
from app.services.jev_scoring import build_lead_state, jev_lead_temperature  # noqa: E402
from app.services.scoring import score_company_rule_based  # noqa: E402
from tests import jev_mock  # noqa: E402
from tests.jev_mock import choice_answer  # noqa: E402

HOT = {"lead_temperature": choice_answer("hot", 0.81, {"hot": 0.81, "warm": 0.14, "cold": 0.05})}


@pytest.fixture(autouse=True)
def _reset():
    jev_service.reset_cooldown()
    yield
    jev_service.reset_cooldown()


# ─── State ───────────────────────────────────────────────────────────────────


def test_state_carries_lead_data_but_not_the_rule_based_score(sample_company, sample_contact):
    state = build_lead_state(sample_company, [sample_contact], signals_in_24h=3)
    assert sample_company.name in state
    assert sample_contact.full_name in state
    assert (sample_contact.title or "no title") in state
    assert "Engagement events in the last 24 hours: 3" in state
    assert "fit" not in state.lower()


def test_state_handles_sparse_company():
    company = Company(name="Bare Co", country="US", source="manual", stage="new", tags=[])
    state = build_lead_state(company, [])
    assert "Contacts: none on file" in state
    assert "Employees: unknown" in state
    assert "Most recent engagement: never" in state


def test_state_lists_intent_signals(sample_company):
    state = build_lead_state(
        sample_company, [],
        intent_breakdown={"active_signals": 2, "types": ["funding", "hiring"]},
    )
    assert "Active buyer-intent signals: 2 (funding, hiring)" in state


async def test_jev_lead_temperature_returns_factor_dict(monkeypatch, sample_company):
    mock = jev_mock.install(monkeypatch)
    mock.answers = HOT
    out = await jev_lead_temperature(sample_company, [])
    assert out["label"] == "hot"
    assert out["probabilities"]["cold"] == pytest.approx(0.05)


# ─── score_company_rule_based ────────────────────────────────────────────────


async def test_rule_based_score_records_jev_temperature(
    monkeypatch, db_session: AsyncSession, sample_company, sample_contact
):
    mock = jev_mock.install(monkeypatch)
    mock.answers = HOT

    score = await score_company_rule_based(db_session, sample_company)

    assert score.factors["jev_temperature"]["label"] == "hot"
    assert score.factors["jev_temperature"]["confidence"] == pytest.approx(0.81)
    assert mock.calls == 1
    assert sample_contact.full_name in mock.requests[0]["state"]


async def test_jev_does_not_move_the_fit_score(
    monkeypatch, db_session: AsyncSession, sample_company, sample_contact
):
    baseline = await score_company_rule_based(db_session, sample_company)

    mock = jev_mock.install(monkeypatch)
    mock.answers = {"lead_temperature": choice_answer("cold", 0.99)}
    with_jev = await score_company_rule_based(db_session, sample_company)

    assert with_jev.fit_score == baseline.fit_score
    assert with_jev.factors["jev_temperature"]["label"] == "cold"


async def test_unconfigured_jev_leaves_factors_unchanged(
    db_session: AsyncSession, sample_company
):
    score = await score_company_rule_based(db_session, sample_company)
    assert "jev_temperature" not in score.factors


async def test_jev_failure_still_scores(
    monkeypatch, db_session: AsyncSession, sample_company
):
    mock = jev_mock.install(monkeypatch)
    mock.status = 503
    score = await score_company_rule_based(db_session, sample_company)
    assert score.fit_score >= 0
    assert "jev_temperature" not in score.factors


# ─── Lead create / update flow ───────────────────────────────────────────────


async def _scores_for(db: AsyncSession, name: str) -> list[LeadScore]:
    company = (
        await db.execute(select(Company).where(Company.name == name))
    ).scalars().first()
    rows = await db.execute(select(LeadScore).where(LeadScore.company_id == company.id))
    return list(rows.scalars().all())


async def test_creating_a_company_scores_it_with_jev(
    monkeypatch, client: AsyncClient, db_session: AsyncSession
):
    mock = jev_mock.install(monkeypatch)
    mock.answers = HOT

    resp = await client.post("/api/v1/companies/", json={
        "name": "Jev Stamping Co", "country": "US", "source": "manual",
        "naics_code": "3321", "employee_count": 120,
    })
    assert resp.status_code == 201

    scores = await _scores_for(db_session, "Jev Stamping Co")
    assert len(scores) == 1
    assert scores[0].factors["jev_temperature"]["label"] == "hot"
    assert "Jev Stamping Co" in mock.requests[0]["state"]


async def test_update_rescores_only_when_lead_data_changes(
    monkeypatch, client: AsyncClient, db_session: AsyncSession
):
    mock = jev_mock.install(monkeypatch)
    mock.answers = HOT
    resp = await client.post("/api/v1/companies/", json={
        "name": "Rescore Co", "country": "US", "source": "manual",
    })
    company_id = resp.json()["id"]
    assert len(await _scores_for(db_session, "Rescore Co")) == 1

    # Notes don't change the lead — no new score row, no Jev call.
    resp = await client.patch(f"/api/v1/companies/{company_id}", json={"notes": "called"})
    assert resp.status_code == 200
    assert len(await _scores_for(db_session, "Rescore Co")) == 1
    assert mock.calls == 1

    # Firmographics do.
    resp = await client.patch(f"/api/v1/companies/{company_id}", json={"employee_count": 300})
    assert resp.status_code == 200
    assert len(await _scores_for(db_session, "Rescore Co")) == 2
    assert mock.calls == 2
    assert "Employees: 300" in mock.requests[1]["state"]


async def test_create_succeeds_when_scoring_raises(
    monkeypatch, client: AsyncClient
):
    async def _boom(*a, **k):
        raise RuntimeError("scorer exploded")

    monkeypatch.setattr("app.services.scoring.score_company_rule_based", _boom)
    resp = await client.post("/api/v1/companies/", json={
        "name": "Resilient Co", "country": "US", "source": "manual",
    })
    assert resp.status_code == 201
    assert resp.json()["name"] == "Resilient Co"
