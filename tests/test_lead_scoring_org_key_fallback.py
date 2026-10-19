"""Lead scoring falls back to the org's own LLM keys when the platform's are spent."""

import uuid
from types import SimpleNamespace

import pytest

from app.services import org_gemini
from app.services.ai_service import AIService
from app.tasks import enrichment

ORG = uuid.uuid4()
COMPANY = SimpleNamespace(
    name="Acme Plastics", industry_segment="plastics", employee_count=120,
    city="Toronto", state_province="ON", country="CA",
)
AI = SimpleNamespace(lead_score_prompt=AIService.lead_score_prompt)


@pytest.fixture(autouse=True)
def _bound_org(monkeypatch):
    monkeypatch.setattr("app.org_context.get_current_org_id", lambda: ORG)


@pytest.mark.asyncio
async def test_org_keys_answer_when_platform_scoring_degraded(monkeypatch):
    seen = {}

    async def fake_generate_json(db, org_id, *, system, prompt, schema, **kw):
        seen["org_id"], seen["prompt"] = org_id, prompt
        return schema(score=82, grade="A", reasoning="mid-size plastics", priority="high")

    monkeypatch.setattr(org_gemini, "generate_json", fake_generate_json)
    result = await enrichment._score_with_org_keys(None, COMPANY, AI)

    assert result["score"] == 82 and result["grade"] == "A"
    assert "degraded" not in result
    assert seen["org_id"] == ORG
    assert "Acme Plastics" in seen["prompt"]  # same prompt as the platform path


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [org_gemini.GeminiKeyMissing(), org_gemini.GeminiError("429 everywhere")])
async def test_no_org_answer_keeps_the_default(monkeypatch, exc):
    async def fail(*a, **kw):
        raise exc

    monkeypatch.setattr(org_gemini, "generate_json", fail)
    assert await enrichment._score_with_org_keys(None, COMPANY, AI) is None


@pytest.mark.asyncio
async def test_no_bound_org_skips_the_fallback(monkeypatch):
    monkeypatch.setattr("app.org_context.get_current_org_id", lambda: None)
    assert await enrichment._score_with_org_keys(None, COMPANY, AI) is None
