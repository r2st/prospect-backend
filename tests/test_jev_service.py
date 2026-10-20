"""The shared Jev (TypeSafe System One) client.

Jev is advisory everywhere it is used, so the contract under test is mostly
about what happens when it *can't* answer: unconfigured, failing, slow, or
off-vocabulary all come back as ``None`` and never raise.
"""

import pytest

pytest.importorskip("typesafe_sdk")

from app.config import get_settings  # noqa: E402
from app.services import jev_service  # noqa: E402
from tests import jev_mock  # noqa: E402
from tests.jev_mock import choice_answer, noul_answer  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    jev_service.reset_cooldown()
    yield
    jev_service.reset_cooldown()


async def test_unconfigured_returns_none_without_network(monkeypatch):
    mock = jev_mock.install(monkeypatch, api_key="")
    assert jev_service.is_configured() is False
    assert await jev_service.classify_lead_temperature("Acme, 200 employees") is None
    assert await jev_service.spam_probability("hi", "body") is None
    assert mock.calls == 0


async def test_lead_temperature_sends_choice_question_and_parses(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {
        "lead_temperature": choice_answer(
            "hot", 0.82, {"hot": 0.82, "warm": 0.15, "cold": 0.03}
        )
    }

    result = await jev_service.classify_lead_temperature("Acme Stamping, 200 employees")

    assert result is not None
    assert result.label == "hot"
    assert result.confidence == pytest.approx(0.82)
    assert result.probabilities["warm"] == pytest.approx(0.15)
    assert result.model == "jev-test"

    body = mock.requests[0]
    assert body["model"] == "jev-latest"
    assert "Acme Stamping" in body["state"]
    q = body["questions"]["lead_temperature"]
    assert q["type"] == "choice"
    assert set(q["criteria"]) == {"hot", "warm", "cold"}


async def test_reply_intent_offers_routing_categories(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"reply_intent": choice_answer("interested", 0.9)}

    result = await jev_service.classify_reply_intent(
        "Re: intro", "Sure, send me some times.", from_email="dana@acme.com"
    )

    assert result.label == "interested"
    criteria = mock.requests[0]["questions"]["reply_intent"]["criteria"]
    assert set(criteria) == {
        "interested", "not_interested", "out_of_office", "bounce", "unsubscribe", "wrong_person",
    }
    assert "dana@acme.com" in mock.requests[0]["state"]


async def test_spam_probability_sends_noul_and_clamps(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.93)}

    p = await jev_service.spam_probability("ACT NOW", "<p>BUY NOW!!!</p>")

    assert p == pytest.approx(0.93)
    q = mock.requests[0]["questions"]["is_spam"]
    assert q["type"] == "noul"
    assert q["instructions"]
    assert "ACT NOW" in mock.requests[0]["state"]


async def test_out_of_vocabulary_label_is_discarded(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"lead_temperature": choice_answer("lukewarm", 0.99)}
    assert await jev_service.classify_lead_temperature("x") is None


async def test_missing_answer_is_none(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {}
    assert await jev_service.spam_probability("s", "b") is None


async def test_api_error_returns_none_and_starts_cooldown(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.status = 500

    assert await jev_service.classify_lead_temperature("x") is None
    assert mock.calls == 1

    # Cooling down: the next call is skipped without touching the API.
    mock.status = 200
    mock.answers = {"lead_temperature": choice_answer("hot", 0.9)}
    assert await jev_service.classify_lead_temperature("x") is None
    assert mock.calls == 1

    jev_service.reset_cooldown()
    assert (await jev_service.classify_lead_temperature("x")).label == "hot"
    assert mock.calls == 2


async def test_transport_exception_never_raises(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.raise_exc = RuntimeError("connection reset")
    assert await jev_service.spam_probability("s", "b") is None


async def test_per_feature_switches(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    s = get_settings()
    monkeypatch.setattr(s, "jev_lead_scoring_enabled", False)
    monkeypatch.setattr(s, "jev_reply_classification_enabled", False)
    monkeypatch.setattr(s, "jev_spam_check_enabled", False)

    assert await jev_service.classify_lead_temperature("x") is None
    assert await jev_service.classify_reply_intent("s", "b") is None
    assert await jev_service.spam_probability("s", "b") is None
    assert mock.calls == 0


async def test_long_state_is_truncated(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"lead_temperature": choice_answer("cold", 0.7)}
    await jev_service.classify_lead_temperature("x" * 20000)
    assert len(mock.requests[0]["state"]) < 7000


def test_choice_as_dict_is_json_shaped():
    c = jev_service.JevChoice("warm", 0.612345, {"warm": 0.612345}, "jev-test")
    assert c.as_dict() == {
        "label": "warm",
        "confidence": 0.6123,
        "probabilities": {"warm": 0.6123},
        "model": "jev-test",
    }
