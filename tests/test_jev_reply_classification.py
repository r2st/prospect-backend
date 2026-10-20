"""Jev on the reply-classification path.

Order of precedence in ``AIService.classify_reply``: keyword pre-filter (never
displaced — opt-outs must survive any model) → a confident Jev label → the LLM,
with Jev's label kept as the answer if the LLM is unavailable.
"""

import uuid as _uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

pytest.importorskip("typesafe_sdk")

from app.models.email_reply import EmailReply  # noqa: E402
from app.services import jev_service  # noqa: E402
from app.services.ai_service import AIService  # noqa: E402
from app.tasks.replies import classify_reply_record  # noqa: E402
from tests import jev_mock  # noqa: E402
from tests.jev_mock import choice_answer  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    jev_service.reset_cooldown()
    yield
    jev_service.reset_cooldown()


class _LLM:
    """Stands in for ``complete_json`` and counts calls."""

    def __init__(self, result: dict):
        self.result = result
        self.calls = 0

    async def __call__(self, prompt, max_tokens=400):
        self.calls += 1
        return dict(self.result)


def _svc(monkeypatch, llm_result: dict) -> tuple[AIService, _LLM]:
    svc = AIService.__new__(AIService)  # skip __init__ (no API keys needed)
    svc.model = "test-model"
    llm = _LLM(llm_result)
    monkeypatch.setattr(svc, "complete_json", llm, raising=False)
    return svc, llm


def _jev(mock, label, confidence):
    mock.answers = {"reply_intent": choice_answer(label, confidence)}


LLM_INTERESTED = {"category": "interested", "confidence": 0.9, "reason": "llm"}
LLM_DOWN = {"error": "all providers down"}


async def test_keyword_unsubscribe_never_reaches_jev(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "interested", 0.99)
    svc, llm = _svc(monkeypatch, LLM_INTERESTED)

    out = await svc.classify_reply("Re: hi", "please unsubscribe me")

    assert out["category"] == "unsubscribe"
    assert out["source"] == "keyword"
    assert mock.calls == 0
    assert llm.calls == 0


async def test_confident_jev_label_skips_the_llm(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "not_interested", 0.91)
    svc, llm = _svc(monkeypatch, LLM_INTERESTED)

    out = await svc.classify_reply("Re: intro", "We're all set, thanks.", from_email="d@acme.com")

    assert out["category"] == "not_interested"
    assert out["source"] == "jev"
    assert out["confidence"] == pytest.approx(0.91)
    assert out["model"] == "jev-test"
    assert out["jev"]["label"] == "not_interested"
    assert llm.calls == 0
    assert mock.calls == 1


async def test_low_confidence_jev_defers_to_llm_and_is_recorded(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "not_interested", 0.4)
    svc, llm = _svc(monkeypatch, LLM_INTERESTED)

    out = await svc.classify_reply("Re: intro", "hmm, maybe later in the year?")

    assert out["category"] == "interested"
    assert out["source"] == "ai"
    assert out["jev"]["label"] == "not_interested"
    assert llm.calls == 1


async def test_wrong_person_goes_to_llm_for_the_referral(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "wrong_person", 0.97)
    svc, llm = _svc(monkeypatch, {
        "category": "wrong_person", "confidence": 0.9, "reason": "points elsewhere",
        "referral": {"name": "Pat", "email": "pat@acme.com"},
    })

    out = await svc.classify_reply("Re: intro", "Not me — try Pat, pat@acme.com")

    assert llm.calls == 1
    assert out["source"] == "ai"
    assert out["referral"]["email"] == "pat@acme.com"


async def test_jev_answer_stands_when_llm_is_down(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "wrong_person", 0.97)
    svc, llm = _svc(monkeypatch, LLM_DOWN)

    out = await svc.classify_reply("Re: intro", "I've left the company.")

    assert llm.calls == 1
    assert out["category"] == "wrong_person"
    assert out["source"] == "jev"


async def test_jev_out_of_office_parses_return_date(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "out_of_office", 0.95)
    svc, _ = _svc(monkeypatch, LLM_INTERESTED)

    # Phrased to dodge the keyword pre-filter, so the label is Jev's.
    out = await svc.classify_reply("Re: intro", "Travelling, back at my desk 2099-03-14.")

    assert out["source"] == "jev"
    assert out["category"] == "out_of_office"
    assert out["return_date"] == "2099-03-14"


async def test_jev_unsubscribe_marks_explicit(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "unsubscribe", 0.9)
    svc, _ = _svc(monkeypatch, LLM_INTERESTED)

    out = await svc.classify_reply("Re: intro", "Stop contacting me.")

    assert out["category"] == "unsubscribe"
    assert out["unsubscribe_explicit"] is True


async def test_unconfigured_jev_is_the_old_llm_path(monkeypatch):
    svc, llm = _svc(monkeypatch, LLM_INTERESTED)
    out = await svc.classify_reply("Re: hi", "yes let's talk")
    assert out["source"] == "ai"
    assert "jev" not in out
    assert llm.calls == 1


async def test_jev_error_falls_back_to_llm(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.status = 500
    svc, llm = _svc(monkeypatch, LLM_INTERESTED)
    out = await svc.classify_reply("Re: hi", "yes let's talk")
    assert out["source"] == "ai"
    assert llm.calls == 1


# ─── End to end through classify_reply_record ────────────────────────────────


class _Notify:
    def __init__(self):
        self.replies = []

    async def notify_reply(self, contact_name, company_name, contact_email):
        self.replies.append(contact_email)

    async def _send_notification(self, subject, body):
        pass


async def test_record_persists_jev_meta_and_routes(
    monkeypatch, db_session: AsyncSession, sample_company, sample_contact
):
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "interested", 0.88)
    svc, llm = _svc(monkeypatch, LLM_DOWN)

    reply = EmailReply(
        id=_uuid.uuid4(),
        contact_id=sample_contact.id,
        company_id=sample_company.id,
        from_email=sample_contact.email,
        subject="Re: intro",
        body_text="Sounds useful — what does onboarding look like?",
        snippet="Sounds useful",
        received_at=datetime.now(timezone.utc),
    )
    db_session.add(reply)
    await db_session.flush()
    notify = _Notify()

    summary = await classify_reply_record(
        db_session, reply, ai_service=svc, notification_service=notify
    )

    assert summary["category"] == "interested"
    assert summary["source"] == "jev"
    assert reply.priority_flag is True
    assert reply.classification_meta["source"] == "jev"
    assert reply.classification_meta["jev"]["probabilities"]["interested"] == pytest.approx(0.88)
    assert notify.replies == [sample_contact.email]
    assert llm.calls == 0
    assert sample_company.name in mock.requests[0]["state"]


async def test_record_gate_still_applies_to_jev(
    monkeypatch, db_session: AsyncSession, sample_company, sample_contact
):
    """Jev below the auto-action threshold → unknown, no side effects.

    The LLM is down, so Jev's low-confidence label is what comes back — and the
    same confidence gate that holds back a weak LLM label holds it back too.
    """
    mock = jev_mock.install(monkeypatch)
    _jev(mock, "not_interested", 0.5)
    svc, _ = _svc(monkeypatch, LLM_DOWN)

    reply = EmailReply(
        id=_uuid.uuid4(),
        contact_id=sample_contact.id,
        company_id=sample_company.id,
        from_email=sample_contact.email,
        subject="Re: intro",
        body_text="not sure",
        snippet="not sure",
        received_at=datetime.now(timezone.utc),
    )
    db_session.add(reply)
    await db_session.flush()

    summary = await classify_reply_record(
        db_session, reply, ai_service=svc, notification_service=_Notify()
    )

    assert summary["raw_category"] == "not_interested"
    assert summary["category"] == "unknown"
    assert summary["handled_action"] == "none"
