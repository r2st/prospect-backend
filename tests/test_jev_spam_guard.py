"""Jev as a second opinion on the pre-send spam gate.

Jev can only make a verdict stricter, can only block under the same switch that
governs heuristic blocking, and with Jev absent or failing the async gate is
indistinguishable from the sync one.
"""

import pytest

pytest.importorskip("typesafe_sdk")

from app.config import get_settings  # noqa: E402
from app.services import jev_service, spam_guard  # noqa: E402
from tests import jev_mock  # noqa: E402
from tests.jev_mock import noul_answer  # noqa: E402
from tests.test_spam_guard import (  # noqa: E402
    CLEAN_BODY,
    CLEAN_SUBJECT,
    SPAMMY_BODY,
    SPAMMY_SUBJECT,
)


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "spam_guard_enabled", True)
    monkeypatch.setattr(s, "spam_guard_flag_threshold", 25)
    monkeypatch.setattr(s, "spam_guard_block_threshold", 50)
    monkeypatch.setattr(s, "spam_guard_block_enabled", False)
    jev_service.reset_cooldown()
    yield
    jev_service.reset_cooldown()


async def test_unconfigured_matches_sync_scan():
    sync = spam_guard.scan(CLEAN_SUBJECT, CLEAN_BODY)
    allowed, verdict = await spam_guard.scan_for_send_async(CLEAN_SUBJECT, CLEAN_BODY)
    assert allowed is True
    assert verdict.verdict == sync.verdict == spam_guard.PASS
    assert verdict.jev_spam_probability is None
    assert "jev_spam_probability" not in verdict.as_metadata()


async def test_low_jev_probability_leaves_clean_copy_passing(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.08)}

    allowed, verdict = await spam_guard.scan_for_send_async(CLEAN_SUBJECT, CLEAN_BODY)

    assert allowed is True
    assert verdict.verdict == spam_guard.PASS
    assert verdict.jev_spam_probability == pytest.approx(0.08)


async def test_jev_flags_copy_the_heuristic_passed(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.82)}

    allowed, verdict = await spam_guard.scan_for_send_async(CLEAN_SUBJECT, CLEAN_BODY)

    assert allowed is True  # flag, not block
    assert verdict.verdict == spam_guard.FLAG
    assert verdict.issues[0] == "Jev spam probability 0.82"
    assert verdict.as_metadata()["jev_spam_probability"] == pytest.approx(0.82)


async def test_jev_sees_visible_text_not_html(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.1)}

    await spam_guard.scan_async(CLEAN_SUBJECT, CLEAN_BODY)

    state = mock.requests[0]["state"]
    assert CLEAN_SUBJECT in state
    assert "stamping presses" in state
    assert "<p>" not in state


async def test_jev_cannot_block_while_blocking_is_off(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.99)}

    allowed, verdict = await spam_guard.scan_for_send_async(CLEAN_SUBJECT, CLEAN_BODY)

    assert allowed is True
    assert verdict.verdict == spam_guard.FLAG


async def test_jev_blocks_when_blocking_is_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "spam_guard_block_enabled", True)
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.95)}

    allowed, verdict = await spam_guard.scan_for_send_async(CLEAN_SUBJECT, CLEAN_BODY)

    assert allowed is False
    assert verdict.verdict == spam_guard.BLOCK
    assert "Jev spam probability 0.95" in verdict.reason()


async def test_jev_never_loosens_a_heuristic_flag(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.01)}

    _, verdict = await spam_guard.scan_for_send_async(SPAMMY_SUBJECT, SPAMMY_BODY)

    assert verdict.verdict == spam_guard.FLAG
    assert verdict.jev_spam_probability == pytest.approx(0.01)


async def test_heuristic_block_skips_jev(monkeypatch):
    monkeypatch.setattr(get_settings(), "spam_guard_block_enabled", True)
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.01)}

    allowed, verdict = await spam_guard.scan_for_send_async(SPAMMY_SUBJECT, SPAMMY_BODY)

    assert allowed is False
    assert verdict.verdict == spam_guard.BLOCK
    assert mock.calls == 0


async def test_disabled_gate_skips_jev(monkeypatch):
    monkeypatch.setattr(get_settings(), "spam_guard_enabled", False)
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.99)}

    allowed, verdict = await spam_guard.scan_for_send_async(SPAMMY_SUBJECT, SPAMMY_BODY)

    assert allowed is True
    assert mock.calls == 0


async def test_jev_outage_keeps_heuristic_verdict(monkeypatch):
    mock = jev_mock.install(monkeypatch)
    mock.status = 503

    allowed, verdict = await spam_guard.scan_for_send_async(CLEAN_SUBJECT, CLEAN_BODY)

    assert allowed is True
    assert verdict.verdict == spam_guard.PASS
    assert verdict.jev_spam_probability is None


async def test_dispatcher_uses_jev(monkeypatch):
    """``send_outreach_email`` refuses copy Jev blocks, before any IO."""
    from app.services import email_dispatch

    monkeypatch.setattr(get_settings(), "spam_guard_block_enabled", True)
    mock = jev_mock.install(monkeypatch)
    mock.answers = {"is_spam": noul_answer(0.97)}

    async def _enabled(db):
        return True

    async def _not_suppressed(db, email):
        return None

    monkeypatch.setattr("app.services.system_settings.is_sending_enabled", _enabled)
    monkeypatch.setattr("app.services.suppression.address_blocking_reason", _not_suppressed)

    result = await email_dispatch.send_outreach_email(
        db=None, to_email="dana@example.com",
        subject=CLEAN_SUBJECT, body_html=CLEAN_BODY,
    )

    assert result["status"] == "skipped"
    assert result["spam_blocked"] is True
    assert result["spam_scan"]["jev_spam_probability"] == pytest.approx(0.97)
