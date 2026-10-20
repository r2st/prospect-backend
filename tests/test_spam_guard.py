"""The pre-send spam gate.

The content scanner existed for a year as an endpoint a human could click. These
tests pin it into the send path: clean copy passes untouched, risky copy is
flagged and recorded, and blocking — which is off by default on purpose — soft-
skips rather than failing the enrollment when it is turned on.
"""

import pytest

from app.config import get_settings
from app.services import spam_guard

CLEAN_SUBJECT = "quick question about your press line"
CLEAN_BODY = (
    "<p>Hi Dana — noticed you run three stamping presses. Most shops that size "
    "lose an hour a shift to unlogged downtime. Worth a short call?</p>"
    "<p>Suman</p>"
)

SPAMMY_SUBJECT = "ACT NOW — LIMITED TIME OFFER!!!"
SPAMMY_BODY = (
    "<p>CLICK HERE to BUY NOW and MAKE MONEY RISK FREE!!! 100% FREE cash bonus, "
    "no obligation, order now!!! This is not spam.</p>"
    "<p>http://a.example http://b.example http://c.example http://d.example "
    "http://e.example</p>"
)


@pytest.fixture(autouse=True)
def _restore_settings():
    s = get_settings()
    before = (
        s.spam_guard_enabled,
        s.spam_guard_flag_threshold,
        s.spam_guard_block_threshold,
        s.spam_guard_block_enabled,
    )
    yield
    (
        s.spam_guard_enabled,
        s.spam_guard_flag_threshold,
        s.spam_guard_block_threshold,
        s.spam_guard_block_enabled,
    ) = before


def test_clean_copy_passes_and_records_nothing():
    verdict = spam_guard.scan(CLEAN_SUBJECT, CLEAN_BODY)
    assert verdict.verdict == spam_guard.PASS
    assert verdict.allowed is True
    assert verdict.flagged is False


def test_spammy_copy_is_flagged_with_the_reasons_named():
    verdict = spam_guard.scan(SPAMMY_SUBJECT, SPAMMY_BODY)
    assert verdict.flagged is True
    assert verdict.score >= 25
    # The operator has to be able to act on this, so the triggers are itemised.
    joined = " ".join(verdict.issues).lower()
    assert "spam-trigger" in joined
    assert any("link" in i for i in verdict.issues)
    assert verdict.suggestions


def test_blocking_is_off_by_default():
    """Even flagrantly spammy copy sends unless an operator opted into blocking.

    A content heuristic that stops outreach on its own judgement is a worse
    failure mode than the spam it prevents — the default must be observe-only.
    """
    assert get_settings().spam_guard_block_enabled is False
    allowed, verdict = spam_guard.scan_for_send(SPAMMY_SUBJECT, SPAMMY_BODY)
    assert allowed is True
    assert verdict.verdict == spam_guard.FLAG


def test_blocking_when_enabled_refuses_the_send():
    s = get_settings()
    s.spam_guard_block_enabled = True

    allowed, verdict = spam_guard.scan_for_send(SPAMMY_SUBJECT, SPAMMY_BODY)
    assert allowed is False
    assert verdict.verdict == spam_guard.BLOCK
    assert "spam risk" in verdict.reason()


def test_clean_copy_is_never_blocked_even_with_blocking_on():
    s = get_settings()
    s.spam_guard_block_enabled = True

    allowed, verdict = spam_guard.scan_for_send(CLEAN_SUBJECT, CLEAN_BODY)
    assert allowed is True
    assert verdict.verdict == spam_guard.PASS


def test_missing_unsubscribe_raises_the_score():
    with_unsub = spam_guard.scan(CLEAN_SUBJECT, CLEAN_BODY, has_unsubscribe=True)
    without = spam_guard.scan(CLEAN_SUBJECT, CLEAN_BODY, has_unsubscribe=False)
    assert without.score > with_unsub.score


def test_thresholds_are_honoured():
    s = get_settings()
    s.spam_guard_flag_threshold = 1
    s.spam_guard_block_threshold = 2
    s.spam_guard_block_enabled = True

    # Even mildly imperfect copy now trips the (absurdly low) block threshold.
    allowed, verdict = spam_guard.scan_for_send("HELLO THERE", "hi!!! click here")
    assert allowed is False
    assert verdict.score >= 2


def test_disabled_guard_is_a_no_op():
    s = get_settings()
    s.spam_guard_enabled = False
    s.spam_guard_block_enabled = True

    verdict = spam_guard.scan(SPAMMY_SUBJECT, SPAMMY_BODY)
    assert verdict.verdict == spam_guard.PASS
    assert verdict.score == 0


def test_scanner_failure_allows_the_send(monkeypatch):
    """A broken scanner must degrade to "send it", never to a mail stoppage."""
    s = get_settings()
    s.spam_guard_block_enabled = True

    import app.services.deliverability_optimizer as opt

    def _boom():
        raise RuntimeError("scanner exploded")

    monkeypatch.setattr(opt, "get_deliverability_optimizer", _boom)

    allowed, verdict = spam_guard.scan_for_send(SPAMMY_SUBJECT, SPAMMY_BODY)
    assert allowed is True
    assert verdict.error == "scanner exploded"


def test_metadata_is_compact_enough_to_store_on_every_flagged_send():
    verdict = spam_guard.scan(SPAMMY_SUBJECT, SPAMMY_BODY)
    meta = verdict.as_metadata()
    assert set(meta) == {"verdict", "score", "risk", "issues"}
    assert len(meta["issues"]) <= 6


# --- Send-path integration ----------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_blocks_spammy_copy(monkeypatch):
    """The dispatcher is the backstop for every path that doesn't scan first."""
    from app.services import email_dispatch

    s = get_settings()
    s.spam_guard_block_enabled = True

    async def _enabled(db):
        return True

    async def _not_suppressed(db, email):
        # The whole compliance gate, not just its list clause: it also reads the
        # per-contact flags now, and this test runs with ``db=None``.
        return None

    monkeypatch.setattr(
        "app.services.system_settings.is_sending_enabled", _enabled
    )
    monkeypatch.setattr(
        "app.services.suppression.address_blocking_reason", _not_suppressed
    )

    result = await email_dispatch.send_outreach_email(
        db=None, to_email="dana@example.com",
        subject=SPAMMY_SUBJECT, body_html=SPAMMY_BODY,
    )
    assert result["status"] == "skipped"
    assert result["spam_blocked"] is True
    assert result["spam_scan"]["verdict"] == "block"


@pytest.mark.asyncio
async def test_dispatcher_scan_can_be_skipped_by_a_caller_that_already_scanned(monkeypatch):
    """The sequence path scans the pre-tracking body; re-scoring wrapped links
    would flag its own instrumentation as "too many links"."""
    from app.services import email_dispatch

    s = get_settings()
    s.spam_guard_block_enabled = True

    async def _enabled(db):
        return True

    async def _not_suppressed(db, email):
        # The whole compliance gate, not just its list clause: it also reads the
        # per-contact flags now, and this test runs with ``db=None``.
        return None

    calls: list = []

    async def _record(*args, **kwargs):
        calls.append(args)
        return True, spam_guard.SpamVerdict()

    monkeypatch.setattr("app.services.system_settings.is_sending_enabled", _enabled)
    monkeypatch.setattr(
        "app.services.suppression.address_blocking_reason", _not_suppressed
    )
    # The dispatcher calls the async (Jev-aware) variant; patching the sync one
    # would make ``calls == []`` true whether or not the scan was skipped.
    monkeypatch.setattr(spam_guard, "scan_for_send_async", _record)

    with pytest.raises(AttributeError):
        # db=None means the dispatcher dies further downstream — which is the
        # point: it got past the gate without the scanner ever being consulted.
        await email_dispatch.send_outreach_email(
            db=None, to_email="dana@example.com",
            subject=SPAMMY_SUBJECT, body_html=SPAMMY_BODY,
            skip_spam_scan=True,
        )
    assert calls == []
