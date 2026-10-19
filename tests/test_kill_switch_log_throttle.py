"""The halted-queue warning is logged once an hour per org, not every sweep."""

import uuid

from app.tasks import outreach


def test_kill_switch_warning_is_throttled_per_org(monkeypatch):
    org_a, org_b = uuid.uuid4(), uuid.uuid4()
    current = {"org": org_a}
    monkeypatch.setattr("app.org_context.get_current_org_id", lambda: current["org"])
    monkeypatch.setattr(outreach, "_kill_switch_last_logged", {})

    assert outreach._log_kill_switch_halt(now=0) is True
    assert outreach._log_kill_switch_halt(now=300) is False      # next sweep: quiet
    assert outreach._log_kill_switch_halt(now=3599) is False
    current["org"] = org_b
    assert outreach._log_kill_switch_halt(now=3599) is True      # other org logs its own
    current["org"] = org_a
    assert outreach._log_kill_switch_halt(now=3600) is True      # an hour on: once more
