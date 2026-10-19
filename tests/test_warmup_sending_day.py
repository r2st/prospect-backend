"""The mailbox warm-up ramp steps on the sending day, like the counter it caps.

``warmup.used_today`` counts a mailbox's sends on ``send_day``'s local day; the
ramp's day number used the UTC date, so a Toronto mailbox's limit moved up at
20:00 local while that day's sends were still being counted against it.
"""

from datetime import date, datetime, timezone

from app.services import send_day, warmup


def test_local_date_is_the_sending_days_date(monkeypatch):
    monkeypatch.setattr(send_day, "send_day_timezone", lambda: __import__("zoneinfo").ZoneInfo("America/Toronto"))
    # 02:00 UTC on the 19th is still 22:00 on the 18th in Toronto.
    assert send_day.local_date(datetime(2026, 10, 19, 2, 0, tzinfo=timezone.utc)) == date(2026, 10, 18)
    assert send_day.local_date(datetime(2026, 10, 19, 5, 0, tzinfo=timezone.utc)) == date(2026, 10, 19)


def test_ramp_day_defaults_to_the_sending_day(monkeypatch):
    monkeypatch.setattr(send_day, "local_date", lambda now=None: date(2026, 10, 18))
    # Started on the 17th: day 2 on the sender's 18th, whatever UTC says.
    assert warmup.day_for(date(2026, 10, 17)) == 2


def test_recompute_does_not_step_before_local_midnight(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(send_day, "local_date", lambda now=None: date(2026, 10, 18))
    plan = SimpleNamespace(status="warming", start_date=date(2026, 10, 17), current_day=2,
                           daily_limit=warmup.limit_for_day(2)[0])
    assert warmup.recompute(plan) is False
    assert plan.current_day == 2
