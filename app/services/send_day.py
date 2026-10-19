"""Which day a send counts against — one boundary, and not the UTC one.

The daily cap, the per-mailbox warm-up counter and the row-count seed behind
both all asked "is this send from today?" and all three answered it in UTC. That
is only the same question as "is this send from this business day?" when the
sender's business day happens to sit inside a UTC day, and this system
deliberately does not send inside one:
``SendLimiter.within_send_window`` evaluates the window in **the contact's own
timezone** (``tasks/outreach.py`` passes ``contact.timezone`` at both dispatch
sites), so a cadence sends into US Pacific business hours and Toronto business
hours on the same afternoon.

Take the shipped defaults — window 08:00–18:00, ``America/Toronto`` — and a
contact in ``America/Los_Angeles``. Their window is 15:00 UTC through 01:00 UTC
*the next day*. So 00:00 UTC lands at 17:00 for them, an hour before their
window closes, and every counter keyed on the UTC date resets **while that
mailbox is still sending**. The last hour of a West-Coast business day gets a
fresh full daily allowance, and the mailbox can put up to twice its limit on the
wire inside one rolling day.

The warm-up ramp is the half that matters. A day-1 mailbox on a 5/day plan sends
five, resets at 17:00 local, and sends five more — and every warm-up check still
reads ``5/5`` afterwards, because the counter it reads is the one that rolled
over. Same failure the ramp exists to prevent, arriving through the clock rather
than through an ungated send path: nothing errors, the graph looks textbook, and
what notices is the receiving side, weeks later.

**The boundary is anchored to ``send_window_timezone``**, and that is the only
setting that can carry it. The window is ``start_hour`` to ``end_hour`` on a 24h
clock in that zone, so it cannot span local midnight — which makes local
midnight the one instant guaranteed to fall *outside* the sender's own sending
hours. ``daily_summary_timezone`` is the operator's reporting zone (shipped as
``Asia/Kolkata``) and would put the rollover at 18:30 UTC, squarely inside the
default Toronto window: worse than the UTC boundary it replaced.

Contacts far enough east still cross it — a Tokyo business day spans local
midnight in Toronto no matter what — but that residue is bounded by the
sender's own ten-hour window, where the UTC anchor had no such bound and broke
on the ordinary case of a North American sender mailing the West Coast.

Everything here returns **aware UTC**, because every column it is compared
against is stored that way.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


def send_day_timezone():
    """The tzinfo the sending day is anchored to, falling back to UTC.

    Fails soft: an unknown or unavailable zone name degrades to UTC — the
    previous behaviour — rather than taking down every send path that needs to
    know what day it is.
    """
    tz_name = get_settings().send_window_timezone
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(tz_name)
    except Exception:  # pragma: no cover - bad/unavailable tz name
        logger.info("unknown send-window timezone %r; day boundary falls back to UTC", tz_name)
        return timezone.utc


def day_start(now: Optional[datetime] = None) -> datetime:
    """Aware-UTC instant at which the current sending day began.

    The lower bound of every "sent today" count. Note it is *not* midnight UTC
    and generally not midnight anything in UTC terms — it is the sender's local
    midnight expressed as an instant.
    """
    now = _aware(now)
    tz = send_day_timezone()
    local_midnight = now.astimezone(tz).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return local_midnight.astimezone(timezone.utc)


def local_date(now: Optional[datetime] = None) -> date:
    """The calendar date of the current sending day.

    For day-granular state that has to agree with the counters above — the
    warm-up ramp's day number most of all. That ramp stepped on the UTC date
    while the per-mailbox count it is compared against rolled on this one, so a
    mailbox's limit changed partway through its own sending day.
    """
    return _aware(now).astimezone(send_day_timezone()).date()


def day_key(now: Optional[datetime] = None) -> str:
    """``YYYY-MM-DD`` of the current sending day, for keying a counter.

    Derived from the same local calendar :func:`day_start` uses, so a key
    rollover and a count rollover cannot disagree — the Redis counter is seeded
    from a row count taken with the other function.
    """
    return _aware(now).astimezone(send_day_timezone()).strftime("%Y-%m-%d")


def seconds_until_next_day(now: Optional[datetime] = None) -> int:
    """Seconds from *now* until the next sending-day boundary, floored at 60.

    The TTL for a per-day counter key: it should outlive the day it counts and
    not much longer. Computed by adding a day to the *local* date and
    re-resolving, so a DST transition inside the window shortens or lengthens it
    correctly instead of assuming 86400.
    """
    now = _aware(now)
    tz = send_day_timezone()
    local = now.astimezone(tz)
    next_local_midnight = (local + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    delta = next_local_midnight.astimezone(timezone.utc) - now
    # A grace margin on top, so a counter cannot expire a moment before the
    # sends it is counting stop being today's.
    return max(60, int(delta.total_seconds()) + 300)


def _aware(now: Optional[datetime]) -> datetime:
    """Normalise a caller's ``now`` to aware UTC.

    Callers pass ``now`` for testability and a few of them hand over a naive
    datetime read back off a column. Treating that as UTC is what every other
    timestamp helper in this codebase does.
    """
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)
