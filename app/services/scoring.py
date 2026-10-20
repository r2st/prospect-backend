"""Lead scoring service — rule-based scoring and AI scoring via Claude API.

Enhanced with sales-book recommendations:
- **Signal Recency Decay** (ai-cold-calling-guidelines.md): Recent engagement
  signals are weighted higher — a click 30 minutes ago is worth 20× more than
  one from two days ago.
- **Engagement Velocity Multiplier** (ai-cold-calling-guidelines.md): Clustered
  signals (2+ events in 24 hours) indicate active buying behaviour, so velocity
  *multiplies* the live-engagement signal rather than adding a flat bonus — a
  lead going 0→3 opens in a day jumps sharply.
- **Account Tier Classification** (digital-sales-strategy.md ABM model):
  Tier 1 (Strategic) / Tier 2 (Target) / Tier 3 (Prospect) classification for
  prioritised resource allocation.
- **Score change history**: every score records its delta versus the previous
  score, and :func:`score_history` / :func:`score_change_since` expose a lead's
  score movement over time ("jumped +40 points this week").
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.company import Company
from app.models.contact import Contact
from app.models.lead_score import LeadScore
from app.services.industry_verticals import matches_manufacturing, resolve_vertical
from app.services.jev_scoring import jev_lead_temperature

logger = logging.getLogger(__name__)

# Letter-grade bands for a 0-100 fit score. A/B are the leads worth working;
# C is nurture; D/F are poor fits / disqualify candidates.
_GRADE_BANDS: list[tuple[int, str]] = [(80, "A"), (65, "B"), (50, "C"), (35, "D")]


def grade_for_fit(fit_score: Optional[int]) -> str:
    """Map a 0-100 fit score to an A/B/C/D/F letter grade.

    A ≥ 80, B ≥ 65, C ≥ 50, D ≥ 35, else F. ``None`` grades as ``F`` (unscored
    leads are treated as the lowest tier until scored).
    """
    if fit_score is None:
        return "F"
    for threshold, letter in _GRADE_BANDS:
        if fit_score >= threshold:
            return letter
    return "F"


# Engagement bonus weights (points added on top of data-completeness). These
# reward real buying signals — a reply is worth far more than an open.
_ENGAGEMENT_WEIGHTS = {"replied": 12, "clicked": 7, "opened": 3}
_NEWS_ACTIONABLE_BONUS = 5
_MAX_ENGAGEMENT_BONUS = 25

# Buyer-intent signals (job postings, funding, expansion, etc.) are strong
# timing signals — a relevant + urgent one can add up to this many fit points.
_MAX_INTENT_BONUS = 15


# ---------------------------------------------------------------------------
# Signal Recency Decay (from ai-cold-calling-guidelines.md)
# ---------------------------------------------------------------------------
_RECENCY_BRACKETS: list[tuple[float, float]] = [
    (0.5, 1.0),    # within 30 min
    (1.0, 0.85),   # within 1 hour
    (2.0, 0.70),   # within 2 hours
    (4.0, 0.50),   # within 4 hours
    (8.0, 0.30),   # within 8 hours
    (24.0, 0.15),  # within 24 hours
    (48.0, 0.05),  # within 48 hours
]
_RECENCY_FLOOR = 0.02


def signal_recency_weight(
    last_event_at: Optional[datetime], now: Optional[datetime] = None
) -> float:
    """Return a 0.0-1.0 weight reflecting how recently the lead engaged.

    Uses the decay curve from *ai-cold-calling-guidelines.md*: a signal within
    30 minutes scores 1.0 and decays through seven brackets down to 0.02 for
    anything older than 48 hours.
    """
    if last_event_at is None:
        return 0.0
    if now is None:
        now = datetime.now(timezone.utc)
    if last_event_at.tzinfo is None:
        last_event_at = last_event_at.replace(tzinfo=timezone.utc)
    hours_ago = max(0, (now - last_event_at).total_seconds() / 3600.0)
    for threshold_hours, weight in _RECENCY_BRACKETS:
        if hours_ago <= threshold_hours:
            return weight
    return _RECENCY_FLOOR


# ---------------------------------------------------------------------------
# Engagement Velocity Multiplier (from ai-cold-calling-guidelines.md)
# ---------------------------------------------------------------------------
# Velocity is a *multiplier* on the live-engagement signal, not a flat bonus.
# A lead that goes from cold to a burst of activity in 24 hours is behaving very
# differently from one with a single stale open, so its engagement score should
# jump — a cluster of 3 events in a day nearly doubles the signal weight, 5+
# more than doubles it. (Firmographic fit is never multiplied; only the
# engagement/recency component is, so a hot burst can't inflate a poor-fit lead
# past a good-fit one on firmographics alone.)
_VELOCITY_MULTIPLIERS: list[tuple[int, float]] = [
    (6, 2.5),   # 6+ distinct events in 24h — very active buying behaviour
    (5, 2.2),   # 5 events
    (4, 1.9),   # 4 events
    (3, 1.6),   # 3 events — a clear cluster
    (2, 1.25),  # 2 events
]
_VELOCITY_NEUTRAL = 1.0


def velocity_multiplier(signal_count: int) -> float:
    """Map a 24-hour engagement signal count to an engagement multiplier.

    Returns ``1.0`` (neutral) for 0-1 signals and rises to ``2.5`` for a burst
    of 6+ events, so clustered activity multiplies the engagement weight rather
    than adding a small fixed bonus.
    """
    for threshold, mult in _VELOCITY_MULTIPLIERS:
        if signal_count >= threshold:
            return mult
    return _VELOCITY_NEUTRAL


async def _engagement_velocity(
    db: AsyncSession, company: Company, now: Optional[datetime] = None
) -> tuple[int, Optional[datetime]]:
    """Count distinct engagement events in the last 24 hours and return
    ``(signal_count, most_recent_event)``.

    Events counted: email opens, clicks, and replies from any contact at
    this company, **across both send tables** — ``email_sends`` (manual,
    template, inbox replies) and ``outreach_events`` (the sequence engine, which
    writes no ``EmailSend`` row). See CLAUDE.md, "Sent mail lives in two tables".

    Reading only ``EmailSend`` here meant velocity was computed over the manual
    send path alone. Nearly all volume — every Dialtone sequence send and every
    open, click and reply it earns — goes through ``outreach_events``, so a
    company that opened three cold emails this morning scored a velocity of zero
    and got the neutral 1.0 multiplier. The burst-detection this function exists
    for could not fire on the only channel that produces bursts.
    """
    from app.models.email_template import EmailSend
    from app.models.outreach import OutreachEnrollment, OutreachEvent

    if now is None:
        now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=24)

    signal_count = 0
    most_recent: Optional[datetime] = None

    def _consider(ts: Optional[datetime]) -> None:
        nonlocal signal_count, most_recent
        if ts is None:
            return
        # SQLite (and any tz-naive source) hands back naive datetimes;
        # normalise to UTC so the window comparison never raises.
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts >= window_start:
            signal_count += 1
            if most_recent is None or ts > most_recent:
                most_recent = ts

    rows = await db.execute(
        select(
            EmailSend.opened_at,
            EmailSend.clicked_at,
            EmailSend.replied_at,
        )
        .join(Contact, Contact.id == EmailSend.contact_id)
        .where(
            Contact.company_id == company.id,
            or_(
                EmailSend.opened_at >= window_start,
                EmailSend.clicked_at >= window_start,
                EmailSend.replied_at >= window_start,
            ),
        )
    )
    for opened, clicked, replied in rows.all():
        for ts in (opened, clicked, replied):
            _consider(ts)

    # Sequence engagement. One row per event here (rather than three nullable
    # columns per send), so each qualifying row is exactly one signal.
    seq_rows = await db.execute(
        select(OutreachEvent.occurred_at)
        .join(OutreachEnrollment, OutreachEvent.enrollment_id == OutreachEnrollment.id)
        .join(Contact, OutreachEnrollment.contact_id == Contact.id)
        .where(
            Contact.company_id == company.id,
            OutreachEvent.event_type.in_(("opened", "clicked", "replied")),
            OutreachEvent.occurred_at >= window_start,
        )
    )
    for (occurred_at,) in seq_rows.all():
        _consider(occurred_at)

    return signal_count, most_recent


# ---------------------------------------------------------------------------
# Account Tier Classification (from digital-sales-strategy.md ABM model)
# ---------------------------------------------------------------------------

def classify_account_tier(
    industry_pts: int, size_pts: int, location_pts: int, engagement_pts: int
) -> tuple[str, float]:
    """Classify an account into ABM tiers based on its sub-scores.

    Returns ``(tier_name, tier_value)`` where tier_value is the account value
    multiplier from *ai-cold-calling-guidelines.md*.
    """
    total = industry_pts + size_pts + location_pts + engagement_pts
    if industry_pts >= 35 and size_pts >= 20 and location_pts >= 15 and total >= 70:
        return "tier_1_strategic", 1.0
    if industry_pts >= 20 and total >= 45:
        return "tier_2_target", 0.75
    if total >= 25:
        return "tier_3_prospect", 0.50
    return "non_target", 0.25


def _industry_score(
    naics_code: Optional[str],
    industry_segment: Optional[str] = None,
    company_name: Optional[str] = None,
) -> tuple[int, str]:
    """Score based on NAICS code (manufacturing = 31-33), falling back to the
    industry-vertical taxonomy when NAICS is absent.

    Places-sourced companies have no NAICS code — Places speaks its own
    taxonomy — but they do carry ``industry_segment`` (the ICP query that found
    them, e.g. "CNC Machining"; see ingestion + 183d5a9). Scoring only NAICS
    gave exactly those companies industry=0 and structurally disqualified the
    leads prospecting was built to find, so a missing/unrelated NAICS now defers
    to the segment before concluding "not manufacturing".
    """
    if naics_code:
        prefix = naics_code[:2]
        if prefix in ("31", "32", "33"):
            return 40, f"Manufacturing NAICS {naics_code} — perfect fit"

    vertical = resolve_vertical(industry_segment or "", company_name or "")
    if vertical is not None:
        return 40, f"Industry segment matches vertical '{vertical.name}' — perfect fit"
    if matches_manufacturing(industry_segment or "", company_name or ""):
        return 40, f"Industry segment '{industry_segment}' — manufacturing"

    if naics_code:
        # Related industries
        related = {
            "42": "Wholesale trade — often serves manufacturing",
            "48": "Transportation — logistics for manufacturing",
            "49": "Warehousing — supports manufacturing supply chain",
            "54": "Professional/technical services — may serve manufacturing",
            "56": "Admin/support services — may serve manufacturing",
        }
        if naics_code[:2] in related:
            return 20, related[naics_code[:2]]
        return 0, f"NAICS {naics_code} not related to manufacturing"
    if industry_segment:
        return 0, f"Segment '{industry_segment}' not related to manufacturing"
    return 0, "No NAICS code available"


def _size_score(employee_count: Optional[int]) -> tuple[int, str]:
    """Score based on company size. Sweet spot is 50-500."""
    if employee_count is None:
        return 5, "Employee count unknown"
    if 50 <= employee_count <= 500:
        return 25, f"{employee_count} employees — ideal mid-market manufacturer"
    if 500 < employee_count <= 2000:
        return 20, f"{employee_count} employees — large manufacturer"
    if 10 <= employee_count < 50:
        return 15, f"{employee_count} employees — small manufacturer"
    if employee_count > 2000:
        return 10, f"{employee_count} employees — enterprise (longer sales cycle)"
    return 5, f"{employee_count} employees — very small, may not need software"


def _location_score(country: str) -> tuple[int, str]:
    """Score based on country. Max 20.

    US and CA keep their historical weights as the primary/secondary markets.
    Every *other supported* market scores 12 rather than 0: scoring a country we
    deliberately prospect as "outside target markets" would structurally
    disqualify every lead the expansion was meant to find — the scorer would
    zero exactly the companies it is supposed to rank.

    A country outside ``settings.supported_countries`` still scores 0; that one
    really is off-market.
    """
    code = (country or "").upper()
    if code == "US":
        return 20, "United States — primary market"
    if code == "CA":
        return 15, "Canada — secondary market (CASL applies)"
    supported = [c.upper() for c in get_settings().supported_countries]
    if code in supported:
        return 12, f"{code} — supported market"
    return 0, f"Country {code} outside target markets"


def _completeness_score(
    company: Company,
    contacts: list[Contact],
) -> tuple[int, dict[str, int]]:
    """Score based on data completeness. Max 15 points."""
    score = 0
    breakdown: dict[str, int] = {}

    # Check contacts for email
    has_email = any(c.email for c in contacts)
    if has_email:
        score += 5
        breakdown["email"] = 5

    # Phone
    if company.phone:
        score += 3
        breakdown["phone"] = 3

    # Website
    if company.website or company.domain:
        score += 3
        breakdown["website"] = 3

    # Contact name
    has_named_contact = any(c.first_name or c.last_name for c in contacts)
    if has_named_contact:
        score += 4
        breakdown["contact_name"] = 4

    return score, breakdown


async def _engagement_signal(
    db: AsyncSession, company: Company
) -> tuple[int, dict]:
    """Score real engagement signals for a company: email opens/clicks/replies
    and recent actionable news. Returns (points, breakdown). Capped so a hot
    lead can't run away with the score.

    Counts engagement from **both send tables** — ``email_sends`` and
    ``outreach_events`` (see CLAUDE.md, "Sent mail lives in two tables"). The
    ``EmailSend``-only version of this query scored the manual send path and was
    blind to every Dialtone sequence send, which is where essentially all cold
    volume and all of its engagement lives. A company that opened four sequence
    emails and clicked twice contributed exactly zero engagement points to its
    own fit score, so the "engagement" component of lead scoring measured a
    channel the lead-gen engine doesn't use.

    Imported lazily to avoid import cycles and to keep the firmographic-only
    unit tests (which never create sends/news) returning zero here.
    """
    from app.models.email_template import EmailSend
    from app.models.company_news import CompanyNews
    from app.models.outreach import OutreachEnrollment, OutreachEvent

    breakdown: dict = {}
    points = 0

    # Email engagement across all of this company's contacts.
    eng = await db.execute(
        select(
            func.count(EmailSend.opened_at),
            func.count(EmailSend.clicked_at),
            func.count(EmailSend.replied_at),
        )
        .join(Contact, Contact.id == EmailSend.contact_id)
        .where(Contact.company_id == company.id)
    )
    opened, clicked, replied = eng.one()

    # Sequence engagement, added to the same counters — the weights below are
    # per *kind* of signal (a reply is worth more than an open), not per table,
    # so which path carried the email is irrelevant to what it says about intent.
    seq = await db.execute(
        select(OutreachEvent.event_type, func.count(OutreachEvent.id))
        .join(OutreachEnrollment, OutreachEvent.enrollment_id == OutreachEnrollment.id)
        .join(Contact, OutreachEnrollment.contact_id == Contact.id)
        .where(
            Contact.company_id == company.id,
            OutreachEvent.event_type.in_(("opened", "clicked", "replied")),
        )
        .group_by(OutreachEvent.event_type)
    )
    seq_counts = {event_type: count for event_type, count in seq.all()}
    opened = (opened or 0) + seq_counts.get("opened", 0)
    clicked = (clicked or 0) + seq_counts.get("clicked", 0)
    replied = (replied or 0) + seq_counts.get("replied", 0)

    if replied:
        points += _ENGAGEMENT_WEIGHTS["replied"]
        breakdown["replied"] = int(replied)
    if clicked:
        points += _ENGAGEMENT_WEIGHTS["clicked"]
        breakdown["clicked"] = int(clicked)
    if opened:
        points += _ENGAGEMENT_WEIGHTS["opened"]
        breakdown["opened"] = int(opened)

    # A recent, actionable news event is a timely outreach trigger.
    news_count = await db.execute(
        select(func.count(CompanyNews.id)).where(
            CompanyNews.company_id == company.id,
            CompanyNews.is_actionable.is_(True),
        )
    )
    actionable_news = news_count.scalar() or 0
    if actionable_news:
        points += _NEWS_ACTIONABLE_BONUS
        breakdown["actionable_news"] = int(actionable_news)

    points = min(points, _MAX_ENGAGEMENT_BONUS)
    return points, breakdown


async def _intent_signal_bonus(
    db: AsyncSession, company: Company, now: Optional[datetime] = None
) -> tuple[int, dict]:
    """Score active buyer-intent signals for a company. Returns (points, breakdown).

    Only signals still inside their buying window (``expires_at`` in the future,
    or null) count. The strongest signal dominates via its relevance-weighted
    urgency, with a smaller bonus for additional corroborating signals — several
    independent intent signals at once is a very strong buy indicator. Capped at
    ``_MAX_INTENT_BONUS`` so intent can't swamp firmographic fit.

    Imported lazily so firmographic-only unit tests (which never create signals)
    return zero here.
    """
    from app.models.intent_signal import IntentSignal

    if now is None:
        now = datetime.now(timezone.utc)

    rows = await db.execute(
        select(IntentSignal)
        .where(IntentSignal.company_id == company.id)
        .where(or_(IntentSignal.expires_at.is_(None), IntentSignal.expires_at >= now))
    )
    signals = list(rows.scalars().all())
    if not signals:
        return 0, {}

    scores = sorted((s.combined_score for s in signals), reverse=True)
    # Top signal contributes most; each extra active signal adds a diminishing tail.
    top = scores[0]
    corroboration = sum(scores[1:4]) * 0.15
    raw = (top / 100.0) * _MAX_INTENT_BONUS + corroboration / 10.0
    points = int(min(raw, _MAX_INTENT_BONUS))

    breakdown = {
        "active_signals": len(signals),
        "top_combined_score": round(top, 1),
        "types": sorted({s.signal_type for s in signals}),
    }
    return points, breakdown


# ---------------------------------------------------------------------------
# Score change history
# ---------------------------------------------------------------------------


async def _latest_score(
    db: AsyncSession, company_id, before: Optional[datetime] = None
) -> Optional[LeadScore]:
    """Return the most recent LeadScore for a company (optionally before a time)."""
    query = select(LeadScore).where(LeadScore.company_id == company_id)
    if before is not None:
        query = query.where(LeadScore.scored_at < before)
    query = query.order_by(LeadScore.scored_at.desc()).limit(1)
    return (await db.execute(query)).scalar_one_or_none()


async def _score_change(
    db: AsyncSession, company_id, new_fit: int, now: Optional[datetime] = None
) -> dict:
    """Build the ``score_change`` factor comparing ``new_fit`` to the last score.

    Returns a small JSON-serialisable dict recording the previous score, the
    signed delta, its direction, and the timestamp of the previous score so the
    UI can render "score jumped +40 since <date>".
    """
    prev = await _latest_score(db, company_id, before=now)
    if prev is None:
        return {
            "previous_fit_score": None,
            "delta": 0,
            "direction": "initial",
            "previous_scored_at": None,
        }
    delta = new_fit - prev.fit_score
    direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
    return {
        "previous_fit_score": prev.fit_score,
        "delta": delta,
        "direction": direction,
        "previous_scored_at": prev.scored_at.isoformat() if prev.scored_at else None,
    }


async def score_history(
    db: AsyncSession, company_id, limit: int = 20
) -> list[dict]:
    """Return a company's score timeline, newest first.

    Each entry is ``{fit_score, score_type, scored_at, delta}`` where ``delta``
    is the change from the immediately older score (``None`` for the oldest).
    Powers the "score over time" view and answers "how has this lead moved".
    """
    rows = (
        await db.execute(
            select(LeadScore)
            .where(LeadScore.company_id == company_id)
            .order_by(LeadScore.scored_at.desc())
            .limit(limit)
        )
    ).scalars().all()

    out: list[dict] = []
    # rows are newest→oldest; delta is vs the next (older) row.
    for i, row in enumerate(rows):
        older = rows[i + 1] if i + 1 < len(rows) else None
        out.append(
            {
                "fit_score": row.fit_score,
                "score_type": row.score_type,
                "scored_at": row.scored_at.isoformat() if row.scored_at else None,
                "delta": (row.fit_score - older.fit_score) if older else None,
            }
        )
    return out


async def score_change_since(
    db: AsyncSession, company_id, since: datetime, now: Optional[datetime] = None
) -> dict:
    """How much a company's score moved over a window (e.g. the last 7 days).

    Compares the latest score to the most recent score at or before ``since``.
    Returns ``{current, baseline, delta, since}``; ``baseline`` is ``None`` when
    there is no score old enough, in which case ``delta`` is ``0``.
    """
    current = await _latest_score(db, company_id, before=now)
    baseline = await _latest_score(db, company_id, before=since)
    current_fit = current.fit_score if current else None
    baseline_fit = baseline.fit_score if baseline else None
    delta = 0
    if current_fit is not None and baseline_fit is not None:
        delta = current_fit - baseline_fit
    return {
        "current": current_fit,
        "baseline": baseline_fit,
        "delta": delta,
        "since": since.isoformat(),
    }


async def score_company_rule_based(
    db: AsyncSession,
    company: Company,
) -> LeadScore:
    """Apply rule-based scoring to a company. Returns the new LeadScore.

    Combines firmographic fit (industry, size, location), data completeness,
    real engagement signals, **signal recency decay**, **engagement velocity**,
    and **account tier classification** per sales-book recommendations.
    """
    now = datetime.now(timezone.utc)

    # Load contacts
    result = await db.execute(
        select(Contact).where(Contact.company_id == company.id)
    )
    contacts = list(result.scalars().all())

    industry_pts, industry_reason = _industry_score(
        company.naics_code, company.industry_segment, company.name
    )
    size_pts, size_reason = _size_score(company.employee_count)
    loc_pts, loc_reason = _location_score(company.country)
    completeness_pts, completeness_breakdown = _completeness_score(company, contacts)
    engagement_pts, engagement_breakdown = await _engagement_signal(db, company)

    # --- Signal recency decay ---
    # Find the most recent engagement event across all contacts at this company.
    velocity_signals, most_recent_event = await _engagement_velocity(db, company, now)
    recency = signal_recency_weight(most_recent_event, now)
    recency_bonus = int(recency * 8)  # up to 8 bonus points for fresh signals

    # --- Engagement velocity MULTIPLIER (not additive) ---
    # The live-engagement signal (real opens/clicks/replies + recency) is scaled
    # by how *fast* the lead is engaging. A burst of activity in the last 24h
    # multiplies that signal so a warming lead jumps, while a single stale open
    # is left untouched.
    v_mult = velocity_multiplier(velocity_signals)
    engagement_signal = engagement_pts + recency_bonus
    boosted_engagement = round(engagement_signal * v_mult)
    velocity_gain = boosted_engagement - engagement_signal  # points added by velocity

    # --- Buyer-intent signal bonus (Phase 2.1) — additive on top of fit ---
    intent_pts, intent_breakdown = await _intent_signal_bonus(db, company, now)

    # --- Account tier classification ---
    tier_name, tier_value = classify_account_tier(
        industry_pts, size_pts, loc_pts, engagement_pts
    )

    # firmographic fit + (completeness + velocity-boosted engagement signal) +
    # buyer-intent bonus. Velocity *multiplies* the live-engagement portion;
    # intent adds on top. The stored ``engagement_score`` column keeps its
    # historical meaning of "everything that isn't industry/size/location".
    non_firmographic_pts = completeness_pts + boosted_engagement
    fit_score = min(
        100, industry_pts + size_pts + loc_pts + non_firmographic_pts + intent_pts
    )

    # --- Score change history ---
    # Compare against the company's most recent prior score so the UI can show
    # "this lead's score jumped N points" without re-deriving it from raw rows.
    change = await _score_change(db, company.id, fit_score, now)

    factors = {
        "industry": {"score": industry_pts, "reason": industry_reason},
        "size": {"score": size_pts, "reason": size_reason},
        "location": {"score": loc_pts, "reason": loc_reason},
        "completeness": {"score": completeness_pts, "breakdown": completeness_breakdown},
        "engagement": {"score": engagement_pts, "breakdown": engagement_breakdown},
        "signal_recency": {
            "weight": recency,
            "bonus_points": recency_bonus,
            "most_recent_event": most_recent_event.isoformat() if most_recent_event else None,
        },
        "engagement_velocity": {
            "signals_in_24h": velocity_signals,
            "multiplier": v_mult,
            "engagement_signal": engagement_signal,
            "boosted_signal": boosted_engagement,
            "points_added": velocity_gain,
        },
        "intent_signals": {
            "bonus_points": intent_pts,
            **intent_breakdown,
        },
        "account_tier": {
            "tier": tier_name,
            "value": tier_value,
        },
        "score_change": change,
    }

    # --- Jev lead temperature (hot / warm / cold) ---
    # An independent second opinion from TypeSafe's Jev, stored beside the
    # rule-based breakdown. Absent (not null) when Jev is unconfigured or
    # unavailable, so the row is exactly the pre-Jev shape in that case.
    jev_temp = await jev_lead_temperature(
        company,
        contacts,
        signals_in_24h=velocity_signals,
        most_recent_event=most_recent_event,
        intent_breakdown=intent_breakdown,
    )
    if jev_temp is not None:
        factors["jev_temperature"] = jev_temp

    lead_score = LeadScore(
        company_id=company.id,
        score_type="rule_based",
        fit_score=fit_score,
        industry_score=industry_pts,
        size_score=size_pts,
        location_score=loc_pts,
        engagement_score=non_firmographic_pts,
        factors=factors,
    )
    db.add(lead_score)
    await db.flush()

    logger.info(
        "Scored company %s (%s): fit=%d [industry=%d, size=%d, location=%d, "
        "completeness=%d, engagement=%d, recency_bonus=%d, velocity_x%.2f (+%d), "
        "intent_bonus=%d, tier=%s, delta=%+d]",
        company.name, company.id, fit_score,
        industry_pts, size_pts, loc_pts, completeness_pts, engagement_pts,
        recency_bonus, v_mult, velocity_gain, intent_pts, tier_name, change["delta"],
    )
    return lead_score


async def score_company_ai(
    db: AsyncSession,
    company: Company,
    anthropic_client=None,
) -> LeadScore:
    """Use the LLM provider chain to produce an AI-driven lead score.

    ``anthropic_client`` is retained for the existing signature but is optional
    and defaults to the chain: passing a raw Anthropic handle pinned this to a
    single provider. The call below was also never awaited, so with a genuine
    async client this function raised on ``message.content`` every time — it has
    no callers, which is presumably why that went unnoticed.

    Sends company data to the model and asks it to evaluate fit for
    Apprend Technologies' manufacturing software products.
    """
    result = await db.execute(
        select(Contact).where(Contact.company_id == company.id)
    )
    contacts = list(result.scalars().all())

    contact_info = ""
    for c in contacts[:5]:
        contact_info += f"  - {c.full_name}, {c.title or 'N/A'}, {c.email or 'N/A'}\n"

    prompt = f"""You are a B2B lead scoring expert for Apprend Technologies, which makes software
for manufacturing companies (press operations, quality control, factory floor management).

Evaluate this company as a potential customer. Return a JSON object with:
- fit_score (0-100): overall likelihood they'd benefit from our software
- industry_score (0-40): how closely aligned to manufacturing
- size_score (0-25): whether size is right for our products
- location_score (0-20): US/Canada market fit
- engagement_score (0-15): data quality and contact accessibility
- reasoning: one paragraph explaining your score

Company Data:
- Name: {company.name}
- Domain: {company.domain or 'N/A'}
- Industry: {company.naics_description or 'N/A'} (NAICS: {company.naics_code or 'N/A'})
- Segment: {company.industry_segment or 'N/A'}
- Location: {company.city or ''}, {company.state_province or ''}, {company.country}
- Employees: {company.employee_count or 'Unknown'}
- Website: {company.website or 'N/A'}
Contacts:
{contact_info or '  None found'}

Respond with ONLY the JSON object, no markdown fences."""

    try:
        # Keep the AI scorer on the same current model as the rest of the app.
        from app.services.ai_service import get_ai_service
        from app.services.llm_providers import chain_client

        client = anthropic_client or chain_client(get_settings().anthropic_api_key)
        message = await client.messages.create(
            model=get_ai_service().model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        import json
        response_text = message.content[0].text.strip()
        # Tolerate markdown code fences around the JSON.
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        data = json.loads(response_text)

        ai_fit = max(0, min(100, data.get("fit_score", 0)))
        change = await _score_change(db, company.id, ai_fit)

        lead_score = LeadScore(
            company_id=company.id,
            score_type="ai",
            fit_score=ai_fit,
            industry_score=max(0, min(40, data.get("industry_score", 0))),
            size_score=max(0, min(25, data.get("size_score", 0))),
            location_score=max(0, min(20, data.get("location_score", 0))),
            engagement_score=max(0, min(15, data.get("engagement_score", 0))),
            factors={
                "ai_reasoning": data.get("reasoning", ""),
                "raw_response": data,
                "score_change": change,
            },
        )
        db.add(lead_score)
        await db.flush()

        logger.info("AI scored company %s: fit=%d", company.name, lead_score.fit_score)
        return lead_score

    except Exception as e:
        logger.error("AI scoring failed for %s: %s", company.name, str(e))
        # Fall back to rule-based
        return await score_company_rule_based(db, company)
