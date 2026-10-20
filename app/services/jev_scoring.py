"""Lead temperature (hot / warm / cold) from Jev.

The rule-based scorer in ``app.services.scoring`` answers "how well does this
company fit?" with a 0-100 number assembled from fixed point tables. Jev answers
a different, coarser question — "should someone contact this lead now?" — from
the same raw lead data, and returns calibrated probabilities with it.

The result is stored alongside the rule-based score in
``LeadScore.factors["jev_temperature"]`` rather than replacing any column: it is
an independent second opinion, it needs no migration, and when Jev is
unavailable the score row is exactly what it was before Jev existed.

The rule-based fit number is deliberately **not** put in the state. Fed its own
conclusion, Jev would mostly echo it back, and the point of asking is to get a
signal the point tables can't produce.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from app.models.company import Company
from app.models.contact import Contact
from app.services import jev_service

# How many contacts' titles go into the state — enough to show who we can reach
# at the company without turning one lead into a long prompt.
MAX_CONTACTS = 8


def _contact_line(c: Contact) -> str:
    parts = [c.full_name or "Unknown", c.title or "no title"]
    if c.department:
        parts.append(f"dept {c.department}")
    if not c.email:
        parts.append("no email")
    return " — ".join(parts)


def build_lead_state(
    company: Company,
    contacts: Sequence[Contact],
    *,
    signals_in_24h: int = 0,
    most_recent_event: Optional[datetime] = None,
    intent_breakdown: Optional[dict] = None,
) -> str:
    """Describe a lead as plain text for Jev."""
    location = ", ".join(
        p for p in (company.city, company.state_province, company.country) if p
    )
    lines = [
        "A B2B company being considered for outbound sales outreach.",
        f"Company: {company.name}",
        f"Domain: {company.domain or 'unknown'}",
        f"Industry: {company.naics_description or 'unknown'}"
        + (f" (NAICS {company.naics_code})" if company.naics_code else ""),
        f"Segment: {company.industry_segment or 'unknown'}",
        f"Employees: {company.employee_count if company.employee_count is not None else 'unknown'}",
        f"Location: {location or 'unknown'}",
        f"Pipeline stage: {company.stage}",
    ]
    if company.tags:
        lines.append(f"Tags: {', '.join(company.tags[:10])}")

    if contacts:
        lines.append(f"Contacts ({len(contacts)}):")
        lines.extend(f"  - {_contact_line(c)}" for c in contacts[:MAX_CONTACTS])
    else:
        lines.append("Contacts: none on file")

    lines.append(
        f"Engagement events in the last 24 hours: {signals_in_24h}"
    )
    lines.append(
        "Most recent engagement: "
        + (most_recent_event.isoformat() if most_recent_event else "never")
    )
    # Shape of ``scoring._intent_signal_bonus``'s breakdown; empty when none.
    intent = intent_breakdown or {}
    if intent.get("active_signals"):
        lines.append(
            f"Active buyer-intent signals: {intent['active_signals']} "
            f"({', '.join(intent.get('types') or [])})"
        )
    return "\n".join(lines)


async def jev_lead_temperature(
    company: Company,
    contacts: Sequence[Contact],
    **signals,
) -> Optional[dict]:
    """The ``factors["jev_temperature"]`` entry for a lead, or ``None``."""
    state = build_lead_state(company, contacts, **signals)
    result = await jev_service.classify_lead_temperature(state)
    return result.as_dict() if result is not None else None
