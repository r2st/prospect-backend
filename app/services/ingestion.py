"""The unified ingestion pipeline — one way into the database for every source.

Before this module there were four ingestion paths with four different dedup
rules:

===========================================  ==========================================
``enrichment.search_companies``              domain -> lower(name); committed internally
``autonomous_prospector._find_or_create``    company_id -> domain -> lower(name)
``import_export.import_csv``                 domain -> **exact** name (case-*sensitive*)
``enrichment.find_contacts``                 email set, scoped to one company
===========================================  ==========================================

The divergence was already producing bugs: CSV's case-sensitive name match
created "Acme Corp" alongside the "ACME CORP" the prospector would have matched,
and the per-company email scope duplicated a person who moved employers.

Every source now funnels through :meth:`IngestionService.ingest`:

    normalize -> dedup -> persist -> score -> route

Two rules this module holds to:

* **It owns no transaction.** ``ingest`` flushes; the task or router commits.
* **Only a certain match auto-merges.** "Precision Manufacturing" in Ohio and in
  Pune share a name and are two real companies. A name-only match is surfaced
  for a human, never silently folded together.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.constants import resolve_country
from app.models.company import Company
from app.models.contact import Contact
from app.models.prospecting import IcpProfile
from app.services.email_pattern_generator import get_pattern_generator
from app.services.normalize import normalize_domain, normalize_name_key, normalize_phone

logger = logging.getLogger(__name__)

# Dedup confidence, strongest first. Only ``certain`` may auto-merge into an
# existing company; ``high`` merges but cannot auto-enroll; ``medium`` creates a
# new company and asks a human. ``none`` is a clean new company.
DEDUP_CERTAIN = "certain"
DEDUP_HIGH = "high"
DEDUP_MEDIUM = "medium"
DEDUP_NONE = "none"

# Where a record ends up.
ROUTE_AUTO_ENROLL = "auto_enroll"
ROUTE_REVIEW = "review"
ROUTE_DISCARD = "discard"
ROUTE_IMPORT = "import"  # persisted without ICP routing (plain CSV import)


class IngestionError(Exception):
    """A record could not be ingested. Carries an operator-readable reason."""


@dataclass
class ContactRecord:
    """A person, normalized across Apollo / CSV / (later) Hunter and Places."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    title: Optional[str] = None
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    source: str = "unknown"
    raw: dict = field(default_factory=dict)

    @property
    def email_key(self) -> Optional[str]:
        return self.email.strip().lower() if self.email and self.email.strip() else None

    @property
    def has_name(self) -> bool:
        return bool(self.first_name or self.last_name)


@dataclass
class CandidateRecord:
    """A company, normalized across every source."""

    name: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[int] = None
    country: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    contacts: list[ContactRecord] = field(default_factory=list)
    source: str = "unknown"
    # Set when a caller already knows the company row (the prospector's old
    # ``company_id`` shortcut, kept so Apollo re-imports resolve for free).
    company_id: Optional[uuid.UUID] = None
    icp_profile_id: Optional[uuid.UUID] = None
    raw: dict = field(default_factory=dict)

    def as_candidate_dict(self) -> dict:
        """The dict shape ``AutonomousProspector.qualify`` consumes."""
        return {
            "name": self.name,
            "domain": self.domain,
            "industry": self.industry,
            "employee_count": self.employee_count,
            "country": self.country,
            "city": self.city,
            "state": self.state,
            "source": self.source,
        }


@dataclass
class ContactOutcome:
    """What happened to one person."""

    record: ContactRecord
    contact: Optional[Contact] = None
    created: bool = False
    # duplicate_email | duplicate_name | suppressed | no_identity | None
    reason: Optional[str] = None


@dataclass
class DedupMatch:
    """The outcome of resolving a record against existing companies."""

    company: Optional[Company]
    confidence: str
    matched_on: Optional[str] = None

    @property
    def is_match(self) -> bool:
        return self.company is not None

    @property
    def can_auto_merge(self) -> bool:
        """Only a certain match may be silently folded into an existing row."""
        return self.confidence in (DEDUP_CERTAIN, DEDUP_HIGH)

    def as_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "matched_on": self.matched_on,
            "matched_company_id": str(self.company.id) if self.company else None,
        }


@dataclass
class IngestionOutcome:
    """What happened to one record."""

    record: CandidateRecord
    route: str
    dedup: DedupMatch
    company: Optional[Company] = None
    created: bool = False
    contacts_created: int = 0
    contacts_skipped: int = 0
    fit_score: int = 0
    qualification: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def company_id(self) -> Optional[uuid.UUID]:
        return self.company.id if self.company is not None else None

    def as_dict(self) -> dict:
        return {
            "name": self.record.name,
            "domain": self.record.domain,
            "route": self.route,
            "created": self.created,
            "company_id": str(self.company_id) if self.company_id else None,
            "dedup": self.dedup.as_dict(),
            "fit_score": self.fit_score,
            "contacts_created": self.contacts_created,
            "error": self.error,
        }


@dataclass
class IngestionResult:
    outcomes: list[IngestionOutcome] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def companies_created(self) -> int:
        return sum(1 for o in self.outcomes if o.created)

    @property
    def companies_matched(self) -> int:
        return sum(1 for o in self.outcomes if o.dedup.is_match and not o.created)

    @property
    def contacts_created(self) -> int:
        return sum(o.contacts_created for o in self.outcomes)

    @property
    def created_company_ids(self) -> list[str]:
        return [str(o.company_id) for o in self.outcomes if o.created and o.company_id]

    @property
    def duplicates(self) -> int:
        """Records that matched something below `certain` — the review pile."""
        return sum(1 for o in self.outcomes if o.dedup.confidence == DEDUP_MEDIUM)


class IngestionService:
    """The single entry point every source ingests through."""

    def __init__(self) -> None:
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # 1. Normalization
    # ------------------------------------------------------------------
    def normalize_country(self, raw: Optional[str]) -> Optional[str]:
        """Validate a country against market policy.

        Returns None when nothing was supplied (the caller defaults it). Raises
        :class:`IngestionError` for a country we do not support — **never
        relabels it**. The old paths coerced anything non-US/CA to ``"US"``,
        which quietly turned an Indian company into an American one; a rejected
        row is recoverable, a mislabelled one is not.

        Resolution goes through :func:`constants.resolve_country` rather than a
        two-character slice. Apollo reports a full country name, and the slice
        turned "United States" into ``"UN"`` and "Germany" into ``"GE"`` — so
        the record was refused for being unsupported, naming a code no source
        ever sent. The error now quotes what the source actually said, because
        "country 'UN' is not supported" sent every reader looking at Apollo.
        """
        if raw is None:
            return None
        original = str(raw).strip()
        if not original:
            return None
        supported = [c.upper() for c in self.settings.supported_countries]
        value = resolve_country(original)
        if value is None or value not in supported:
            raise IngestionError(
                f"country '{original}' is not supported "
                f"(supported: {', '.join(supported)})"
            )
        return value

    def normalize(self, record: CandidateRecord) -> CandidateRecord:
        """Canonicalize a record in place and return it.

        Raises :class:`IngestionError` when the record can't be ingested at all.
        """
        name = (record.name or "").strip()
        if not name:
            raise IngestionError("company name is required")
        record.name = name
        record.domain = normalize_domain(record.domain or record.website)
        record.phone = normalize_phone(record.phone)
        record.country = self.normalize_country(record.country)
        if record.employee_count is not None:
            try:
                record.employee_count = int(record.employee_count)
            except (TypeError, ValueError):
                record.employee_count = None
        return record

    # ------------------------------------------------------------------
    # 2. Cross-source deduplication
    # ------------------------------------------------------------------
    async def resolve_company(
        self, db: AsyncSession, record: CandidateRecord
    ) -> DedupMatch:
        """Find the existing company this record refers to. Never writes.

        Strongest signal first. The ordering is the whole point: a domain is an
        identity, a name is a coincidence waiting to happen.

        Re-normalizes the fields it compares rather than trusting the caller to
        have run :meth:`normalize` first — every helper here is idempotent, and
        a public resolver that silently mismatches raw input is a trap.
        """
        # 0. The caller already resolved it (Apollo re-import handing us an id).
        if record.company_id:
            company = (
                await db.execute(select(Company).where(Company.id == record.company_id))
            ).scalar_one_or_none()
            if company is not None:
                return DedupMatch(company, DEDUP_CERTAIN, "company_id")

        # 1. Domain exact (normalized). A shared domain is the same company.
        domain = normalize_domain(record.domain or record.website)
        if domain:
            company = (
                await db.execute(select(Company).where(Company.domain == domain))
            ).scalar_one_or_none()
            if company is not None:
                return DedupMatch(company, DEDUP_CERTAIN, "domain")

        name_key = normalize_name_key(record.name)
        if not name_key:
            return DedupMatch(None, DEDUP_NONE)

        # Candidates sharing the canonical name. Matched case-insensitively via
        # the stored key — the CSV path's old exact-match was the bug that let
        # "Acme Corp" and "ACME CORP" both exist.
        rows = list(
            (
                await db.execute(select(Company).where(Company.name_key == name_key))
            ).scalars().all()
        )
        if not rows:
            return DedupMatch(None, DEDUP_NONE)

        # 2. name_key + country + state. Same name in the same place.
        if record.country and record.state:
            country_key = record.country.strip().upper()
            state_key = record.state.strip().casefold()
            for c in rows:
                if (
                    c.country
                    and c.country.upper() == country_key
                    and c.state_province
                    and c.state_province.strip().casefold() == state_key
                ):
                    return DedupMatch(c, DEDUP_HIGH, "name_key+country+state")

        # 3. name_key + phone. Same name, same switchboard.
        phone = normalize_phone(record.phone)
        if phone:
            for c in rows:
                if c.phone and normalize_phone(c.phone) == phone:
                    return DedupMatch(c, DEDUP_HIGH, "name_key+phone")

        # 4. name_key alone — **medium, do not auto-merge**. "Precision
        # Manufacturing" in Ohio and in Pune are different companies; folding
        # them together destroys a real prospect. Surface it for a human.
        return DedupMatch(rows[0], DEDUP_MEDIUM, "name_key")

    # ------------------------------------------------------------------
    # 3. Contact dedup + persistence
    # ------------------------------------------------------------------
    async def _find_contact_by_email(
        self, db: AsyncSession, email_key: str
    ) -> Optional[Contact]:
        """Look a person up by email **globally**, not per company.

        The old per-company scope duplicated anyone who changed employers, and —
        worse — let a re-import re-create a contact who had been suppressed.
        """
        return (
            await db.execute(
                select(Contact)
                .where(func.lower(Contact.email) == email_key)
                .limit(1)
            )
        ).scalar_one_or_none()

    async def ingest_contacts(
        self,
        db: AsyncSession,
        company: Company,
        records: list[ContactRecord],
        *,
        default_source: str = "unknown",
    ) -> list[ContactOutcome]:
        """Create people under ``company``, deduplicated by email **globally**.

        This is the fourth old ingestion path, collapsed: Apollo's
        ``find_contacts``/``discover_executives`` each kept their own
        per-company email set, so the same person at two companies produced two
        rows, and a re-import happily resurrected a suppressed contact.
        """
        outcomes: list[ContactOutcome] = []
        created_any = False
        # Track within-batch emails too, so one payload carrying the same person
        # twice doesn't insert them twice.
        seen: set[str] = set()
        for cr in records:
            if not cr.has_name and not cr.email_key:
                outcomes.append(ContactOutcome(record=cr, reason="no_identity"))
                continue
            key = cr.email_key
            if key:
                if key in seen:
                    outcomes.append(ContactOutcome(record=cr, reason="duplicate_email"))
                    continue
                seen.add(key)
                existing = await self._find_contact_by_email(db, key)
                if existing is not None:
                    # Already known. The case that matters most: a suppressed
                    # contact must never be resurrected by a re-import — leave
                    # the existing row exactly as it is.
                    outcomes.append(
                        ContactOutcome(
                            record=cr,
                            contact=existing,
                            reason="suppressed"
                            if getattr(existing, "suppressed", False)
                            else "duplicate_email",
                        )
                    )
                    continue
            else:
                # No address to dedup on: match the person by name at this
                # company. Without this, every discovery run that returned the
                # same company re-inserted its people — Stripe and Shopify
                # each held 15 rows for 3 invented executives after five runs.
                existing = await self._find_named_contact(db, company.id, cr)
                if existing is not None:
                    outcomes.append(
                        ContactOutcome(record=cr, contact=existing, reason="duplicate_name")
                    )
                    continue
            contact = Contact(
                company_id=company.id,
                first_name=cr.first_name,
                last_name=cr.last_name,
                email=cr.email or None,
                title=cr.title,
                phone=cr.phone,
                linkedin_url=cr.linkedin_url,
                source=cr.source or default_source,
            )
            db.add(contact)
            created_any = True
            outcomes.append(ContactOutcome(record=cr, contact=contact, created=True))
        if created_any:
            await db.flush()
        return outcomes

    async def _find_named_contact(
        self, db: AsyncSession, company_id, cr: ContactRecord
    ) -> Optional[Contact]:
        """An existing contact at ``company_id`` with the same first and last name."""
        first = (cr.first_name or "").strip().lower()
        last = (cr.last_name or "").strip().lower()
        if not first and not last:
            return None
        stmt = select(Contact).where(
            Contact.company_id == company_id,
            func.lower(func.coalesce(Contact.first_name, "")) == first,
            func.lower(func.coalesce(Contact.last_name, "")) == last,
        )
        return (
            await db.execute(stmt.order_by(Contact.created_at, Contact.id).limit(1))
        ).scalars().first()

    async def discover_contact_emails(
        self,
        db: AsyncSession,
        company: Company,
        contacts: list[Contact],
    ) -> int:
        """Best-effort: derive addresses for named contacts that arrived without one.

        Apollo hands back people with ``email: null`` constantly and Places gives
        us businesses with no people at all, so ``ingest_contacts`` regularly
        persists a row that names a decision-maker we have no way to reach.
        This converts some of those dead rows into contactable ones by guessing
        the address from the name + the company's domain — see
        ``email_pattern_generator``. Returns the number of contacts filled in.

        **Called explicitly, not folded into ``ingest_contacts``.** Discovery does
        DNS and (optionally) SMTP I/O per person, and ``ingest_contacts`` runs
        inside the caller's open transaction — putting network probes in that
        loop would hold a transaction open across a stranger's mail server, and
        make a 500-row CSV import take minutes. Callers opt in once the cheap
        work is committed.

        Keeps this module's rules: owns no transaction (flushes, never commits),
        and never invents deliverability — an unproven guess lands with
        ``email_status='unknown'``, which the send paths treat as sendable-by-a-
        human but never auto-enrollable.
        """
        settings = get_settings()
        if not settings.pattern_discovery_enabled:
            return 0
        domain = getattr(company, "domain", None)
        if not domain:
            # No domain, nothing to build a local part onto. Places-sourced
            # companies often land here — that's P4's scrape/domain-search job.
            return 0

        generator = get_pattern_generator()
        filled = 0
        for contact in contacts:
            if contact.email:
                continue
            if not (contact.first_name and contact.last_name):
                # A lone first name generates 'first@' for every junior at the
                # company — collides across people and isn't worth a probe.
                continue

            verdict = await generator.discover(
                contact.first_name, contact.last_name, domain
            )
            email = verdict.get("email")
            if verdict.get("status") != "ok" or not email:
                continue

            # A dead domain says nothing about the person — leave the row alone
            # rather than stamping a contact 'invalid' over an address we made up.
            if verdict.get("result") == "undeliverable":
                continue

            # The global email dedup rule applies to discovered addresses too.
            # Guessing an address that already belongs to someone would hand two
            # contacts one mailbox and mail them both — and worse, could
            # resurrect a suppressed person under a new row.
            existing = await self._find_contact_by_email(db, email.strip().lower())
            if existing is not None:
                logger.info(
                    "Pattern discovery guessed %s for contact %s, but it already "
                    "belongs to contact %s — leaving unset",
                    email, contact.id, existing.id,
                )
                continue

            contact.email = email
            contact.email_status = verdict.get("email_status")
            contact.email_checked_at = datetime.now(timezone.utc)
            contact.email_verification_result = verdict.get("result")
            contact.email_verification_source = verdict.get("source")
            contact.email_verification_meta = verdict.get("meta") or {}
            # `email_verified_at` stays NULL unless a probe actually confirmed the
            # mailbox, so the Hunter sweep still picks a guess up and re-checks it.
            if verdict.get("email_status") == "valid":
                contact.email_verified_at = datetime.now(timezone.utc)
            filled += 1

        if filled:
            await db.flush()
        return filled

    def _apply_fields(self, company: Company, record: CandidateRecord) -> None:
        """Fill blanks on an existing company. Never overwrites real data."""
        if record.domain and not company.domain:
            company.domain = record.domain
        if record.industry and not company.industry_segment:
            company.industry_segment = record.industry
        if record.employee_count and not company.employee_count:
            company.employee_count = record.employee_count
        if record.city and not company.city:
            company.city = record.city
        if record.state and not company.state_province:
            company.state_province = record.state
        if record.phone and not company.phone:
            company.phone = record.phone
        if record.website and not company.website:
            company.website = record.website
        # Backfill the match key for rows created before it existed.
        if not company.name_key:
            company.name_key = normalize_name_key(company.name)

    async def _create_company(
        self, db: AsyncSession, record: CandidateRecord
    ) -> Company:
        company = Company(
            name=record.name,
            name_key=normalize_name_key(record.name),
            domain=record.domain,
            industry_segment=record.industry,
            employee_count=record.employee_count,
            # Absent country defaults; an *unsupported* one already raised in
            # normalize(). Defaulting a blank is not relabelling.
            country=record.country or "US",
            city=record.city,
            state_province=record.state,
            phone=record.phone,
            website=record.website,
            source=record.source or "unknown",
            stage="new",
            pipeline_stage="new_lead",
        )
        db.add(company)
        await db.flush()
        return company

    # ------------------------------------------------------------------
    # 4. Routing
    # ------------------------------------------------------------------
    def route(
        self,
        qualification: Optional[dict],
        dedup: DedupMatch,
        *,
        enroll_budget: int = 0,
    ) -> str:
        """Decide where a record goes.

        Beyond the fit thresholds the prospector already applied, **dedup
        confidence can veto auto-enroll**: enrolling a sequence against a
        company we are not sure we identified is how two prospects become one
        annoyed recipient.
        """
        if qualification is None:
            return ROUTE_IMPORT
        if qualification.get("excluded"):
            return ROUTE_DISCARD
        s = self.settings
        fit = qualification.get("fit_score", 0)
        if fit >= s.prospecting_auto_enroll_threshold:
            if dedup.confidence == DEDUP_MEDIUM:
                # Fit alone is not enough — we're unsure who this even is.
                return ROUTE_REVIEW
            if enroll_budget > 0:
                return ROUTE_AUTO_ENROLL
            return ROUTE_REVIEW
        if fit >= s.prospecting_review_threshold:
            return ROUTE_REVIEW
        return ROUTE_DISCARD

    # ------------------------------------------------------------------
    # 5. Orchestration
    # ------------------------------------------------------------------
    async def ingest_one(
        self,
        db: AsyncSession,
        record: CandidateRecord,
        *,
        icp: Optional[IcpProfile] = None,
        enroll_budget: int = 0,
        dry_run: bool = False,
        score: bool = False,
    ) -> IngestionOutcome:
        """Normalize, dedup, persist, score and route a single record."""
        record = self.normalize(record)
        dedup = await self.resolve_company(db, record)

        qualification = None
        if icp is not None:
            from app.services.autonomous_prospector import get_prospector

            qualification = get_prospector().qualify(record.as_candidate_dict(), icp)

        route = self.route(qualification, dedup, enroll_budget=enroll_budget)
        outcome = IngestionOutcome(
            record=record,
            route=route,
            dedup=dedup,
            fit_score=(qualification or {}).get("fit_score", 0),
            qualification=qualification or {},
        )

        # A dry run answers "what would happen" and must touch nothing.
        if dry_run:
            outcome.company = dedup.company
            return outcome

        if route == ROUTE_DISCARD:
            return outcome

        if dedup.is_match and dedup.can_auto_merge:
            company = dedup.company
            self._apply_fields(company, record)
            await db.flush()
        else:
            # No match, or a medium one we refuse to auto-merge: this is its own
            # company until a human says otherwise.
            company = await self._create_company(db, record)
            outcome.created = True

        outcome.company = company
        contact_outcomes = await self.ingest_contacts(
            db, company, record.contacts, default_source=record.source
        )
        outcome.contacts_created = sum(1 for c in contact_outcomes if c.created)
        outcome.contacts_skipped = sum(1 for c in contact_outcomes if not c.created)

        if score:
            await self._score(db, company)
        return outcome

    async def _score(self, db: AsyncSession, company: Company) -> None:
        """Seed the real lead score post-persistence.

        ``qualify()`` answers "does this match the ICP?" before import;
        ``scoring.py`` answers "how good is this lead?" and needs a persisted
        company with contacts. Both are wanted — this is the wiring that was
        missing, so only CSV imports ever got a lead score.

        Best-effort: a scoring failure must not lose the ingested company.
        """
        try:
            from app.services.scoring import score_company_rule_based

            await score_company_rule_based(db, company)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Post-ingest scoring failed for %s: %s", company.name, e)

    async def ingest(
        self,
        db: AsyncSession,
        records: list[CandidateRecord],
        *,
        icp: Optional[IcpProfile] = None,
        enroll_budget: int = 0,
        dry_run: bool = False,
        score: bool = False,
    ) -> IngestionResult:
        """Ingest a batch. One bad record never fails the batch.

        ``score`` runs the rule-based scorer inline per company — right for a
        prospecting run's ~25 candidates, wrong for a 5k-row CSV, which should
        keep dispatching a batched Celery pass over ``created_company_ids``.
        """
        result = IngestionResult()
        for record in records:
            try:
                outcome = await self.ingest_one(
                    db,
                    record,
                    icp=icp,
                    enroll_budget=enroll_budget,
                    dry_run=dry_run,
                    score=score,
                )
            except IngestionError as e:
                result.errors.append(f"{record.name}: {e}")
                result.outcomes.append(
                    IngestionOutcome(
                        record=record,
                        route=ROUTE_DISCARD,
                        dedup=DedupMatch(None, DEDUP_NONE),
                        error=str(e),
                    )
                )
                continue
            except Exception as e:
                logger.exception("Ingestion failed for %s", record.name)
                result.errors.append(f"{record.name}: {e}")
                result.outcomes.append(
                    IngestionOutcome(
                        record=record,
                        route=ROUTE_DISCARD,
                        dedup=DedupMatch(None, DEDUP_NONE),
                        error=str(e),
                    )
                )
                continue
            if outcome.route == ROUTE_AUTO_ENROLL:
                enroll_budget = max(0, enroll_budget - 1)
            result.outcomes.append(outcome)
        return result


_ingestion: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    global _ingestion
    if _ingestion is None:
        _ingestion = IngestionService()
    return _ingestion
