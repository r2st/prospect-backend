"""Shared test fixtures for Landline backend tests."""

import os

# A signing secret must exist before importing app modules — the app now
# refuses to run without one, and tests mint JWTs. Set it first.
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
# Disable the short-TTL response cache in tests — a hit from a prior test's data
# would leak across the per-test in-memory DBs. The cache is verified separately.
os.environ.setdefault("RESPONSE_CACHE_ENABLED", "false")
# Jev (TypeSafe AI) is consulted on the scoring, reply and send paths whenever a
# key is configured. Force it off so a key in .env or the shell can never make
# the suite call the real API; Jev tests enable it explicitly with a mock.
os.environ["TYPESAFE_API_KEY"] = ""

import asyncio
import json
import uuid
import uuid as uuid_mod
from datetime import datetime, timezone, timedelta
from typing import Optional

import pytest
import pytest_asyncio
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import Text, String, event, text, TypeDecorator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from httpx import AsyncClient, ASGITransport

from app.database import Base, get_db
from app.dependencies import get_current_user, get_current_user_optional
from app.main import app
from app.models.company import Company
from app.models.contact import Contact
from app.models.consent import ConsentRecord
from app.models.user import User, UserRole
from app.rate_limit import limiter
from app.security import create_access_token, hash_password

# Rate limiting is opt-in per test — disable globally so unrelated tests that
# hammer an endpoint don't trip a 429. The dedicated rate-limit test re-enables
# it. (See tests/test_security_hardening.py.)
limiter.enabled = False


# Use SQLite for tests (async via aiosqlite)
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


# ---------------------------------------------------------------------------
# ``CAST(ts AS DATE)`` means something different in SQLite, and it fails loudly
# ---------------------------------------------------------------------------
# SQLite has no DATE type. ``CAST('2026-08-02 09:00:00+00:00' AS DATE)`` falls
# through to NUMERIC affinity, which reads the leading digits and yields the
# integer ``2026``; SQLAlchemy's Date result processor then raises
# ``TypeError: fromisoformat: argument must be str``.
#
# The cost isn't a confusing error, it's the coverage hole it creates. Any
# endpoint grouping by day — a shape every "daily sends" chart in this codebase
# uses — cannot be called at all under the test harness, so it doesn't get
# tested, so its *other* logic doesn't get tested either. ``get_campaign_
# analytics`` had zero tests for exactly this reason, and shipped four
# percentages divided by a denominator that counted mail the kill switch had
# stopped.
#
# ``date(x)`` is valid in both dialects (Postgres accepts function-style casts
# for type names) and returns an ISO string SQLite's own processor parses, so
# rewriting the cast here makes the harness agree with production rather than
# papering over a difference.
from sqlalchemy import Date as _SADate  # noqa: E402
from sqlalchemy.ext.compiler import compiles as _compiles  # noqa: E402
from sqlalchemy.sql.elements import Cast as _Cast  # noqa: E402


@_compiles(_Cast, "sqlite")
def _sqlite_cast_to_date(element, compiler, **kw):
    if isinstance(element.type, _SADate):
        return "date(%s)" % compiler.process(element.clause, **kw)
    return compiler.visit_cast(element, **kw)


class SQLiteUUID(TypeDecorator):
    """Store UUID as a plain string in SQLite."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return str(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return uuid_mod.UUID(value)
        return value


class SQLiteJSON(TypeDecorator):
    """Store dicts/lists as JSON strings in SQLite."""
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value


# --- SQLite compatibility: patch PG-only types before table creation ---

_patched = False


def _patch_pg_types_for_sqlite():
    """Replace PostgreSQL-specific column types (ARRAY, JSONB, UUID)
    with SQLite-compatible types across all mapped tables, and strip
    all check constraints that use PG-specific SQL syntax."""
    global _patched
    if _patched:
        return
    _patched = True

    from sqlalchemy import CheckConstraint

    for table in Base.metadata.tables.values():
        # Remove check constraints (they use PG syntax like IN (...))
        to_remove = [
            c for c in table.constraints
            if isinstance(c, CheckConstraint)
        ]
        for c in to_remove:
            table.constraints.discard(c)

        # Patch column types by class name (avoids PG vs generic import issues)
        for col in table.columns:
            type_name = type(col.type).__name__
            if type_name == "ARRAY":
                col.type = SQLiteJSON()
            elif type_name in ("JSONB", "JSON"):
                col.type = SQLiteJSON()
            elif type_name == "UUID":
                col.type = SQLiteUUID()

            # Remove server_default that uses PG functions (gen_random_uuid, now())
            if col.server_default is not None:
                sd_text = str(getattr(col.server_default, "arg", ""))
                if "gen_random_uuid" in sd_text or "uuid" in sd_text.lower():
                    col.server_default = None

    # Freeze the mappers *now*, while the patched (SQLite-compatible) column
    # types are in place. SQLAlchemy snapshots each column's type into the
    # InstrumentedAttribute comparator the first time the mapper is configured;
    # that comparator is what generates bind parameters for ``Model.col == x``.
    # If a mapper is configured *before* this patch runs (e.g. a pure-unit test
    # module that imports a model triggers global ``configure_mappers()`` before
    # any ``db_engine`` fixture is instantiated), the comparator would freeze the
    # original PostgreSQL ``UUID`` type and serialize UUID binds with the wrong
    # dialect processor — every ``WHERE uuid_col == :param`` then silently
    # matches nothing under the SQLite shim, poisoning later DB tests with empty
    # reads and spurious 401s. Configuring here, at conftest import time (before
    # any test runs), makes the ordering deterministic.
    from sqlalchemy.orm import configure_mappers

    configure_mappers()


# Patch PG-only types at import time — conftest is imported before any test is
# collected or run, so this executes before any model usage can configure the
# mappers with the un-patched native types. See _patch_pg_types_for_sqlite.
_patch_pg_types_for_sqlite()


@pytest.fixture(autouse=True)
def _send_window_open_by_default():
    """Hold the business-hours send window open for every test that doesn't want it.

    ``send_sequence_step`` now refuses to send outside the configured window, so
    without this the entire send path is wall-clock dependent: the suite passes
    on a Tuesday morning and fails on a Sunday, and — worse — the gates *after*
    the window check go untested whenever CI happens to run at 3 AM. A test
    should fail because the code is wrong, not because of when it ran.

    Tests that are *about* the window re-enable it (see
    tests/test_contact_timezone_window.py) — a later monkeypatch wins over this
    one, so opting back in costs one line.
    """
    from app.config import get_settings

    settings = get_settings()
    original = settings.send_window_enabled
    settings.send_window_enabled = False
    try:
        yield
    finally:
        settings.send_window_enabled = original


@pytest.fixture(autouse=True)
def _isolated_send_counter(monkeypatch):
    """Give every test its own daily-send counter instead of the machine's.

    ``SendLimiter.reserve_daily_send`` claims a slot with an ``INCR`` on
    ``dialtone:daily_sends:<org>:<day>`` in **real Redis**, which on a developer
    box is reachable — so that key was shared by every test in the suite *and*
    by every previous run on the same calendar day. Around a hundred sends in,
    the cap is spent and the next test that sends is refused with
    ``Daily limit reached (100/100)``.

    The symptoms are what make it worth a fixture rather than a note. It is a
    function of machine state and time of day, so the suite is green on CI and
    on a fresh container and red on a laptop by mid-afternoon; *which* test
    fails is decided by execution order, so it moves between runs and reads as
    four unrelated flakes; and it fails an assertion about the sending code
    (``status == 'sent'``) rather than anything about limits, so it looks like a
    regression in whatever it landed on.

    A per-test fake, not a disabled limiter: the reservation is the mechanism
    that makes the cap correct under concurrency, and a suite that stubs it out
    stops testing it. Tests that want to drive Redis behaviour themselves patch
    ``_get_redis`` again in their own fixture (``test_send_limit_race`` and
    friends), and a later monkeypatch wins.
    """
    from app.services import send_limiter as _sl

    class _CounterRedis:
        """The commands the limiter uses, over a dict private to one test.

        Real Redis runs commands on one thread, so ``INCR`` is indivisible;
        asyncio gives the same guarantee to a synchronous section between
        awaits, which is what these are — so the concurrency the limiter
        depends on is reproduced, not bypassed.
        """

        def __init__(self):
            self.store: dict[str, int] = {}
            self.ttl_store: dict[str, int] = {}

        async def set(self, key, value, nx=False, ex=None):
            if nx and key in self.store:
                return None
            self.store[key] = int(value)
            if ex is not None:
                self.ttl_store[key] = ex
            return True

        async def incr(self, key):
            self.store[key] = self.store.get(key, 0) + 1
            return self.store[key]

        async def decr(self, key):
            self.store[key] = self.store.get(key, 0) - 1
            return self.store[key]

        async def get(self, key):
            value = self.store.get(key)
            return None if value is None else str(value)

        async def ttl(self, key):
            return self.ttl_store.get(key, -1)

        async def expire(self, key, seconds):
            self.ttl_store[key] = seconds
            return True

    client = _CounterRedis()

    async def _get_redis():
        return client

    monkeypatch.setattr(_sl, "_get_redis", _get_redis)
    # The module caches its client in globals; clear them so a test that did
    # reach real Redis before this fixture existed cannot leave one behind.
    _sl._reset_redis_for_tests()
    yield client
    _sl._reset_redis_for_tests()


@pytest.fixture(autouse=True, scope="session")
def _no_ambient_google_oauth():
    """Google OAuth is *unconfigured* in tests unless a test configures it.

    Several tests assert the unconfigured behaviour — ``/gmail/oauth/authorize``
    returns 503, ``/gmail/status`` reports ``configured: false`` — and
    ``Settings`` reads ``.env``. So whether they passed depended on whether the
    developer's machine happened to hold real credentials: green on CI and on a
    fresh clone, red the moment anyone put a working client id in ``.env``
    (which is exactly what rotating the Google project required). The test was
    asserting a property of the *machine*, not of the code.

    This is the mirror of the ``_google_configured`` fixture that already exists
    in ``tests/test_gmail.py``: that one turns the credentials on for the tests
    that need them, and nothing turned them off for the tests that need them
    off. Session-scoped so it is in place before any function-scoped fixture
    saves the value to restore.
    """
    from app.config import get_settings

    settings = get_settings()
    previous = (settings.google_client_id, settings.google_client_secret)
    settings.google_client_id = ""
    settings.google_client_secret = ""
    try:
        yield
    finally:
        settings.google_client_id, settings.google_client_secret = previous


@pytest.fixture(autouse=True)
def _reset_llm_breaker():
    """Clear the process-wide LLM circuit breaker and rate limiter around every test.

    Both are deliberately long-lived in production — a provider that is out of
    quota stays skipped for the cooldown so a batch send doesn't pay its timeout
    on every email, and the rate-limit window has to span calls to mean anything.
    Under pytest that same longevity means one test's simulated provider failures
    trip the breaker for every test that follows, which reads as a mysterious
    cascade of "provider skipped" failures in files that never touched the
    breaker at all.

    The rate limiter fails the same way but more quietly: its window is a rolling
    minute, so a file that makes more mocked calls than the configured RPM starts
    deferring providers partway through, and a test that simulates a 429 parks
    that provider for every test that runs in the next 20 seconds.
    """
    from app.services.llm_providers import reset_breaker
    from app.services.llm_rate_limit import reset_rate_limiter

    reset_breaker()
    reset_rate_limiter()
    yield
    reset_breaker()
    reset_rate_limiter()


@pytest_asyncio.fixture(autouse=True)
async def _reset_org_context():
    """Clear the multi-tenant org contextvar around every test.

    In production each request runs in its own asyncio task, so the org context
    set by ``get_current_user`` is task-local and never leaks. Under pytest's
    session-scoped loop, tests can share a context, so a prior API request's org
    could bleed into a later direct-``db_session`` test and skew the global read
    filter. Async (not sync) so the reset runs in the same task/context as the
    test body. Resetting to ``None`` before and after keeps tests isolated.
    """
    from app.org_context import set_current_org_id

    set_current_org_id(None)
    yield
    set_current_org_id(None)


@pytest_asyncio.fixture(scope="function")
async def db_engine():
    """Create a fresh test database engine per test."""
    _patch_pg_types_for_sqlite()

    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed the first-party "Apprend Technologies" org so the multi-tenant layer
    # has a real org for fixtures/users to belong to (mirrors migration 078).
    from app.models.tenant import Organization
    from app.org_context import APPREND_ORG_ID, APPREND_ORG_SLUG

    seed_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with seed_factory() as s:
        s.add(Organization(
            id=APPREND_ORG_ID, name="Apprend Technologies",
            slug=APPREND_ORG_SLUG, is_active=True, billing_plan="pro",
        ))
        await s.commit()

    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(db_engine):
    """Create a database session for a test."""
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


async def _install_test_db(db_engine):
    """Wire the app to the test engine by overriding get_db.

    Returns the session_factory and a teardown callable.
    """
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    async def override_get_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    def teardown():
        app.dependency_overrides.clear()

    return session_factory, teardown


@pytest_asyncio.fixture(scope="function")
async def unauth_client(db_engine):
    """HTTP client that sends NO credentials — for asserting 401 on protected
    routes and exercising the real auth path."""
    _, teardown = await _install_test_db(db_engine)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    teardown()


@pytest_asyncio.fixture(scope="function")
async def client(db_engine):
    """HTTP test client, authenticated by default as a seeded admin.

    The API now fails closed (every non-public route needs credentials), so the
    default client carries an admin Bearer token to keep existing tests working
    without threading auth through every call. Tests that need a specific
    role/user pass their own ``Authorization`` header per request (which
    overrides this default), and ``unauth_client`` sends nothing at all.
    """
    session_factory, teardown = await _install_test_db(db_engine)

    # Seed a default admin so the real get_current_user + middleware checks pass.
    async with session_factory() as session:
        from app.org_context import APPREND_ORG_ID
        admin = User(
            id=uuid.uuid4(),
            email="_default_test_admin@landline.app",
            name="Default Test Admin",
            role=UserRole.ADMIN.value,
            password_hash=hash_password("defaultpass123"),
            is_active=True,
            # Belong the default admin to the first-party org so rows it creates
            # (org_id = Apprend) and its own org-scoped reads line up.
            org_id=APPREND_ORG_ID,
        )
        session.add(admin)
        await session.commit()

    # The default client acts as a "back-office" admin that sees ALL data —
    # matching the legacy unscoped behavior that pre-scoping tests rely on when
    # they call endpoints without an explicit per-user token. We do this by
    # overriding get_current_user_optional so the DEFAULT admin resolves to
    # "no user" (→ get_scope sees_all, and owner-scoped helpers apply no filter),
    # exactly like the old API-key path. This cascades everywhere the optional
    # dependency is used (get_scope AND direct per-endpoint owner checks).
    #
    # Requests carrying a real user's token (RBAC / scoping tests) still resolve
    # to that user and get properly scoped. A present-but-invalid/deactivated
    # token still raises 401 (fail closed preserved).
    _optional_bearer = HTTPBearer(auto_error=False)

    async def _default_admin_is_unscoped(
        creds: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
        db: AsyncSession = Depends(get_db),
    ):
        if creds is None:
            return None
        user = await get_current_user(creds, db)  # raises 401 if invalid/inactive
        return None if user.id == admin.id else user

    app.dependency_overrides[get_current_user_optional] = _default_admin_is_unscoped

    token = create_access_token(admin.id, admin.role)
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as ac:
        yield ac

    teardown()


@pytest_asyncio.fixture
async def sample_company(db_session: AsyncSession) -> Company:
    """Create a sample US manufacturing company."""
    company = Company(
        id=uuid.uuid4(),
        name="Acme Stamping Co.",
        domain="acmestamp.com",
        city="Cleveland",
        state_province="OH",
        country="US",
        employee_count=150,
        naics_code="3321",
        naics_description="Forging and Stamping",
        industry_segment="Metal Stamping",
        source="test",
        website="https://acmestamp.com",
        phone="216-555-0100",
    )
    db_session.add(company)
    await db_session.flush()
    return company


@pytest_asyncio.fixture
async def sample_contact(db_session: AsyncSession, sample_company: Company) -> Contact:
    """Create a sample contact at the sample company."""
    contact = Contact(
        id=uuid.uuid4(),
        company_id=sample_company.id,
        first_name="John",
        last_name="Smith",
        title="VP Manufacturing",
        email="john.smith@acmestamp.com",
        phone="216-555-0101",
        is_primary=True,
        source="test",
    )
    db_session.add(contact)
    await db_session.flush()
    return contact


@pytest_asyncio.fixture
async def canadian_company(db_session: AsyncSession) -> Company:
    """Create a sample Canadian company (for CASL testing)."""
    company = Company(
        id=uuid.uuid4(),
        name="Maple Fabrication Ltd.",
        domain="maplefab.ca",
        city="Toronto",
        state_province="ON",
        country="CA",
        employee_count=80,
        naics_code="3323",
        naics_description="Architectural and Structural Metals Manufacturing",
        source="test",
    )
    db_session.add(company)
    await db_session.flush()
    return company


@pytest_asyncio.fixture
async def canadian_contact(
    db_session: AsyncSession, canadian_company: Company
) -> Contact:
    """Create a contact at the Canadian company."""
    contact = Contact(
        id=uuid.uuid4(),
        company_id=canadian_company.id,
        first_name="Marie",
        last_name="Tremblay",
        title="Plant Manager",
        email="marie@maplefab.ca",
        source="test",
    )
    db_session.add(contact)
    await db_session.flush()
    return contact


# ─── Auth / user fixtures ────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def make_user(db_session: AsyncSession):
    """Factory to create a User with a real bcrypt-hashed password."""

    from app.org_context import APPREND_ORG_ID

    async def _make(
        email: str,
        *,
        password: str = "password123",
        role: str = UserRole.AGENT.value,
        name: str = "Test User",
        is_active: bool = True,
        must_change_password: bool = False,
        created_by=None,
        org_id=APPREND_ORG_ID,
    ) -> User:
        user = User(
            id=uuid.uuid4(),
            email=email.lower(),
            name=name,
            role=role,
            password_hash=hash_password(password),
            is_active=is_active,
            must_change_password=must_change_password,
            created_by=created_by,
            org_id=org_id,
        )
        db_session.add(user)
        # Commit (not just flush): a request that raises (e.g. a 403 from a role
        # guard) rolls back the shared in-memory connection, which would wipe a
        # merely-flushed row. Committing keeps seeded users durable across
        # multiple requests within one test.
        await db_session.commit()
        return user

    return _make


@pytest_asyncio.fixture
async def admin_user(make_user) -> User:
    return await make_user(
        "admin@landline.app", role=UserRole.ADMIN.value, name="Admin", password="adminpass123"
    )


@pytest_asyncio.fixture
async def manager_user(make_user) -> User:
    return await make_user(
        "manager@landline.app", role=UserRole.MANAGER.value, name="Manager", password="managerpass123"
    )


@pytest_asyncio.fixture
async def agent_user(make_user) -> User:
    return await make_user(
        "agent@landline.app", role=UserRole.AGENT.value, name="Agent", password="agentpass123"
    )


@pytest.fixture
def auth_headers():
    """Return a helper that builds an Authorization header for a given user."""

    def _headers(user: User) -> dict:
        token = create_access_token(user.id, user.role)
        return {"Authorization": f"Bearer {token}"}

    return _headers
