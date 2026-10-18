"""Landline Backend — B2B Lead Generation API for Apprend Technologies.

Main FastAPI application entry point. Wires up routers, middleware, and lifecycle events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.auth import ApiKeyMiddleware
from app.dependencies import get_current_user
from app.rate_limit import RateLimitStateMiddleware, limiter
from app.security import get_configured_secret
from app.routers import auth, admin, companies, contacts, leads, outreach, dashboard, import_export, email_templates, email_signatures, follow_ups, gmail, smtp, discovery, tracking, ai, groups, news, tactics, pipeline, activity, sequences, suggestions, deals, tasks, interactions, health, clients, reports, emails, knowledge, calls, investors, sentiment, email_testing, intent, coaching, competitive, routing, tenants, integrations, workflows, prospecting, deliverability, orchestration, relationships, meetings, system, email_test, dialtone, replies, smart_reply, domain_health, danger_signals, ab_tests, send_timing, verification, ai_drafts, platform, org, gdpr, deal_agent, referral_agent, social_listening_agent, meeting_agent, client_success_agent, channels, postmaster, analytics, api_keys, credentials, settings as settings_router, discovery_documents, campaigns, feedback
from app.routers.public_api import public_app

settings = get_settings()


def require_secret_key() -> None:
    """Refuse to boot without a configured signing secret.

    Signing JWTs with a hardcoded default would let anyone mint valid tokens,
    so a missing secret is a fatal misconfiguration rather than something to
    paper over with a fallback.
    """
    if not get_configured_secret():
        raise RuntimeError(
            "Refusing to start: no signing secret configured. Set SECRET_KEY "
            "(or JWT_SECRET_KEY / TOKEN_ENCRYPTION_KEY) in the environment."
        )


def is_valid_origin(value: str) -> bool:
    """Whether ``value`` is a bare ``scheme://host[:port]`` — what CORS compares.

    The middleware matches this list by exact string against the browser's
    ``Origin`` header, which is always exactly that shape. ``apprend.tech``,
    ``https://app.apprend.tech/`` and ``*.apprend.tech`` are all things people
    write in this variable, and all three match nothing at all.
    """
    from urllib.parse import urlparse

    parsed = urlparse(value)
    return (
        parsed.scheme in ("http", "https")
        and bool(parsed.netloc)
        and "*" not in parsed.netloc
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def resolve_cors_origins(origins: list[str], *, is_prod: bool) -> list[str]:
    """The CORS list actually installed, with the two invisible faults handled.

    Neither of these can be seen from the server side once it is running, which
    is why they are resolved at boot rather than left to a review.

    **The wildcard is a security fault and is fatal in production.**
    ``allow_credentials=True`` is set below because the frontend authenticates
    with a header. Combined with ``"*"``, Starlette stops replying
    ``Access-Control-Allow-Origin: *`` and starts echoing back whatever
    ``Origin`` the request carried — every origin, individually approved. It
    reads like the loosest available setting and is looser than that. A dev box
    legitimately wants it, so outside production it is a warning.

    **A malformed entry is a config fault and is dropped, loudly.** It is
    tempting to make this fatal too — it is the one that hides, since the
    frontend fails every request and only the browser console says why. But the
    entry grants nothing, so refusing to boot over a trailing slash would take
    the whole API down on deploy to fix an origin that was already inert. Drop
    it and log at ERROR: same information, and the endpoints that don't need a
    browser keep serving.
    """
    if "*" in origins:
        message = (
            "CORS_ORIGINS contains '*' while credentials are allowed. Starlette "
            "answers that by echoing the caller's Origin, so every site becomes "
            "an approved one. List the frontend origins explicitly."
        )
        if is_prod:
            raise RuntimeError(f"Refusing to start: {message}")
        logger.warning(message)

    resolved = []
    for origin in origins:
        if origin != "*" and not is_valid_origin(origin):
            logger.error(
                "Ignoring CORS origin %r — not a bare scheme://host[:port], so it "
                "could never match a browser Origin header. Any frontend served "
                "from it will be blocked, with the error visible only in the "
                "browser console.",
                origin,
            )
            continue
        resolved.append(origin)
    return resolved

# Configure logging — every line is tagged with the current org (multi-tenant
# observability), so a stray cross-tenant read/write is traceable to a request.
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | org=%(org_id)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
from app.org_context import OrgContextLogFilter  # noqa: E402

for _handler in logging.getLogger().handlers:
    _handler.addFilter(OrgContextLogFilter())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Fail fast on a missing signing secret rather than booting insecurely.
    # (The CORS list is resolved at import, below — the middleware stack has to
    # be built before the first request, which is before this runs.)
    require_secret_key()
    logger.info("Starting Landline API (%s)", settings.app_env)

    # Apply any credential overrides an operator set from the dashboard, before
    # the first request can resolve a Google client id or a provider key. A
    # failure here must not stop the app booting: every managed credential has
    # its environment value as the fallback, and an API that will not start is a
    # worse outcome than one running on the deployed credentials.
    try:
        from app.database import async_session_factory
        from app.services import credential_store

        async with async_session_factory() as session:
            await credential_store.hydrate(session)
    except Exception as exc:  # pragma: no cover - exercised via the beat task
        logger.warning(
            "Could not load credential overrides at startup (%s); "
            "using environment values", exc,
        )

    yield
    logger.info("Shutting down Landline API")
    # Close the database engine
    from app.database import engine
    await engine.dispose()


# Hide interactive API docs in production — the schema is an information-
# disclosure surface (enumerates every endpoint, model, and auth scheme).
_is_prod = settings.app_env.lower() == "production"

app = FastAPI(
    title="DoAide Prospect — Lead Generation API",
    description=(
        "B2B lead generation backend for Apprend Technologies. "
        "Discovers, scores, and engages manufacturing companies in the US and Canada."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
    openapi_url=None if _is_prod else "/openapi.json",
)

# Rate limiting (slowapi) — endpoints opt in via @limiter.limit(...).
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Required for the limiter to survive its own storage being unreachable: slowapi
# swallows the error and then reads state it skipped setting, which surfaces as
# a 500 on exactly the successful requests. See RateLimitStateMiddleware.
app.add_middleware(RateLimitStateMiddleware)

# CORS. ``allow_credentials=True`` is what makes the list worth checking — see
# resolve_cors_origins for what a "*" turns into once credentials are allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=resolve_cors_origins(settings.cors_origins, is_prod=_is_prod),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key / JWT authentication — ALWAYS installed. The middleware fails closed:
# every non-public request must present a valid API key or Bearer token.
app.add_middleware(ApiKeyMiddleware)

# Dependency that enforces an authenticated, active user on a whole router.
_auth = [Depends(get_current_user)]

# Mount routers under the API prefix
app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(admin.router, prefix=settings.api_prefix)
app.include_router(companies.router, prefix=settings.api_prefix, dependencies=_auth)
# contacts: was mounted WITHOUT auth, so its endpoints (none of which take a
# user/scope dependency) ran with no org context bound → the global read filter
# was a no-op and they saw/wrote every tenant's contacts. Require auth so the
# caller's org is bound on every request.
app.include_router(contacts.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(leads.router, prefix=settings.api_prefix, dependencies=_auth)
# outreach.public_router carries the ESP event webhook. "Public" is now only
# about *which* credential it takes, not whether it takes one: it is mounted
# without the router-level JWT dependency so a signed ESP callback can reach it,
# and its own route dependency requires either that signature or a JWT.
# The main outreach.router (sequences, enrollments, stats) is org-scoped data and
# none of its endpoints take a user dependency, so it MUST require auth to bind
# org context — otherwise it ran unscoped across all tenants.
app.include_router(outreach.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(outreach.public_router, prefix=settings.api_prefix)
app.include_router(dashboard.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(import_export.router, prefix=settings.api_prefix, dependencies=_auth)
# email_templates/deals/emails/follow_ups/calls were all mounted WITHOUT auth
# and have endpoints that take no user/scope dependency, so they ran unscoped
# across all tenants. Require auth so org context is bound (see contacts above).
app.include_router(email_templates.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(email_signatures.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(feedback.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(follow_ups.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(gmail.router, prefix=settings.api_prefix)
app.include_router(smtp.router, prefix=settings.api_prefix)
app.include_router(discovery_documents.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(campaigns.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(discovery.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(tracking.router, prefix=settings.api_prefix)
app.include_router(ai.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(groups.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(news.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(tactics.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(pipeline.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(activity.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(sequences.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(suggestions.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(deals.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(tasks.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(interactions.router, prefix=settings.api_prefix, dependencies=_auth)
# NOT a liveness probe — this is client *health scoring* on
# ``/companies/{id}/health``, and it needs `_auth` like every other tenant
# surface. See tests/test_route_org_binding.py: `_auth` is what binds the
# caller's org, so a router mounted without it runs the whole request with no
# org context, and the read filter is a no-op rather than a filter.
app.include_router(health.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(clients.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(reports.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(emails.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(knowledge.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(calls.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(investors.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(sentiment.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(email_testing.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(intent.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(coaching.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(competitive.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(routing.router, prefix=settings.api_prefix, dependencies=_auth)
# Phase 4 — Platform & Ecosystem
app.include_router(tenants.router, prefix=settings.api_prefix, dependencies=_auth)
# SaaS Phase 2 — Platform Admin (cross-org management; platform_admin only).
app.include_router(platform.router, prefix=settings.api_prefix, dependencies=_auth)
# Platform credential management (view/rotate API keys from the dashboard).
app.include_router(credentials.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(integrations.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(workflows.router, prefix=settings.api_prefix, dependencies=_auth)
# Phase 3 — Advanced AI & Autonomous Operations
app.include_router(prospecting.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(deliverability.router, prefix=settings.api_prefix, dependencies=_auth)
# Phase 2 — Orchestration, relationship intelligence, meeting briefings
app.include_router(orchestration.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(relationships.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(meetings.router, prefix=settings.api_prefix, dependencies=_auth)
# System control — global sending kill switch + warm-up reset.
app.include_router(system.router, prefix=settings.api_prefix, dependencies=_auth)
# Test-send + deliverability verification.
app.include_router(email_test.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(dialtone.router, prefix=settings.api_prefix, dependencies=_auth)
# Unified per-sequence / per-step reply-rate funnel (single source of truth).
app.include_router(analytics.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(danger_signals.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(replies.router, prefix=settings.api_prefix, dependencies=_auth)
# Smart-reply: manual sweep trigger + operator approve-and-send for inbox drafts.
app.include_router(smart_reply.router, prefix=settings.api_prefix, dependencies=_auth)
# GDPR: data-subject export (SAR) + right-to-erasure (admin-only, self-gated).
app.include_router(gdpr.router, prefix=settings.api_prefix, dependencies=_auth)
# Feature 5 — Domain Health Dashboard (SPF/DKIM/DMARC + DNSBL, scored & persisted).
app.include_router(domain_health.router, prefix=settings.api_prefix, dependencies=_auth)
# Feature 3 — A/B Testing (variant persistence, assignment, results dashboard).
app.include_router(ab_tests.router, prefix=settings.api_prefix, dependencies=_auth)
# Human review queue for AI-composed first touches.
app.include_router(ai_drafts.router, prefix=settings.api_prefix, dependencies=_auth)
# Feature 6 — Smart Send-Time Optimization (per-recipient timezone/window + A/B).
app.include_router(send_timing.router, prefix=settings.api_prefix, dependencies=_auth)
# P3 — Hunter.io email verification (on-demand verify, bulk verify, stats).
app.include_router(verification.router, prefix=settings.api_prefix, dependencies=_auth)
# P4 — Google Places SMB discovery. A second router out of routers/discovery.py,
# on the /dialtone/discovery prefix; the /discovery one above is Apollo's.
app.include_router(discovery.places_router, prefix=settings.api_prefix, dependencies=_auth)
# Phase 3 — organization self-service management (settings, team, usage). Auth
# is required; per-route OrgScope enforces org_admin for writes.
app.include_router(org.router, prefix=settings.api_prefix, dependencies=_auth)
# Deal Progression Agent — monitors the pipeline, nudges stalled deals. Auth
# required; per-route OrgScope enforces org_admin for config writes.
app.include_router(deal_agent.router, prefix=settings.api_prefix, dependencies=_auth)
# Social Listening Agent — public buying-intent signals → warm leads. Auth
# required; per-route OrgScope enforces org_admin for config/monitoring writes.
app.include_router(social_listening_agent.router, prefix=settings.api_prefix, dependencies=_auth)
# Client Success Agent — autonomous client-health monitoring + proactive
# outreach. Auth required; config writes enforce org_admin via OrgScope.
app.include_router(client_success_agent.router, prefix=settings.api_prefix, dependencies=_auth)

# Referral & Warm-Intro agent — config, referral pipeline, and action queue.
# Auth required; per-route OrgScope enforces org_admin for config writes + action
# execution.
app.include_router(referral_agent.router, prefix=settings.api_prefix, dependencies=_auth)
# Meeting Booking Agent — autonomous scheduling on a positive reply. Auth
# required; per-route OrgScope enforces org_admin for config writes.
app.include_router(meeting_agent.router, prefix=settings.api_prefix, dependencies=_auth)
# P2 audit — multi-channel step execution (email real; LinkedIn/phone stubs).
app.include_router(channels.router, prefix=settings.api_prefix, dependencies=_auth)
# P2 audit — Google Postmaster Tools (Gmail reputation/spam/auth visibility).
# The public OAuth callback stays unauthenticated (like Gmail's) on its own
# router; the rest carries org-scoped data and requires auth.
app.include_router(postmaster.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(postmaster.public_router, prefix=settings.api_prefix)
# Public-API key management (mint/list/revoke). Auth required; each route
# enforces org_admin via OrgScope. The keys minted here authenticate the public
# API mounted below.
app.include_router(api_keys.router, prefix=settings.api_prefix, dependencies=_auth)
app.include_router(settings_router.router, prefix=settings.api_prefix, dependencies=_auth)

# Public REST API v1 — a mounted sub-application with its own OpenAPI docs at
# /api/v1/public/docs and X-API-Key authentication enforced per route. It shares
# the parent's dependency_overrides so test/DB overrides propagate into it, and
# its prefix is whitelisted in the auth middleware (which would otherwise demand
# a JWT) because the sub-app authenticates callers itself.
public_app.dependency_overrides = app.dependency_overrides
app.mount(f"{settings.api_prefix}/public", public_app)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "env": settings.app_env,
        "version": "1.0.0",
    }


@app.get(f"{settings.api_prefix}/health")
async def api_health_check():
    """Health check under the API prefix."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "env": settings.app_env,
        "version": "1.0.0",
    }


@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "name": "DoAide Prospect Lead Generation API",
        "by": "Apprend Technologies",
        "docs": "/docs",
        "health": "/health",
    }
