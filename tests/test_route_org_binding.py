"""Every route binds the caller's org, or is a reviewed public one.

The tenancy machinery is a *read filter* that needs something to filter by, and
exactly one thing supplies it: :func:`app.dependencies.get_current_user` calls
``set_current_org_id(user.org_id)``. No org bound is not "unscoped pending
review" — it is the documented no-op, the same branch a platform admin takes,
and it is reached by any route that never runs that dependency.

So a router mounted with ``dependencies=_auth`` is isolated and one mounted
without it is not, and **the difference is one keyword argument on a line in
``app/main.py``**, four screens away from the endpoint it decides the fate of.
``ApiKeyMiddleware`` does not close the gap: it fails closed on credentials, so
the request is authenticated — it just isn't attributed to anyone. Every
reviewer's instinct that "you need a token to reach this" is correct and beside
the point.

``health.router`` sat that way. Its prefix is ``/companies`` and its module is
named for a liveness probe, but it serves client *health scoring* — and with no
org bound, ``GET /companies/{id}/health`` computed and returned any tenant's
account health from an id, ``/health/history`` returned their snapshot series,
and ``POST /health/snapshot`` **wrote** a row: no org context also means
``org_id_for_insert()`` falls back to the first-party org, so the write landed
under Apprend's org pointing at a stranger's company. Ids are not secrets — they
come back in exports, webhooks and error bodies.

This test is the thing that has an opinion about all of them at once. The
allowlist is the whole design: a genuinely public route (login, an OAuth
callback, the tracking pixel) has to be *named*, with a reason, by whoever adds
it. Adding a router to ``main.py`` and forgetting ``dependencies=_auth`` fails
here instead of shipping.
"""

from fastapi.routing import APIRoute

from app.main import app


# Dependency names that bind (or deliberately decline to bind) an org for the
# request. ``get_current_user`` is the one that calls ``set_current_org_id``;
# the others reach it transitively.
BINDING_DEPENDENCIES = {
    "get_current_user",
    "get_current_user_optional",
    "get_org_scope",
    "get_scope",
}


# Routes that legitimately run with no org context, and why. The reason is part
# of the entry: an entry without one is an unreviewed exception, so don't add
# one. "It has its own check" is not a reason to be here — a route that runs
# ``get_current_user`` itself already passes without an entry.
PUBLIC_ROUTES: dict[str, str] = {
    "GET /": "root banner — a static string, no database read",
    "GET /health": "liveness probe for the load balancer, before any tenant exists",
    "GET /api/v1/health": (
        "the same liveness probe under the API prefix, which is what the "
        "container healthcheck polls"
    ),
    "POST /api/v1/auth/login": (
        "the caller has no org yet — establishing which one is the point of "
        "the request"
    ),
    "POST /api/v1/auth/refresh": (
        "exchanges a refresh token for an access token; reads the User row by "
        "the token's subject and nothing org-scoped"
    ),
    "POST /api/v1/auth/register": "creates the organization, so there is none to bind",
    "GET /api/v1/gmail/oauth/callback": (
        "Google redirects the browser here with a code and no Authorization "
        "header; the org is recovered from the signed OAuth state parameter"
    ),
    "GET /api/v1/postmaster/oauth/callback": "same OAuth redirect shape as the Gmail one",
    "GET /api/v1/track/open/{token}": (
        "the tracking pixel, loaded by the recipient's mail client. The org "
        "comes from the signed token and is bound with org_scope()"
    ),
    "GET /api/v1/track/click/{token}": "link tracking, same signed-token shape as the pixel",
    "GET /api/v1/track/unsubscribe": (
        "the RFC 8058 / footer opt-out link, which by definition is reached by "
        "someone with no account. Refusing it is a compliance failure"
    ),
    "POST /api/v1/track/unsubscribe": "the one-click POST half of the same opt-out",
    "POST /api/v1/track/resubscribe": (
        "undoes the footer opt-out from the same page; reached by the same "
        "account-less recipient and authorised by the same signed token"
    ),
    "POST /api/v1/outreach/events": (
        "the send-ledger webhook, authenticated by HMAC signature or JWT in "
        "require_signed_webhook_or_user rather than by a session"
    ),
}


def _dependency_names(route: APIRoute) -> set[str]:
    """Every dependency callable reachable from *route*, transitively."""
    names: set[str] = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if dep.call is not None:
            names.add(getattr(dep.call, "__name__", type(dep.call).__name__))
        stack.extend(dep.dependencies)
    return names


def _unbound_routes() -> list[str]:
    problems = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            key = f"{method} {route.path}"
            if key in PUBLIC_ROUTES:
                continue
            if _dependency_names(route) & BINDING_DEPENDENCIES:
                continue
            problems.append(key)
    return sorted(problems)


def test_every_route_binds_an_org_or_is_a_reviewed_public_one():
    assert not _unbound_routes(), (
        "These routes run with no org context, so the read filter is a no-op "
        "and every tenant's rows are reachable by id. Almost always the fix is "
        "`dependencies=_auth` on the include_router line in app/main.py. If the "
        "route is genuinely public, add it to PUBLIC_ROUTES in this file with "
        "the reason:\n  " + "\n  ".join(_unbound_routes())
    )


def test_public_routes_all_still_exist():
    """An allowlist entry for a route that is gone is a stale exemption.

    It is also a live hazard: paths get reused, and an entry naming a path some
    future router happens to mount would exempt it silently.
    """
    live = {
        f"{method} {route.path}"
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods - {"HEAD", "OPTIONS"}
    }
    stale = sorted(set(PUBLIC_ROUTES) - live)
    assert not stale, f"PUBLIC_ROUTES entries matching no route: {stale}"


def test_public_route_entries_carry_a_reason():
    thin = sorted(k for k, v in PUBLIC_ROUTES.items() if len(v.strip()) < 40)
    assert not thin, f"PUBLIC_ROUTES entries without a real reason: {thin}"


def test_the_check_catches_a_router_mounted_without_auth():
    """A checker that matches nothing passes forever and reads as coverage.

    Mount a router the way ``health`` was mounted — no ``dependencies=_auth``,
    nothing but ``get_db`` — and assert the scan names it.
    """
    from fastapi import APIRouter, Depends, FastAPI

    from app.database import get_db

    probe = APIRouter(prefix="/probe")

    @probe.get("/{thing_id}")
    async def _read(thing_id: str, db=Depends(get_db)):  # pragma: no cover
        return {}

    scratch = FastAPI()
    scratch.include_router(probe)

    unbound = [
        f"{method} {route.path}"
        for route in scratch.routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"})
        if not _dependency_names(route) & BINDING_DEPENDENCIES
    ]
    assert unbound == ["GET /probe/{thing_id}"]
