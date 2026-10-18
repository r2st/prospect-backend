"""Celery task: re-apply credential overrides on every worker.

:mod:`app.services.credential_store` resolves overrides once per process and
writes them onto the ``Settings`` singleton, because the fifteen call sites that
read ``settings.google_client_id`` are spread across request handlers, beat
tasks and synchronous OAuth helpers — several with no database session in hand.
That makes the resolution cheap and the *propagation* explicit: the worker that
served the write has the new value immediately, and every other worker has the
old one until it refreshes.

This is that refresh. Without it, rotating a credential from the dashboard would
take effect on one API worker and nowhere else until the next deploy — which is
worse than not having the feature, because the UI would report success while the
beat worker went on using the retired key.

Five minutes: these are credentials, not feature flags. They change about once a
year, and a shorter interval buys nothing but load.

The beat alone leaves a hole at the start of every process. Overrides are
applied in memory, so a worker that was just deployed or restarted runs on
bare ``.env`` values until the first tick — and
an interval entry first fires one interval *after* beat starts. Five minutes
in which every dashboard-only key is absent, and those are the five minutes in
which a new campaign's first run is dispatched: with ``hunter_api_key`` set
only from the dashboard, the finder found nothing, verification was skipped,
and drafts were written for guessed addresses Hunter never saw. Hence
``worker_process_init``: each pool process applies the overrides before it
takes its first task.

Deliberately **not** per-org (no ``for_each_org``): the table is platform-wide,
and the value being applied is a process-global. Never raises — a refresh that
fails leaves the previously applied values in place, which is the last known
good state.
"""

import asyncio
import logging

from celery.signals import worker_process_init

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _apply_overrides() -> dict:
    """Re-read ``app_credentials`` and apply it to this process's settings."""

    async def _run():
        # celery_session, not the API's engine: each call runs on a fresh event
        # loop, and a pooled connection from the previous loop is unusable.
        from app.database_celery import celery_session
        from app.services import credential_store

        async with celery_session() as db:
            overrides = await credential_store.hydrate(db)
            return {"applied": sorted(overrides)}

    try:
        return _run_async(_run())
    except Exception as exc:
        # The previously applied values stay in force — last known good.
        logger.error("Credential refresh failed: %s", exc)
        return {"applied": [], "error": str(exc)}


@celery_app.task(name="app.tasks.credentials.refresh_credentials")
def refresh_credentials():
    """Re-read ``app_credentials`` and apply it to this worker's settings."""
    return _apply_overrides()


@worker_process_init.connect
def hydrate_on_process_start(**_kwargs):
    """Apply overrides before this pool process runs its first task."""
    result = _apply_overrides()
    logger.info("Credential overrides applied at worker start: %s", result.get("applied"))
