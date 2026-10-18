"""Celery tasks for Autopilot Outreach campaigns (see ``app.services.campaigns``).

* ``run_campaign_scheduler`` — beat, every 15 minutes, once per org via
  ``for_each_org``: reconcile, bounce guard, warm-up, daily discovery, drafts,
  and scheduling this tick's sends.
* ``run_campaign_now`` — the same tick for one campaign, dispatched by
  ``POST /campaigns`` so a new campaign starts searching immediately.
* ``send_approved_now`` — just verification and scheduling for one campaign,
  dispatched on approve and on resume so an approved email doesn't wait for
  the next tick.
* ``send_campaign_prospect`` — one send, dispatched by the scheduler with a
  countdown so consecutive sends are 120–300s apart. Bound to the prospect's
  org with ``org_scope``; the claim inside ``send_prospect`` makes a duplicate
  delivery of this task harmless.
"""

import asyncio
import logging
import uuid

from app.tasks.celery_app import celery_app
from app.tasks.org_runner import for_each_org

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _schedule(org_id: uuid.UUID, prospect_id: uuid.UUID, countdown: int) -> None:
    send_campaign_prospect.apply_async(args=[str(org_id), str(prospect_id)], countdown=countdown)


@celery_app.task(name="app.tasks.campaigns.run_campaign_scheduler")
def run_campaign_scheduler():
    """Beat task: advance every active campaign one tick."""

    async def _body(db):
        from app.services.campaigns import run_campaigns_tick

        return await run_campaigns_tick(db, schedule=_schedule)

    try:
        return _run_async(for_each_org(_body, label="run_campaign_scheduler"))
    except Exception as exc:  # noqa: BLE001
        logger.error("run_campaign_scheduler failed: %s", exc)
        return {"status": "error", "error": str(exc)}


@celery_app.task(name="app.tasks.campaigns.run_campaign_now")
def run_campaign_now(org_id: str, campaign_id: str):
    """Run one campaign's tick now — dispatched when the campaign is created."""

    async def _run():
        from app.database_celery import celery_session
        from app.org_context import org_scope
        from app.services.campaigns import run_campaign_now as run_now

        async with org_scope(uuid.UUID(org_id)):
            async with celery_session() as db:
                return await run_now(db, uuid.UUID(campaign_id), schedule=_schedule)

    return _run_async(_run())


@celery_app.task(name="app.tasks.campaigns.send_approved_now")
def send_approved_now(org_id: str, campaign_id: str):
    """Schedule one campaign's approved emails now — dispatched on approve/resume."""

    async def _run():
        from app.database_celery import celery_session
        from app.org_context import org_scope
        from app.services.campaigns import send_approved_now as send_now

        async with org_scope(uuid.UUID(org_id)):
            async with celery_session() as db:
                return await send_now(db, uuid.UUID(campaign_id), schedule=_schedule)

    return _run_async(_run())


@celery_app.task(name="app.tasks.campaigns.send_campaign_prospect")
def send_campaign_prospect(org_id: str, prospect_id: str):
    """Send one campaign prospect's approved email."""

    async def _run():
        from app.database_celery import celery_session
        from app.org_context import org_scope
        from app.services.campaigns import send_prospect

        async with org_scope(uuid.UUID(org_id)):
            async with celery_session() as db:
                outcome = await send_prospect(db, uuid.UUID(prospect_id))
                return {"status": outcome.status, "detail": outcome.detail}

    return _run_async(_run())
