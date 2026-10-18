"""A freshly started worker applies dashboard credential overrides before its first task.

Overrides from ``app_credentials`` are applied in memory, per process. The only
thing that applied them in a Celery worker was the ``refresh-credentials`` beat
entry, and an interval entry first fires one interval *after* beat starts — so
for five minutes after every deploy or restart the worker ran on bare ``.env``
values. That is exactly when a new campaign's first run is dispatched: with the
Hunter key set only from the dashboard, the finder found nothing, verification
was skipped, and drafts were written for guessed addresses Hunter never saw.

These pin the ``worker_process_init`` hook rather than the beat.
"""

from contextlib import asynccontextmanager

import pytest
from celery.signals import worker_process_init

import app.database_celery as database_celery
from app.services import credential_store
from app.tasks import credentials as credentials_task


@pytest.fixture
def fake_store(monkeypatch):
    calls = []

    @asynccontextmanager
    async def fake_session():
        yield "session"

    async def fake_hydrate(db):
        calls.append(db)
        return {"hunter_api_key": "x"}

    monkeypatch.setattr(database_celery, "celery_session", fake_session)
    monkeypatch.setattr(credential_store, "hydrate", fake_hydrate)
    return calls


def test_the_hook_is_connected_by_importing_the_task_module():
    receivers = [r() for _, r in worker_process_init.receivers]
    assert credentials_task.hydrate_on_process_start in receivers


def test_a_new_pool_process_applies_overrides_before_any_beat_tick(fake_store):
    worker_process_init.send(sender=None)
    assert fake_store == ["session"]


def test_the_beat_task_still_applies_them(fake_store):
    assert credentials_task.refresh_credentials.run() == {"applied": ["hunter_api_key"]}
    assert fake_store == ["session"]


def test_a_failing_store_does_not_stop_the_worker_from_starting(monkeypatch):
    @asynccontextmanager
    async def broken_session():
        raise RuntimeError("database unreachable")
        yield  # pragma: no cover

    monkeypatch.setattr(database_celery, "celery_session", broken_session)
    # Must not raise: a worker that can't read overrides keeps the .env values.
    worker_process_init.send(sender=None)
    assert credentials_task.refresh_credentials.run()["error"] == "database unreachable"
