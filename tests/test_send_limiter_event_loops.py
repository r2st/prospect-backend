"""The send limiter's Redis client must not outlive the event loop it was made on.

Each Celery task runs in its own fresh loop; a client cached across them handed
the next task a pool bound to a closed loop ("Event loop is closed"), and the
cross-worker daily cap quietly fell back to per-worker row counts.
"""

import asyncio
import importlib.util

import app.services.send_limiter as _patched


def _fresh_module():
    """A private copy of the module: conftest swaps ``_get_redis`` for a fake
    on the shared one, and this test is about the real implementation."""
    spec = importlib.util.spec_from_file_location("_send_limiter_copy", _patched.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_redis_client_is_rebuilt_for_each_event_loop():
    sl = _fresh_module()

    async def grab():
        first = await sl._get_redis()
        again = await sl._get_redis()
        return first, again

    try:
        a1, a2 = asyncio.run(grab())
        b1, _ = asyncio.run(grab())
    finally:
        sl._reset_redis_for_tests()

    assert a1 is not None and a1 is a2  # reused within one loop
    assert b1 is not a1                 # a new loop gets a new client
