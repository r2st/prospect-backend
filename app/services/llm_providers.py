"""Multi-provider LLM fallback chain for the OpenAI-compatible backends.

Why this exists
---------------
Every AI feature in Landline/Dialtone — cold-email composition, reply
classification, conversation analysis, lead scoring, tactics — funnels through
:class:`app.services.ai_service.AIService`. When its single backend runs out of
quota, every one of those calls raises, each caller catches, and the product
degrades to static mail-merge copy and keyword-only routing **while continuing
to report success**. That has now happened twice: the Anthropic credits ran dry,
we crossed to OpenRouter, and OpenRouter's free tier then hit its account-level
daily cap — at which point the sequence engine went back to sending
un-personalised template copy and nothing in the logs said "broken".

One provider's quota wall should not be the product's outage. This module owns a
CHAIN of OpenAI-compatible providers tried in order, so an exhausted provider
costs a few hundred milliseconds of latency instead of a silent quality cliff.
The static template stays the last resort it was always meant to be — reached
only when every provider is genuinely unavailable.

The chain
---------
Default order (``LLM_PROVIDER_CHAIN``)::

    openrouter → gemini → groq → cerebras

All four speak the OpenAI ``/chat/completions`` shape, so one request builder
serves them; they differ only in base URL, model id, and a couple of optional
parameters. Gemini, Groq and Cerebras all have free tiers that need no card,
and — critically — their quotas are *independent*, which the OpenRouter
fallback-model list never was: on the free tier that quota is account-level, so
a second ``:free`` model id shares the same exhausted allowance.

A provider with no API key is skipped when the chain is built, not tried and
failed — an unconfigured provider is not an outage and should not burn a
request or trip its breaker.

Circuit breaker
---------------
A provider that fails ``LLM_BREAKER_THRESHOLD`` times in a row is skipped
entirely for ``LLM_BREAKER_COOLDOWN_SECONDS``. Without it, a provider that is
out of quota for the rest of the day is re-tried on every single call, paying
its full timeout each time — on a batch send that is the difference between
"the chain adds 200ms" and "the chain adds a minute per email". The breaker is
per-process (the API and each Celery worker keep their own view), which is the
right granularity: it is a latency optimisation, not a distributed lock, and a
worker that has not yet learned a provider is down loses one request finding out.

Monitoring
----------
Every served request logs which provider and model answered it, and
:func:`llm_status` exposes the running counts plus breaker state for the
``GET /api/v1/ai/providers`` endpoint. "Which provider wrote this email?" was
previously unanswerable, which is a large part of why two quota outages went
unnoticed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import httpx

logger = logging.getLogger(__name__)

# Providers this module knows how to talk to, and their OpenAI-compatible
# endpoints. The key is the token used in ``LLM_PROVIDER_CHAIN``.
#
# Only ``openrouter`` and ``gemini`` are marked vision-capable: the default Groq
# and Cerebras models are text-only Llama variants, and handing them an image
# block yields a confidently wrong text answer rather than an error — worse than
# skipping them, because a garbage business-card extraction looks like a real one.
KNOWN_PROVIDERS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "supports_vision": True,
        # Attribution headers OpenRouter asks integrators to send.
        "headers": {"HTTP-Referer": "https://apprend.tech", "X-Title": "DoAide Prospect"},
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "supports_vision": True,
        "headers": {},
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "supports_vision": False,
        "headers": {},
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "supports_vision": False,
        "headers": {},
    },
}


def _retryable_on_sibling_model(status: int) -> bool:
    """True when another model from the SAME provider is worth trying.

    429 is a per-model rate/quota answer, 5xx is transient, and 404 means the
    model id does not exist — the previous default ``:free`` id was retired
    without notice and 404'd every call, which looked exactly like an outage.
    A 400/401/402/403 is about the request or the credential, both of which a
    sibling model shares, so retrying one only spends money to fail again.
    """
    return status == 429 or status == 404 or status >= 500


class LLMChainExhausted(RuntimeError):
    """Every provider in the chain failed or was skipped.

    Subclasses ``RuntimeError`` so existing ``except Exception`` degradation
    paths in callers keep working unchanged — this is the signal that the static
    template is now the correct answer.
    """


@dataclass(frozen=True)
class ProviderSpec:
    """One configured provider: where to send, with what key, using which models.

    ``models`` is a tuple because OpenRouter carries a primary plus the
    configured fallback ids; the other providers carry exactly one. Models
    within a spec are tried before moving to the next provider.
    """

    name: str
    base_url: str
    api_key: str
    models: tuple[str, ...]
    headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    supports_vision: bool = False
    # Anthropic's current models answer 400 "`temperature` is deprecated for this
    # model" to any request carrying one — so every call failed over to the next
    # provider and a saved Claude key was never actually used.
    supports_temperature: bool = True

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


# ---------------------------------------------------------------------------
# Circuit breaker + request accounting
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """Skip a provider that has just failed ``threshold`` times in a row.

    State is per-provider-name and process-local. ``is_open`` self-heals: once
    the cooldown has elapsed the provider is cleared back to closed and gets a
    full fresh allowance of attempts, so recovery needs no sweeper task.
    """

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 300.0):
        self.threshold = max(1, int(threshold))
        self.cooldown = max(0.0, float(cooldown_seconds))
        self._consecutive_failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}
        self._lock = threading.Lock()

    def is_open(self, name: str) -> bool:
        """True when ``name`` should be skipped right now."""
        with self._lock:
            opened = self._opened_at.get(name)
            if opened is None:
                return False
            if (time.monotonic() - opened) >= self.cooldown:
                # Cooldown elapsed — close the breaker and let the next call
                # probe the provider for real.
                self._opened_at.pop(name, None)
                self._consecutive_failures[name] = 0
                logger.info("LLM circuit breaker for %s closed after cooldown", name)
                return False
            return True

    def record_failure(self, name: str) -> None:
        with self._lock:
            count = self._consecutive_failures.get(name, 0) + 1
            self._consecutive_failures[name] = count
            if count >= self.threshold and name not in self._opened_at:
                self._opened_at[name] = time.monotonic()
                logger.warning(
                    "LLM circuit breaker OPEN for %s after %d consecutive failures; "
                    "skipping it for %.0fs",
                    name, count, self.cooldown,
                )

    def record_success(self, name: str) -> None:
        with self._lock:
            self._consecutive_failures[name] = 0
            self._opened_at.pop(name, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Per-provider breaker state, for the monitoring endpoint."""
        now = time.monotonic()
        with self._lock:
            names = set(self._consecutive_failures) | set(self._opened_at)
            out = {}
            for name in sorted(names):
                opened = self._opened_at.get(name)
                remaining = None
                if opened is not None:
                    remaining = max(0.0, self.cooldown - (now - opened))
                out[name] = {
                    "consecutive_failures": self._consecutive_failures.get(name, 0),
                    "open": opened is not None and (remaining or 0) > 0,
                    "cooldown_remaining_seconds": round(remaining, 1) if remaining else None,
                }
            return out

    def reset(self) -> None:
        """Drop all state. Used by tests; never called in production paths."""
        with self._lock:
            self._consecutive_failures.clear()
            self._opened_at.clear()


_breaker: CircuitBreaker | None = None
_breaker_lock = threading.Lock()

# Cumulative per-provider accounting for the monitoring endpoint. Deliberately
# in-memory and per-process: this answers "is the chain working right now",
# which is the question the two silent outages needed answered.
_served: dict[str, int] = {}
_failed: dict[str, int] = {}
_last: dict[str, Any] = {"provider": None, "model": None}
_stats_lock = threading.Lock()


def get_breaker(settings: Any = None) -> CircuitBreaker:
    """The process-wide breaker, built from settings on first use."""
    global _breaker
    with _breaker_lock:
        if _breaker is None:
            if settings is None:
                from app.config import get_settings

                settings = get_settings()
            _breaker = CircuitBreaker(
                threshold=getattr(settings, "llm_breaker_threshold", 3),
                cooldown_seconds=getattr(settings, "llm_breaker_cooldown_seconds", 300),
            )
        return _breaker


def reset_breaker() -> None:
    """Drop the process-wide breaker and counters. Test-only helper."""
    global _breaker
    with _breaker_lock:
        _breaker = None
    with _stats_lock:
        _served.clear()
        _failed.clear()
        _last["provider"] = None
        _last["model"] = None


def _record_served(provider: str, model: str) -> None:
    with _stats_lock:
        _served[provider] = _served.get(provider, 0) + 1
        _last["provider"] = provider
        _last["model"] = model


def _record_failed(provider: str) -> None:
    with _stats_lock:
        _failed[provider] = _failed.get(provider, 0) + 1


def llm_status() -> dict[str, Any]:
    """Chain health for ``GET /api/v1/ai/providers``. Never includes key material."""
    from app.config import get_settings

    settings = get_settings()
    chain = build_provider_chain(settings)
    with _stats_lock:
        served = dict(_served)
        failed = dict(_failed)
        last = dict(_last)
    return {
        "configured_chain": [
            {"provider": s.name, "models": list(s.models)} for s in chain
        ],
        "unconfigured": [
            name
            for name in _chain_order(settings)
            if name not in {s.name for s in chain}
        ],
        "requests_served": served,
        "requests_failed": failed,
        "last_provider": last.get("provider"),
        "last_model": last.get("model"),
        "circuit_breakers": get_breaker(settings).snapshot(),
        # Pacing state alongside breaker state, because "outreach went out as
        # templates" and "outreach went out personalized but throttled" are
        # indistinguishable from outside the process, and the difference is
        # entirely here.
        "rate_limits": get_rate_limiter().snapshot(),
        "configured_rpm": {
            s.name: _rate_limit_module().provider_rpm(settings, s.name) for s in chain
        },
    }


def _rate_limit_module():
    """Imported lazily — ``llm_rate_limit`` reads config, which imports this."""
    from app.services import llm_rate_limit

    return llm_rate_limit


def get_rate_limiter():
    """The process-wide rate limiter. Re-exported so callers need one import."""
    return _rate_limit_module().get_rate_limiter()


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------
def _chain_order(settings: Any) -> list[str]:
    raw = getattr(settings, "llm_provider_chain", "") or ""
    order = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return [p for p in order if p in KNOWN_PROVIDERS]


def _provider_key(settings: Any, name: str) -> str:
    return (getattr(settings, f"{name}_api_key", "") or "").strip()


def _provider_models(settings: Any, name: str) -> tuple[str, ...]:
    model = (getattr(settings, f"{name}_model", "") or "").strip()
    return (model,) if model else ()


def build_provider_chain(
    settings: Any,
    *,
    openrouter_key: str | None = None,
    openrouter_models: Sequence[str] | None = None,
) -> list[ProviderSpec]:
    """Resolve ``LLM_PROVIDER_CHAIN`` into the providers we can actually call.

    Providers with no API key are dropped here rather than attempted and failed:
    an unconfigured provider is not an outage, and letting it consume a request
    slot would trip its own circuit breaker for a problem that is purely
    configuration.

    ``openrouter_key`` / ``openrouter_models`` let a caller pin a *specific*
    OpenRouter credential and model list — the Social Listening, Deal
    Progression and Client Success agents each carry their own per-org key. The
    rest of the chain still comes from process settings, so a per-org agent that
    exhausts its own OpenRouter allowance still crosses to Gemini/Groq/Cerebras
    rather than dropping to a template.
    """
    specs: list[ProviderSpec] = []
    for name in _chain_order(settings):
        meta = KNOWN_PROVIDERS[name]
        if name == "openrouter":
            key = (openrouter_key if openrouter_key is not None else _provider_key(settings, "openrouter")) or ""
            key = key.strip()
            models = tuple(openrouter_models or _provider_models(settings, "openrouter"))
            base_url = (
                getattr(settings, "openrouter_base_url", "") or meta["base_url"]
            )
            extra_body: dict[str, Any] = {}
            # Only OpenRouter accepts this. A reasoning model bills its
            # chain-of-thought against the same max_tokens budget as the answer
            # and emits it FIRST, so an uncapped trace returns empty content with
            # finish_reason="length" — see app.config for the measurement.
            effort = (getattr(settings, "openrouter_reasoning_effort", "") or "").strip()
            if effort:
                extra_body["reasoning"] = {"effort": effort}
        else:
            key = _provider_key(settings, name)
            models = _provider_models(settings, name)
            base_url = getattr(settings, f"{name}_base_url", "") or meta["base_url"]
            extra_body = {}

        if not key or not models:
            continue
        specs.append(
            ProviderSpec(
                name=name,
                base_url=base_url,
                api_key=key,
                models=models,
                headers=dict(meta["headers"]),
                extra_body=extra_body,
                supports_vision=bool(meta["supports_vision"]),
            )
        )
    return specs


def has_any_provider_key(settings: Any, *, openrouter_key: str | None = None) -> bool:
    """True when at least one chain provider is usable.

    ``AIService.is_configured`` gates on this. Asking only about the OpenRouter
    key — which is what it used to do — means a deployment carrying only a
    Gemini key answers every compose with a static template without ever trying
    the provider it does have.
    """
    return bool(build_provider_chain(settings, openrouter_key=openrouter_key))


# ---------------------------------------------------------------------------
# Message translation
# ---------------------------------------------------------------------------
def to_openai_message(message: dict) -> dict:
    """Translate one Anthropic-shaped message to OpenAI chat format.

    Plain string content passes through unchanged. A list of content blocks
    (used by the vision card-scanner) has its ``image``/``text`` blocks rewritten
    to the OpenAI ``image_url``/``text`` block form.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return {"role": message.get("role", "user"), "content": content}

    blocks = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            blocks.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            source = block.get("source", {})
            media_type = source.get("media_type", "image/jpeg")
            data = source.get("data", "")
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                }
            )
    return {"role": message.get("role", "user"), "content": blocks}


def has_image(messages: Iterable[dict]) -> bool:
    """True when any message carries an image block (Anthropic or OpenAI shape)."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                return True
    return False


# ---------------------------------------------------------------------------
# The chain call
# ---------------------------------------------------------------------------
async def chat_completion(
    messages: list[dict],
    *,
    providers: Sequence[ProviderSpec],
    max_tokens: int,
    temperature: float | None = None,
    system: str | None = None,
    timeout: float = 60.0,
    breaker: CircuitBreaker | None = None,
) -> tuple[str, str, str]:
    """Try each provider in order; return ``(content, provider_name, model)``.

    ``messages`` use the Anthropic/OpenAI-common shape; ``system`` is carried as
    a leading ``role: "system"`` message, which is what OpenAI-compatible
    endpoints want.

    Failure handling distinguishes two moves, because they cost different things:

    * **Next model, same provider** — 429 (quota), 5xx (transient), 404 (a
      ``:free`` model id retired without notice, which has happened here), and
      empty completions. These are properties of one *model*, so a sibling model
      is a real chance.
    * **Next provider** — everything else: 400 (our bug), 401 (bad or rotated
      key), 402/403 (out of credits, blocked). A sibling model shares the same
      credential and the same request, so retrying it just spends money to fail
      again — but a *different provider* shares neither.

    A provider is *skipped* — not failed — when its breaker is open, or when
    ``services.llm_rate_limit`` says it is paced out or parked by a
    ``Retry-After``. The distinction matters: a skipped provider records no
    failure and does not move the breaker, because being early is not being
    broken, and three paced requests must not blackhole a healthy provider for
    five minutes. If *every* provider is paced out at once, the chain waits out
    the soonest rather than returning — degrading to a static template because
    we were two seconds early is the failure this module exists to prevent.

    An empty completion is treated as a failure, not an answer. Returning ""
    here is what made the last outage invisible: every caller's JSON parse
    failed and degraded to a template, so a broken model looked exactly like a
    healthy one writing un-personalised copy.

    Raises :class:`LLMChainExhausted` when nothing answered, so callers run
    their existing degradation path (static template, neutral score).
    """
    if not providers:
        raise LLMChainExhausted("no LLM provider is configured")

    breaker = breaker or get_breaker()
    openai_messages = [to_openai_message(m) for m in messages]
    if system:
        openai_messages.insert(0, {"role": "system", "content": system})

    from app.config import get_settings
    from app.services import llm_rate_limit

    settings = get_settings()
    limiter = llm_rate_limit.get_rate_limiter()
    pass_budget = float(getattr(settings, "llm_rate_limit_max_wait_seconds", 0.0) or 0.0)
    max_stall = float(getattr(settings, "llm_rate_limit_max_stall_seconds", 0.0) or 0.0)

    needs_vision = has_image(messages)
    last_error: Exception | None = None
    attempted = False
    # ``{provider: seconds until it frees}`` for providers skipped *only* because
    # they were paced or parked. Not failures — they get a second pass if the
    # whole chain declines, rather than letting a two-second wait become a
    # static-template send.
    deferred: dict[str, float] = {}

    async def _run_pass(
        http: httpx.AsyncClient,
        specs: Sequence[ProviderSpec],
        wait_budget: float,
    ) -> tuple[str, str, str] | None:
        nonlocal last_error, attempted

        for spec in specs:
            if needs_vision and not spec.supports_vision:
                logger.debug(
                    "Skipping %s for an image request — model is text-only", spec.name
                )
                continue
            if breaker.is_open(spec.name):
                logger.info("Skipping %s — circuit breaker open", spec.name)
                continue

            # Once per provider, before its model list — not once per request.
            #
            # A rate limit answers "which provider should serve this call", and
            # that question is settled the moment we commit to one. Re-asking it
            # between sibling models would break the case the sibling list exists
            # for: OpenRouter's free model 429s and the *paid* fallback model,
            # which has an entirely separate allowance, is exactly what should be
            # tried next. Parking the provider on that first 429 and then
            # refusing to try the paid model would turn a working escape hatch
            # into a dead one. The cost is that a sibling retry isn't counted
            # against the window; sibling retries only happen on 429/404/5xx, so
            # that under-count is small and self-limiting.
            rpm = llm_rate_limit.provider_rpm(settings, spec.name)
            granted, wait = await limiter.acquire(spec.name, rpm, max_wait=wait_budget)
            if not granted:
                # Paced or parked, not broken. No failure recorded and the
                # breaker is untouched — being early is not being down.
                deferred[spec.name] = wait
                logger.info(
                    "Deferring %s — rate limited, %.1fs until a slot; "
                    "moving to the next provider",
                    spec.name, wait,
                )
                continue
            deferred.pop(spec.name, None)

            for model in spec.models:
                attempted = True
                payload: dict[str, Any] = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": openai_messages,
                }
                payload.update(spec.extra_body)
                if temperature is not None and spec.supports_temperature:
                    payload["temperature"] = temperature

                headers = {
                    "Authorization": f"Bearer {spec.api_key}",
                    "Content-Type": "application/json",
                    **spec.headers,
                }

                try:
                    resp = await http.post(spec.endpoint, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    last_error = exc
                    _record_failed(spec.name)
                    breaker.record_failure(spec.name)
                    if status == 429:
                        # Believe the provider's own statement about when it will
                        # take us back, so the rest of a batch skips it instantly
                        # instead of each email re-earning the same 429. This
                        # call keeps walking the sibling models — the slot was
                        # claimed before the loop, and the paid fallback model
                        # the free one 429'd past has its own allowance.
                        # ``getattr``: httpx allows a status error with no
                        # response object, and a missing header must cost us the
                        # default park, not the send.
                        rl_headers = getattr(exc.response, "headers", None) or {}
                        park = limiter.note_rate_limited(
                            spec.name, rl_headers.get("Retry-After")
                        )
                        deferred[spec.name] = park
                    retry_sibling = _retryable_on_sibling_model(status)
                    logger.warning(
                        "LLM provider %s model %s returned HTTP %s — %s",
                        spec.name, model, status,
                        "trying the next model" if retry_sibling else "moving to the next provider",
                    )
                    if retry_sibling:
                        continue
                    break
                except Exception as exc:  # transport error, timeout, bad JSON
                    last_error = exc
                    _record_failed(spec.name)
                    breaker.record_failure(spec.name)
                    logger.warning(
                        "LLM provider %s model %s failed (%s: %s) — trying next",
                        spec.name, model, type(exc).__name__, exc,
                    )
                    continue

                choice = (data.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                if not content.strip():
                    usage = data.get("usage") or {}
                    reasoning_tokens = (
                        usage.get("completion_tokens_details") or {}
                    ).get("reasoning_tokens")
                    last_error = LLMChainExhausted(
                        f"{spec.name}/{model} returned empty content "
                        f"(finish_reason={choice.get('finish_reason')}, "
                        f"reasoning_tokens={reasoning_tokens})"
                    )
                    _record_failed(spec.name)
                    breaker.record_failure(spec.name)
                    logger.warning(
                        "LLM provider %s model %s returned EMPTY content — "
                        "finish_reason=%s reasoning_tokens=%s of max_tokens=%s. A "
                        "reasoning trace that consumes the whole budget leaves no "
                        "room for the answer; lower OPENROUTER_REASONING_EFFORT or "
                        "raise OPENROUTER_MIN_MAX_TOKENS.",
                        spec.name, model, choice.get("finish_reason"),
                        reasoning_tokens, max_tokens,
                    )
                    continue

                breaker.record_success(spec.name)
                limiter.note_success(spec.name)
                _record_served(spec.name, model)
                logger.info("LLM request served by %s (model=%s)", spec.name, model)
                return content, spec.name, model

        return None

    async with httpx.AsyncClient(timeout=timeout) as http:
        served = await _run_pass(http, providers, pass_budget)
        if served is not None:
            return served

        # Nothing was even attempted: every provider that could have answered was
        # paced or parked before we sent a byte. Declining here would degrade the
        # caller to static mail-merge copy because we were a few seconds early —
        # the exact silent quality cliff this chain exists to prevent. So wait
        # out the one that frees soonest and try it alone.
        #
        # Gated on ``not attempted`` deliberately. If we did reach a provider and
        # it answered 429, the chain has already spent its round trips and been
        # told no by the providers themselves; stalling on top of that adds
        # latency to a request that has already failed on the merits.
        if deferred and not attempted and max_stall > 0:
            name, wait = llm_rate_limit.shortest_wait(deferred)
            if name is not None and 0 < wait <= max_stall:
                logger.info(
                    "Every LLM provider is rate limited — waiting %.1fs for %s "
                    "rather than degrading to a static template",
                    wait, name,
                )
                await llm_rate_limit.wait(wait)
                retry_specs = [s for s in providers if s.name == name]
                served = await _run_pass(http, retry_specs, max_stall)
                if served is not None:
                    return served

    if deferred and not attempted:
        raise LLMChainExhausted(
            "every LLM provider is rate limited; soonest available in "
            f"{llm_rate_limit.shortest_wait(deferred)[1]:.0f}s"
        )
    if not attempted:
        raise LLMChainExhausted(
            "every LLM provider was skipped (circuit breakers open, or no "
            "vision-capable provider configured for an image request)"
        )
    raise LLMChainExhausted(f"all LLM providers failed; last error: {last_error}")


# ---------------------------------------------------------------------------
# Drop-in adapter for the direct-Anthropic services
# ---------------------------------------------------------------------------
# Nine services (sales coach, orchestrator, meeting/relationship/competitive
# intelligence, sentiment, script generator, investor outreach, news monitor)
# predate AIService and call ``anthropic.AsyncAnthropic`` themselves. They are
# therefore Anthropic-only, which means they are *entirely* down right now: the
# Anthropic credits are the first thing that ran out.
#
# Rewriting nine call sites onto AIService's prompt-shaped methods would be a
# large, risky diff for a availability fix. Instead this presents the one piece
# of the Anthropic SDK those call sites actually use —
# ``client.messages.create(...) -> message.content[0].text`` — and serves it
# from the chain. Each service then changes by one line, at the point where it
# constructs its client, and every call site, response parse, ``asyncio.wait_for``
# and test around it keeps working untouched.
class _TextBlock:
    """One ``content`` block, shaped like the Anthropic SDK's text block."""

    __slots__ = ("type", "text")

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class ChainMessage:
    """An Anthropic-shaped response carrying which provider actually answered.

    ``content[0].text`` is the only field the adapted call sites read; the
    ``provider``/``model`` attributes are additive, for logging and for tests
    that want to assert the hop rather than infer it.
    """

    __slots__ = ("content", "provider", "model", "stop_reason")

    def __init__(self, text: str, provider: str, model: str):
        self.content = [_TextBlock(text)]
        self.provider = provider
        self.model = model
        self.stop_reason = "end_turn"


class _ChainMessagesAPI:
    """The ``.messages`` namespace of the adapter."""

    def __init__(self, anthropic_api_key: str = ""):
        self._anthropic_api_key = (anthropic_api_key or "").strip()

    async def create(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        system: str | None = None,
        temperature: float | None = None,
        **_ignored: Any,
    ) -> ChainMessage:
        """Run the chain, then Anthropic, and answer in the Anthropic shape.

        ``model`` is the caller's *Claude* model id. It cannot be forwarded to
        Gemini or Groq — those have their own ids from settings — so it is used
        only for the Anthropic hop, where it is the correct and intended value.
        Passing it through would 404 every non-Anthropic provider.
        """
        from app.config import get_settings

        settings = get_settings()
        providers = build_provider_chain(settings)
        last_exc: Exception | None = None
        if providers:
            try:
                text, served_by, served_model = await chat_completion(
                    messages,
                    providers=providers,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                )
                return ChainMessage(text, served_by, served_model)
            except LLMChainExhausted as exc:
                last_exc = exc

        key = self._anthropic_api_key or (settings.anthropic_api_key or "").strip()
        if not key:
            raise last_exc or LLMChainExhausted("no LLM provider is configured")

        if last_exc is not None:
            logger.warning(
                "LLM provider chain exhausted (%s) — falling back to Anthropic", last_exc
            )
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=key)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        message = await client.messages.create(**kwargs)
        text = message.content[0].text if message.content else ""
        _record_served("anthropic", model)
        logger.info("LLM request served by anthropic (model=%s)", model)
        return ChainMessage(text, "anthropic", model)


class ChainClient:
    """Drop-in replacement for ``anthropic.AsyncAnthropic``, chain-backed.

    Only ``.messages.create`` is implemented, because that is the entire surface
    the adapted services use. Anything else should move to
    :class:`~app.services.ai_service.AIService` rather than grow this shim.
    """

    def __init__(self, api_key: str = ""):
        self.messages = _ChainMessagesAPI(api_key)


def chain_client(api_key: str = "") -> ChainClient:
    """Build a chain-backed stand-in for an Anthropic client."""
    return ChainClient(api_key)


def any_llm_available(settings: Any = None) -> bool:
    """True when *some* backend can serve a request — chain provider or Anthropic.

    The adapted services each gated on ``bool(settings.anthropic_api_key)`` and
    returned a static/neutral result when it was empty. Left alone, that gate
    would keep them dark on a deployment that carries only Gemini or Groq keys —
    the feature would be fixed and still report itself unavailable.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    if (getattr(settings, "anthropic_api_key", "") or "").strip():
        return True
    return has_any_provider_key(settings)
