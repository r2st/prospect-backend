"""LLM calls billed to the org's own keys, for the Discovery features.

The org's keys live Fernet-encrypted in ``org_llm_keys`` (managed through
``/settings/llm-key``), one per provider: ``gemini`` first, then ``groq`` and
``openrouter`` as fallbacks for when Gemini is out of quota, rate-limited, or
has closed its model to the key's account. These features deliberately do
**not** fall back to the *platform* LLM chain: with no org key at all the
caller gets :class:`GeminiKeyMissing`, which the routers turn into
``gemini_key_required`` so the UI can send the user to Settings → API Keys.

Calls go through :func:`llm_providers.chat_completion` (rate limiting, retry,
sibling-model and next-provider fallback, empty-completion handling) with
per-org providers and a breaker that is private to this module, so one org's
bad key can never trip the shared platform breakers for everyone else.

Only Gemini has Google Search grounding. :func:`generate_grounded_json` tries
it first and, when it fails, answers from Groq/OpenRouter's training data —
flagged ``grounded=False`` with a ``notice`` the caller must surface, because
an ungrounded company list can be stale or partly invented.

The keys never leave this module: they are not logged, not returned, and not
included in the errors raised here. (The module keeps its Gemini-era name and
exception names because every Discovery caller imports them.)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.org_llm_key import OrgLlmKey
from app.services import crypto
from app.services.llm_providers import (
    KNOWN_PROVIDERS,
    CircuitBreaker,
    LLMChainExhausted,
    ProviderSpec,
    chat_completion,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Keyed by the per-org provider name, so orgs never share breaker state.
_breaker = CircuitBreaker(threshold=3, cooldown_seconds=120.0)


# Fallback order. Gemini stays the default; Claude is an optional second
# choice on the org's own Anthropic API key (not a Claude.ai subscription).
DISCOVERY_PROVIDERS: tuple[str, ...] = ("gemini", "anthropic", "groq", "openrouter")
# Web search APIs, stored in the same table. Not LLMs: app.services.web_search
# reads them, and Discovery uses one to ground any LLM in real pages.
SEARCH_PROVIDERS: tuple[str, ...] = ("tavily", "serper", "brave")
PROVIDER_LABELS = {
    "gemini": "Gemini", "anthropic": "Claude", "groq": "Groq", "openrouter": "OpenRouter",
    "tavily": "Tavily", "serper": "Serper", "brave": "Brave Search",
}


class GeminiKeyMissing(Exception):
    """The org has saved no Discovery key (Gemini, Groq or OpenRouter)."""


class GeminiError(Exception):
    """No provider produced a usable answer. ``str()`` is safe to show users."""


async def get_org_llm_keys(db: AsyncSession, org_id: uuid.UUID) -> dict[str, str]:
    """The org's decrypted Discovery LLM keys, ``{provider: key}`` in fallback order."""
    return await _org_keys(db, org_id, DISCOVERY_PROVIDERS)


async def get_org_search_keys(db: AsyncSession, org_id: uuid.UUID) -> dict[str, str]:
    """The org's decrypted web-search keys, ``{provider: key}`` in preference order."""
    return await _org_keys(db, org_id, SEARCH_PROVIDERS)


async def _org_keys(
    db: AsyncSession, org_id: uuid.UUID, providers: tuple[str, ...]
) -> dict[str, str]:
    """Decrypted keys for ``providers``, ``{provider: key}`` in that order.

    ``org_llm_keys`` has no unique constraint on ``(org_id, provider)``, so a
    duplicate row is possible; the most recently updated one wins, with ``id``
    breaking a tie so repeat calls pick the same row. A key that won't decrypt
    is skipped (and logged), not fatal — the next provider can still serve.
    """
    rows = (
        await db.execute(
            select(OrgLlmKey)
            .where(
                OrgLlmKey.org_id == org_id,
                OrgLlmKey.provider.in_(providers),
            )
            .order_by(OrgLlmKey.updated_at.desc(), OrgLlmKey.id)
        )
    ).scalars().all()
    found: dict[str, str] = {}
    for row in rows:
        if row.provider in found:
            continue
        try:
            key = crypto.decrypt(row.key_encrypted).strip()
        except Exception:
            logger.error("Org %s %s key could not be decrypted", org_id, row.provider)
            continue
        if key:
            found[row.provider] = key
    return {p: found[p] for p in providers if p in found}


async def get_org_gemini_key(db: AsyncSession, org_id: uuid.UUID) -> Optional[str]:
    """Decrypt the org's Gemini key, or ``None`` when it has none (or it's unreadable)."""
    return (await get_org_llm_keys(db, org_id)).get("gemini")


async def has_org_gemini_key(db: AsyncSession, org_id: uuid.UUID) -> bool:
    return (await get_org_gemini_key(db, org_id)) is not None


async def has_org_llm_key(db: AsyncSession, org_id: uuid.UUID) -> bool:
    """True when any Discovery provider is usable — what gates the features."""
    return bool(await get_org_llm_keys(db, org_id))


def _models(raw: str | None, default: str) -> tuple[str, ...]:
    models = tuple(m.strip() for m in (raw or "").split(",") if m.strip())
    return models or (default,)


def discovery_models() -> tuple[str, ...]:
    """``discovery_gemini_model`` as an ordered fallback list."""
    return _models(get_settings().discovery_gemini_model, "gemini-flash-latest")


def gemini_provider(org_id: uuid.UUID, api_key: str) -> ProviderSpec:
    settings = get_settings()
    extra: dict = {"response_format": {"type": "json_object"}}
    effort = (settings.discovery_gemini_reasoning_effort or "").strip()
    if effort:
        extra["reasoning_effort"] = effort
    return ProviderSpec(
        name=f"gemini-org-{org_id}",
        base_url=settings.gemini_base_url or KNOWN_PROVIDERS["gemini"]["base_url"],
        api_key=api_key,
        models=discovery_models(),
        extra_body=extra,
        supports_vision=True,
    )


def groq_provider(org_id: uuid.UUID, api_key: str) -> ProviderSpec:
    settings = get_settings()
    return ProviderSpec(
        name=f"groq-org-{org_id}",
        base_url=settings.groq_base_url or KNOWN_PROVIDERS["groq"]["base_url"],
        api_key=api_key,
        models=_models(settings.discovery_groq_models, "llama-3.3-70b-versatile"),
        # Groq's JSON mode needs "json" in the prompt, which every Discovery
        # prompt has. No reasoning_effort: the Llama models reject it.
        extra_body={"response_format": {"type": "json_object"}},
        supports_vision=False,
    )


def openrouter_provider(org_id: uuid.UUID, api_key: str) -> ProviderSpec:
    settings = get_settings()
    return ProviderSpec(
        name=f"openrouter-org-{org_id}",
        base_url=settings.openrouter_base_url or KNOWN_PROVIDERS["openrouter"]["base_url"],
        api_key=api_key,
        models=_models(
            settings.discovery_openrouter_models, "meta-llama/llama-3.3-70b-instruct:free"
        ),
        headers=KNOWN_PROVIDERS["openrouter"]["headers"],
        # No response_format: free OpenRouter models support it unevenly and a
        # 400 abandons the whole provider rather than trying the sibling model.
        # The prompt asks for JSON; _json_object_text tolerates prose around it.
        supports_vision=False,
    )


def anthropic_provider(org_id: uuid.UUID, api_key: str) -> ProviderSpec:
    """Claude through Anthropic's OpenAI-compatible endpoint.

    Same Bearer auth and ``/chat/completions`` shape as every other provider,
    so it shares chat_completion's retry, sibling-model fallback and breaker.
    No response_format: the compatibility layer doesn't honour JSON mode, and
    every Discovery prompt asks for JSON; _json_object_text tolerates prose.
    No temperature either: the current models refuse the parameter outright.
    """
    settings = get_settings()
    return ProviderSpec(
        name=f"anthropic-org-{org_id}",
        base_url=settings.anthropic_openai_base_url,
        api_key=api_key,
        models=_models(settings.discovery_anthropic_models, "claude-haiku-5-5"),
        supports_vision=True,
        # Current Claude models reject `temperature` with a 400 (llm_providers).
        supports_temperature=False,
    )


_BUILDERS = {
    "gemini": gemini_provider,
    "anthropic": anthropic_provider,
    "groq": groq_provider,
    "openrouter": openrouter_provider,
}


def discovery_providers(
    org_id: uuid.UUID, keys: dict[str, str], *, exclude: tuple[str, ...] = ()
) -> list[ProviderSpec]:
    return [
        _BUILDERS[p](org_id, keys[p])
        for p in DISCOVERY_PROVIDERS
        if p in keys and p not in exclude
    ]


def _provider_family(spec_name: str) -> str:
    """``"groq-org-<uuid>"`` → ``"groq"``."""
    return spec_name.split("-org-", 1)[0]


def _labels(providers: list[str]) -> str:
    return ", ".join(PROVIDER_LABELS.get(p, p) for p in providers)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_json_answer(content: str, schema: type[T], *, provider: str = "Gemini") -> T:
    """Parse a model answer into ``schema``, tolerating a ```json fence."""
    text = _FENCE.sub("", content or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"{provider} returned malformed JSON — try again.") from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise GeminiError(
            f"{provider}'s answer did not match the expected format — try again."
        ) from exc


@dataclass
class ChainAnswer(Generic[T]):
    data: T
    provider: str  # "gemini" | "groq" | "openrouter"
    model: str


async def _chain_json(
    org_id: uuid.UUID,
    providers: list[ProviderSpec],
    *,
    system: str,
    prompt: str,
    schema: type[T],
    max_tokens: int,
    temperature: float | None,
    timeout: float,
) -> ChainAnswer[T]:
    try:
        content, spec_name, model = await chat_completion(
            [{"role": "user", "content": prompt}],
            providers=providers,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            timeout=timeout,
            breaker=_breaker,
        )
    except LLMChainExhausted as exc:
        # The message can carry the upstream status line; log it, but give the
        # user a fixed sentence (it never contains a key either way).
        tried = [_provider_family(p.name) for p in providers]
        logger.warning("Org %s Discovery LLM call failed (%s): %s", org_id, tried, exc)
        keys, verb = ("keys", "are") if len(tried) > 1 else ("key", "is")
        raise GeminiError(
            f"AI request failed on every configured provider ({_labels(tried)}). "
            f"Check that the {_labels(tried)} API {keys} in Settings → API Keys "
            f"{verb} valid and {'have' if len(tried) > 1 else 'has'} quota, then try again."
        ) from exc
    family = _provider_family(spec_name)
    logger.info("Org %s Discovery answer from %s/%s", org_id, family, model)
    label = PROVIDER_LABELS.get(family, family)
    return ChainAnswer(
        data=parse_json_answer(_json_object_text(content), schema, provider=label),
        provider=family,
        model=model,
    )


async def generate_json(
    db: AsyncSession,
    org_id: uuid.UUID,
    *,
    system: str,
    prompt: str,
    schema: type[T],
    max_tokens: int = 8192,
    temperature: float | None = 0.2,
    timeout: float = 30.0,
) -> T:
    """Ask the org's providers (Gemini → Groq → OpenRouter) for JSON matching ``schema``.

    ``chat_completion`` walks the chain: a 404/429/5xx tries the provider's next
    model, anything else (or running out of models) the next provider.

    Raises :class:`GeminiKeyMissing` when the org has no key at all and
    :class:`GeminiError` when every provider fails or answers off-contract.
    """
    keys = await get_org_llm_keys(db, org_id)
    if not keys:
        raise GeminiKeyMissing()
    answer = await _chain_json(
        org_id,
        discovery_providers(org_id, keys),
        system=system,
        prompt=prompt,
        schema=schema,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    return answer.data


@dataclass
class GroundedAnswer(Generic[T]):
    """A search answer plus the web pages Google Search grounded it on.

    ``grounded`` is False when Gemini's grounded call failed and a fallback
    provider answered from training data; ``notice`` then says so in words the
    caller must show next to the results.
    """

    data: T
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    provider: str = "gemini"
    grounded: bool = True
    notice: Optional[str] = None


def _json_object_text(content: str) -> str:
    """The JSON object inside a grounded answer.

    Grounded calls cannot use JSON mode (Gemini rejects ``responseMimeType``
    alongside the search tool), so the model sometimes wraps the object in a
    sentence or a fence. Take the outermost ``{...}``.
    """
    text = _FENCE.sub("", content or "").strip()
    if text.startswith("{"):
        return text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _grounding_sources(candidate: dict) -> list[dict]:
    chunks = ((candidate.get("groundingMetadata") or {}).get("groundingChunks")) or []
    sources: list[dict] = []
    seen: set[str] = set()
    for chunk in chunks:
        web = chunk.get("web") or {}
        uri = web.get("uri")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        sources.append({"title": web.get("title") or "", "url": uri})
    return sources


async def generate_grounded_json(
    db: AsyncSession,
    org_id: uuid.UUID,
    *,
    system: str,
    prompt: str,
    schema: type[T],
    fallback_system: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float | None = 0.2,
    timeout: float = 120.0,
    skip_gemini_grounding: Optional[str] = None,
) -> GroundedAnswer[T]:
    """Grounded Gemini search, falling back to ungrounded Groq/OpenRouter.

    ``skip_gemini_grounding`` is the reason grounding was already tried and
    failed upstream (Discovery's web-search path runs in between): the grounded
    call is not repeated — another 429 is all it would buy — and the reason is
    what the notice reports.

    Gemini (when the org has a key) is tried first with Google Search. If it
    fails for any reason, the org's other providers answer the same prompt from
    training data, with ``fallback_system`` (which must not promise a web
    search the model can't do) and ``grounded=False``.

    Raises :class:`GeminiKeyMissing` with no key at all, and
    :class:`GeminiError` when every provider fails.
    """
    keys = await get_org_llm_keys(db, org_id)
    if not keys:
        raise GeminiKeyMissing()

    gemini_error: Optional[GeminiError] = (
        GeminiError(skip_gemini_grounding) if skip_gemini_grounding else None
    )
    if "gemini" in keys and not skip_gemini_grounding:
        try:
            return await _grounded_gemini(
                org_id, keys["gemini"], system=system, prompt=prompt, schema=schema,
                max_tokens=max_tokens, temperature=temperature, timeout=timeout,
            )
        except GeminiError as exc:
            gemini_error = exc

    fallbacks = discovery_providers(org_id, keys, exclude=("gemini",))
    if not fallbacks:
        assert gemini_error is not None  # keys is non-empty, so gemini was tried
        raise gemini_error

    try:
        answer = await _chain_json(
            org_id, fallbacks, system=fallback_system or system, prompt=prompt,
            schema=schema, max_tokens=max_tokens, temperature=temperature,
            timeout=timeout,
        )
    except GeminiError as exc:
        if gemini_error is None:
            raise
        raise GeminiError(f"{gemini_error} Fallback providers also failed: {exc}") from exc

    label = PROVIDER_LABELS.get(answer.provider, answer.provider)
    why = (
        f"Gemini search was unavailable ({gemini_error})"
        if gemini_error is not None
        else "No Gemini key is saved"
    )
    return GroundedAnswer(
        data=answer.data,
        sources=[],
        model=answer.model,
        provider=answer.provider,
        grounded=False,
        notice=(
            f"{why}, so these results come from {label}'s training data, not a "
            "live web search. They may be out of date or include companies that "
            "no longer exist — verify before outreach. Contact emails are omitted."
        ),
    )


async def generate_gemini_grounded_json(
    db: AsyncSession,
    org_id: uuid.UUID,
    *,
    system: str,
    prompt: str,
    schema: type[T],
    max_tokens: int = 8192,
    temperature: float | None = 0.2,
    timeout: float = 120.0,
) -> GroundedAnswer[T]:
    """Gemini with Google Search grounding and nothing else — no fallback.

    For callers with their own next step (Discovery's web search). Raises
    :class:`GeminiKeyMissing` without a Gemini key, :class:`GeminiError` when
    the grounded call fails.
    """
    key = (await get_org_llm_keys(db, org_id)).get("gemini")
    if not key:
        raise GeminiKeyMissing()
    return await _grounded_gemini(
        org_id, key, system=system, prompt=prompt, schema=schema,
        max_tokens=max_tokens, temperature=temperature, timeout=timeout,
    )


async def generate_json_answer(
    db: AsyncSession,
    org_id: uuid.UUID,
    *,
    system: str,
    prompt: str,
    schema: type[T],
    max_tokens: int = 8192,
    temperature: float | None = 0.2,
    timeout: float = 60.0,
) -> ChainAnswer[T]:
    """:func:`generate_json`, also reporting which provider and model answered."""
    keys = await get_org_llm_keys(db, org_id)
    if not keys:
        raise GeminiKeyMissing()
    return await _chain_json(
        org_id, discovery_providers(org_id, keys), system=system, prompt=prompt,
        schema=schema, max_tokens=max_tokens, temperature=temperature, timeout=timeout,
    )


async def _grounded_gemini(
    org_id: uuid.UUID,
    key: str,
    *,
    system: str,
    prompt: str,
    schema: type[T],
    max_tokens: int,
    temperature: float | None,
    timeout: float,
) -> GroundedAnswer[T]:
    """Gemini with Google Search grounding.

    Grounding is only on the native ``generateContent`` API, so this is the one
    call here that does not go through ``chat_completion``. It keeps the same
    contract: the org's key only, the private per-org breaker, and errors that
    are safe to show and never carry the key.
    """
    settings = get_settings()
    name = f"gemini-org-{org_id}"
    if _breaker.is_open(name):
        raise GeminiError(
            "Gemini has failed several times in a row for this workspace — "
            "wait a couple of minutes, then try again."
        )

    generation: dict = {"maxOutputTokens": max_tokens}
    if temperature is not None:
        generation["temperature"] = temperature
    budget = settings.discovery_search_thinking_budget
    if budget is not None and budget >= 0:
        generation["thinkingConfig"] = {"thinkingBudget": budget}
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": generation,
    }
    base = settings.gemini_native_base_url.rstrip("/")
    models = discovery_models()

    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            for i, model in enumerate(models):
                resp = await http.post(
                    f"{base}/models/{model}:generateContent",
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                    json=payload,
                )
                # Both are properties of the model, not the key, so the next
                # model is a real chance (same rule as chat_completion): 404 is
                # a model retired or closed to accounts created after a cutoff;
                # 429 on the free tier is a per-model daily allowance (20/day
                # on some models), so a spent model says nothing of its sibling.
                if resp.status_code in (404, 429) and i < len(models) - 1:
                    logger.warning(
                        "Org %s grounded Gemini model %s returned %s — trying %s",
                        org_id, model, resp.status_code, models[i + 1],
                    )
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        _breaker.record_failure(name)
        logger.warning("Org %s grounded Gemini call returned HTTP %s", org_id, status)
        if status == 429:
            raise GeminiError(
                "Gemini rate limit or quota reached for your API key — wait a "
                "minute, or check the key's quota in Google AI Studio."
            ) from exc
        if status == 404:
            raise GeminiError(
                "None of the configured Gemini models is available to your API "
                "key. Ask your administrator to update DISCOVERY_GEMINI_MODEL."
            ) from exc
        if status in (400, 401, 403):
            raise GeminiError(
                "Gemini refused the request. Check that the Gemini API key in "
                "Settings → API Keys is valid and allows Google Search grounding."
            ) from exc
        raise GeminiError("Gemini is unavailable right now — try again shortly.") from exc
    except Exception as exc:  # transport error, timeout, non-JSON body
        _breaker.record_failure(name)
        logger.warning(
            "Org %s grounded Gemini call failed (%s)", org_id, type(exc).__name__
        )
        raise GeminiError("Gemini request failed — try again shortly.") from exc

    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    parts = (candidate.get("content") or {}).get("parts") or []
    content = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not content.strip():
        _breaker.record_failure(name)
        logger.warning(
            "Org %s grounded Gemini returned no text (finish_reason=%s, block=%s)",
            org_id,
            candidate.get("finishReason"),
            (data.get("promptFeedback") or {}).get("blockReason"),
        )
        raise GeminiError("Gemini returned an empty answer — try again.")

    _breaker.record_success(name)
    logger.info("Org %s grounded Gemini answer from %s", org_id, model)
    return GroundedAnswer(
        data=parse_json_answer(_json_object_text(content), schema),
        sources=_grounding_sources(candidate),
        model=model,
    )


def reset_breaker() -> None:
    """Tests only."""
    _breaker.reset()
