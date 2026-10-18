"""Web search for Discovery, on the org's own search-API key.

Gemini's Google Search grounding is the only *built-in* way an LLM here can see
the live web, and on the free tier it has its own small quota: for this
platform's first org it returned 429 on every grounded call it ever made, so
every company Discovery found came from Groq's training data — the same
household names every run (Stripe, Salesforce, Shopify), with CEO names from
memory, several of them wrong.

This module is the alternative grounding. A search API finds real pages; the
org's LLM (any provider — Gemini, Claude, Groq, OpenRouter) then reads *only
those pages* and extracts companies from them (:mod:`gemini_discovery`). The
LLM no longer has to know which companies exist, only to read.

Three providers, tried in this order, all with free tiers:

* **Tavily** — built for LLM agents; returns cleaned page content, not just a
  snippet. 1,000 searches/month free, no card. The recommended default.
* **Serper** — Google results as JSON. 2,500 free searches on signup.
* **Brave Search** — independent index. Free plan with a monthly allowance.

Keys live in ``org_llm_keys`` beside the LLM keys (Settings → API Keys) and are
never logged or returned. Every call fails soft: a provider that errors is
skipped for the next one, and only when none answers does the caller get
:class:`WebSearchError` — which Discovery treats as "fall back", never fatal.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import org_gemini
from app.services.normalize import normalize_domain

logger = logging.getLogger(__name__)

TIMEOUT = 20.0
SNIPPET_CHARS = 600


class WebSearchError(Exception):
    """No configured search provider answered. ``str()`` is safe to show users."""


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str

    @property
    def domain(self) -> Optional[str]:
        return normalize_domain(urlparse(self.url).netloc) if self.url else None


async def _tavily(http: httpx.AsyncClient, key: str, query: str, n: int) -> list[SearchHit]:
    resp = await http.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query, "max_results": n, "search_depth": "basic"},
    )
    resp.raise_for_status()
    return [
        SearchHit(r.get("title") or "", r.get("url") or "", (r.get("content") or "")[:SNIPPET_CHARS])
        for r in (resp.json().get("results") or [])
    ]


async def _serper(http: httpx.AsyncClient, key: str, query: str, n: int) -> list[SearchHit]:
    resp = await http.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": n},
    )
    resp.raise_for_status()
    return [
        SearchHit(r.get("title") or "", r.get("link") or "", (r.get("snippet") or "")[:SNIPPET_CHARS])
        for r in (resp.json().get("organic") or [])
    ]


async def _brave(http: httpx.AsyncClient, key: str, query: str, n: int) -> list[SearchHit]:
    resp = await http.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"X-Subscription-Token": key, "Accept": "application/json"},
        params={"q": query, "count": min(n, 20)},
    )
    resp.raise_for_status()
    return [
        SearchHit(r.get("title") or "", r.get("url") or "", (r.get("description") or "")[:SNIPPET_CHARS])
        for r in ((resp.json().get("web") or {}).get("results") or [])
    ]


_PROVIDERS = {"tavily": _tavily, "serper": _serper, "brave": _brave}


async def has_search_key(db: AsyncSession, org_id: uuid.UUID) -> bool:
    return bool(await org_gemini.get_org_search_keys(db, org_id))


async def search(
    db: AsyncSession, org_id: uuid.UUID, queries: list[str], *, per_query: int = 10
) -> tuple[str, list[SearchHit]]:
    """Run ``queries`` on the org's first working search provider.

    Returns ``(provider, hits)`` with hits de-duplicated by URL, in query order.
    A provider that fails on its first query is abandoned for the next one; a
    single failing query on a working provider is skipped, not fatal.

    Raises :class:`WebSearchError` when the org has no search key, or when no
    provider answered any query.
    """
    keys = await org_gemini.get_org_search_keys(db, org_id)
    if not keys:
        raise WebSearchError("No web search key is saved (Settings → API Keys).")
    queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as http:
        for provider, key in keys.items():
            fn = _PROVIDERS[provider]
            results = await asyncio.gather(
                *(fn(http, key, q, per_query) for q in queries), return_exceptions=True
            )
            hits: list[SearchHit] = []
            seen: set[str] = set()
            ok = 0
            for q, r in zip(queries, results):
                if isinstance(r, BaseException):
                    status = getattr(getattr(r, "response", None), "status_code", None)
                    logger.warning(
                        "Org %s %s search failed for one query (%s)",
                        org_id, provider, status or type(r).__name__,
                    )
                    continue
                ok += 1
                for h in r:
                    if h.url and h.url not in seen:
                        seen.add(h.url)
                        hits.append(h)
            if ok:
                logger.info(
                    "Org %s web search via %s: %d/%d queries answered, %d results",
                    org_id, provider, ok, len(queries), len(hits),
                )
                return provider, hits
            failures.append(org_gemini.PROVIDER_LABELS.get(provider, provider))
    raise WebSearchError(
        f"Web search failed on every configured provider ({', '.join(failures)}). "
        "Check the key and its remaining quota in Settings → API Keys."
    )
