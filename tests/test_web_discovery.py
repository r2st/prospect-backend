"""Discovery on a real web search when Gemini's Google Search grounding isn't available.

In production the org's Gemini key 429'd every grounded call it ever made, so
every discovered company came from Groq's training data — the same household
names each run, CEO names from memory. These pin the alternative: a search API
(Tavily / Serper / Brave) finds pages, any LLM reads only those, and anything it
did not read there is dropped. Plus Claude as an optional LLM provider, and the
ingestion fix for people re-inserted on every run.
"""

import json

import httpx
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from app.config import get_settings
from app.models.company import Company
from app.models.contact import Contact
from app.models.org_llm_key import OrgLlmKey
from app.org_context import APPREND_ORG_ID
from app.services import crypto, gemini_discovery, org_gemini, web_search
from app.services.gemini_discovery import GeminiCompany, GeminiContact, GeminiSearchAnswer
from app.services.web_search import SearchHit

ORG = APPREND_ORG_ID


@pytest.fixture(autouse=True)
def _crypto_key():
    settings = get_settings()
    prev = settings.token_encryption_key
    settings.token_encryption_key = Fernet.generate_key().decode()
    crypto._fernet.cache_clear()
    org_gemini.reset_breaker()
    yield
    settings.token_encryption_key = prev
    crypto._fernet.cache_clear()
    org_gemini.reset_breaker()


async def _keys(db, **keys):
    for provider, key in keys.items():
        db.add(OrgLlmKey(org_id=ORG, provider=provider, key_encrypted=crypto.encrypt(key), key_hint="••••"))
    await db.commit()


def _mock_http(monkeypatch, handler):
    calls = []
    real = httpx.AsyncClient

    def wrapped(request):
        calls.append(request)
        return handler(request)

    class _Client(real):
        def __init__(self, *a, **kw):
            kw["transport"] = httpx.MockTransport(wrapped)
            super().__init__(*a, **kw)

    monkeypatch.setattr(web_search.httpx, "AsyncClient", _Client)
    return calls


# ── the search providers ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_without_a_search_key_there_is_no_web_search(db_session):
    await _keys(db_session, gemini="g")
    assert not await web_search.has_search_key(db_session, ORG)
    with pytest.raises(web_search.WebSearchError):
        await web_search.search(db_session, ORG, ["x"])


@pytest.mark.asyncio
async def test_tavily_results_are_parsed_and_deduplicated(db_session, monkeypatch):
    await _keys(db_session, tavily="tvly-k")

    def handler(req):
        assert req.url.host == "api.tavily.com"
        assert req.headers["authorization"] == "Bearer tvly-k"
        q = json.loads(req.content)["query"]
        return httpx.Response(200, json={"results": [
            {"title": "Acme Ops", "url": "https://acme.io/about", "content": f"Acme runs SRE ({q})"},
            {"title": "Shared", "url": "https://shared.com", "content": "both queries see this"},
        ]})

    calls = _mock_http(monkeypatch, handler)
    provider, hits = await web_search.search(db_session, ORG, ["one", "two", "one"])
    assert provider == "tavily"
    assert len(calls) == 2  # the repeated query is asked once
    assert [h.url for h in hits] == ["https://acme.io/about", "https://shared.com"]
    assert hits[0].domain == "acme.io"


@pytest.mark.asyncio
async def test_a_failing_provider_falls_through_to_the_next(db_session, monkeypatch):
    await _keys(db_session, tavily="tvly-k", serper="srp-k")

    def handler(req):
        if req.url.host == "api.tavily.com":
            return httpx.Response(432, json={"detail": "quota"})
        assert req.headers["x-api-key"] == "srp-k"
        return httpx.Response(200, json={"organic": [
            {"title": "Beta", "link": "https://beta.dev", "snippet": "Beta DevOps"}]})

    _mock_http(monkeypatch, handler)
    provider, hits = await web_search.search(db_session, ORG, ["q"])
    assert provider == "serper" and hits[0].url == "https://beta.dev"


@pytest.mark.asyncio
async def test_brave_is_parsed(db_session, monkeypatch):
    await _keys(db_session, brave="brv-k")

    def handler(req):
        assert req.headers["x-subscription-token"] == "brv-k"
        return httpx.Response(200, json={"web": {"results": [
            {"title": "Gamma", "url": "https://gamma.ai", "description": "Gamma AIOps"}]}})

    _mock_http(monkeypatch, handler)
    assert (await web_search.search(db_session, ORG, ["q"]))[1][0].snippet == "Gamma AIOps"


@pytest.mark.asyncio
async def test_every_provider_failing_is_a_web_search_error_that_names_no_key(db_session, monkeypatch):
    await _keys(db_session, tavily="tvly-secret")
    _mock_http(monkeypatch, lambda req: httpx.Response(401))
    with pytest.raises(web_search.WebSearchError) as exc:
        await web_search.search(db_session, ORG, ["q"])
    assert "Tavily" in str(exc.value) and "tvly-secret" not in str(exc.value)


# ── query angles and grounding ───────────────────────────────────────────────

def test_one_profile_becomes_several_distinct_queries():
    qs = gemini_discovery.web_queries(
        query="Mid-market SaaS with SRE teams\n\nDo not include: x.com",
        industry="Software, Fintech", location="Global",
        keywords=["incident remediation", "AIOps"], n=6,
    )
    assert len(qs) >= 4 and len(set(qs)) == len(qs)
    assert all("Global" not in q and "Do not include" not in q for q in qs)
    assert any("Fintech" in q for q in qs) and any("incident remediation" in q for q in qs)


def _hits():
    return [
        SearchHit("Top AIOps startups: Acme, Beta Labs", "https://blog.example/top-aiops",
                  "Acme (acme.io) and Beta Labs automate remediation. Priya Shah, CTO at Acme; "
                  "contact sales@acme.io"),
        SearchHit("Beta Labs | LinkedIn", "https://linkedin.com/company/beta-labs", "Beta Labs"),
    ]


def test_grounding_drops_what_the_results_never_mention():
    answer = GeminiSearchAnswer(companies=[
        GeminiCompany(name="Acme", domain="acme.io", contacts=[
            GeminiContact(name="Priya Shah", title="CTO", email="sales@acme.io"),
            GeminiContact(name="Invented Person", title="CEO", email="ceo@acme.io"),
        ]),
        GeminiCompany(name="Beta Labs", domain="betalabs.com"),   # name grounded, domain from memory
        GeminiCompany(name="Stripe", domain="stripe.com"),        # never in the results
        GeminiCompany(name=None, domain="linkedin.com"),          # an aggregator, not a company
    ])
    kept = gemini_discovery.ground_in_hits(answer, _hits())
    by_name = {c.name: c for c in kept.companies}
    assert set(by_name) == {"Acme", "Beta Labs"}
    assert [p.name for p in by_name["Acme"].contacts] == ["Priya Shah"]
    assert by_name["Acme"].contacts[0].email == "sales@acme.io"   # printed in the results
    assert by_name["Beta Labs"].domain == "betalabs.com"          # kept: the name is grounded


def test_an_email_the_results_dont_print_is_dropped():
    answer = GeminiSearchAnswer(companies=[GeminiCompany(name="Acme", domain="acme.io", contacts=[
        GeminiContact(name="Priya Shah", email="priya.shah@acme.io")])])
    kept = gemini_discovery.ground_in_hits(answer, _hits())
    assert kept.companies[0].contacts[0].email is None


# ── the order: Gemini grounded → web search + LLM → training data ────────────

def _answer(*companies):
    return GeminiSearchAnswer(companies=list(companies))


@pytest.mark.asyncio
async def test_gemini_grounding_still_comes_first(db_session, monkeypatch):
    await _keys(db_session, gemini="g", tavily="t")

    async def grounded(*a, **kw):
        return org_gemini.GroundedAnswer(data=_answer(GeminiCompany(name="G", domain="g.com")))

    async def no_web(*a, **kw):
        raise AssertionError("web search must not run when Gemini grounding works")

    monkeypatch.setattr(org_gemini, "generate_gemini_grounded_json", grounded)
    monkeypatch.setattr(web_search, "search", no_web)
    found = await gemini_discovery.search_companies(db_session, ORG, query="q")
    assert found.grounded and [r.domain for r in found.records] == ["g.com"]


@pytest.mark.asyncio
async def test_gemini_429_with_a_search_key_uses_the_web_and_keeps_it_grounded(db_session, monkeypatch):
    await _keys(db_session, gemini="g", groq="gq", tavily="t")

    async def grounded_429(*a, **kw):
        raise org_gemini.GeminiError("Gemini rate limit or quota reached")

    async def fake_search(db, org_id, queries, *, per_query=10):
        assert len(queries) > 1
        return "tavily", _hits()

    async def fake_llm(db, org_id, *, prompt, schema, **kw):
        assert "Acme" in prompt and "RESULTS" in prompt
        return org_gemini.ChainAnswer(
            data=_answer(GeminiCompany(name="Acme", domain="acme.io", employee_count=120, contacts=[
                GeminiContact(name="Priya Shah", title="CTO", email="sales@acme.io")]),
                GeminiCompany(name="Stripe", domain="stripe.com")),
            provider="groq", model="m")

    async def no_fallback(*a, **kw):
        raise AssertionError("the training-data fallback must not run")

    monkeypatch.setattr(org_gemini, "generate_gemini_grounded_json", grounded_429)
    monkeypatch.setattr(web_search, "search", fake_search)
    monkeypatch.setattr(org_gemini, "generate_json_answer", fake_llm)
    monkeypatch.setattr(org_gemini, "generate_grounded_json", no_fallback)

    found = await gemini_discovery.search_companies(db_session, ORG, query="SRE teams", industry="Software")
    assert found.grounded is True and found.provider == "tavily+groq"
    assert [r.domain for r in found.records] == ["acme.io"]           # Stripe: not in the results
    assert found.records[0].contacts[0].email == "sales@acme.io"      # grounded → emails kept
    assert {s["url"] for s in found.sources} >= {"https://blog.example/top-aiops"}


@pytest.mark.asyncio
async def test_web_failure_falls_back_without_asking_gemini_twice(db_session, monkeypatch):
    await _keys(db_session, gemini="g", groq="gq", tavily="t")
    asked = []

    async def grounded_429(*a, **kw):
        asked.append("grounded")
        raise org_gemini.GeminiError("Gemini rate limit")

    async def broken_search(*a, **kw):
        raise web_search.WebSearchError("Tavily quota")

    async def fallback(db, org_id, *, skip_gemini_grounding=None, **kw):
        assert skip_gemini_grounding and "Gemini" in skip_gemini_grounding and "Tavily" in skip_gemini_grounding
        return org_gemini.GroundedAnswer(data=_answer(GeminiCompany(name="X", domain="x.com")),
                                         provider="groq", grounded=False, notice="training data")

    monkeypatch.setattr(org_gemini, "generate_gemini_grounded_json", grounded_429)
    monkeypatch.setattr(web_search, "search", broken_search)
    monkeypatch.setattr(org_gemini, "generate_grounded_json", fallback)
    found = await gemini_discovery.search_companies(db_session, ORG, query="q")
    assert asked == ["grounded"] and found.grounded is False


@pytest.mark.asyncio
async def test_known_domains_are_filtered_out_of_the_results(db_session, monkeypatch):
    await _keys(db_session, gemini="g")

    async def grounded(*a, prompt, **kw):
        assert "old.com" in prompt
        return org_gemini.GroundedAnswer(data=_answer(
            GeminiCompany(name="Old", domain="old.com"), GeminiCompany(name="New", domain="new.com")))

    monkeypatch.setattr(org_gemini, "generate_grounded_json", grounded)
    found = await gemini_discovery.search_companies(db_session, ORG, query="q", exclude_domains=["old.com"])
    assert [r.domain for r in found.records] == ["new.com"]


# ── Claude as an optional provider ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_claude_is_second_in_the_chain_after_gemini(db_session):
    await _keys(db_session, groq="gq", anthropic="sk-ant-x", gemini="g")
    keys = await org_gemini.get_org_llm_keys(db_session, ORG)
    assert list(keys) == ["gemini", "anthropic", "groq"]
    specs = org_gemini.discovery_providers(ORG, keys)
    claude = specs[1]
    assert claude.name == f"anthropic-org-{ORG}"
    assert claude.endpoint == "https://api.anthropic.com/v1/chat/completions"
    assert claude.api_key == "sk-ant-x" and claude.models[0].startswith("claude-")


@pytest.mark.asyncio
async def test_search_keys_are_not_llm_keys(db_session):
    await _keys(db_session, tavily="t")
    assert await org_gemini.get_org_llm_keys(db_session, ORG) == {}
    assert await org_gemini.get_org_search_keys(db_session, ORG) == {"tavily": "t"}


# ── ingestion: the same person found twice is one person ────────────────────

@pytest.mark.asyncio
async def test_a_person_without_an_email_is_not_re_inserted_on_the_next_run(db_session):
    from app.services.ingestion import CandidateRecord, ContactRecord, get_ingestion_service

    def record():
        return CandidateRecord(name="Stripe", domain="stripe.com", source="groq", contacts=[
            ContactRecord(first_name="Patrick", last_name="Collison", source="groq"),
            ContactRecord(first_name="John", last_name="Collison", source="groq"),
        ])

    svc = get_ingestion_service()
    for _ in range(3):
        await svc.ingest(db_session, [record()])
    await db_session.commit()
    co = (await db_session.execute(select(Company).where(Company.domain == "stripe.com"))).scalars().one()
    names = (await db_session.execute(select(Contact.first_name).where(Contact.company_id == co.id))).scalars().all()
    assert sorted(names) == ["John", "Patrick"]



# ── customers, not competitors; domains from knowledge; names excluded ──────

def test_the_prompts_ask_for_buyers_not_competitors():
    criteria_prompt = gemini_discovery.build_prompt(
        query="SRE teams", industry="", location="", keywords=[], min_employees=None,
        max_employees=None, limit=5, offering="Autonomous incident remediation")
    web = gemini_discovery.web_prompt(_hits(), "criteria", 5, "Autonomous incident remediation")
    for p in (criteria_prompt, web):
        assert "CUSTOMERS" in p and "Autonomous incident remediation" in p
        assert "competitors, not buyers" in p
    # Without an offering (the manual Discovery search) the prompt is unchanged.
    plain = gemini_discovery.build_prompt(query="x", industry="", location="", keywords=[],
                                          min_employees=None, max_employees=None, limit=5)
    assert "CUSTOMERS" not in plain


def test_the_web_prompt_asks_for_a_known_domain_when_the_results_omit_it():
    web = gemini_discovery.web_prompt(_hits(), "criteria", 5)
    assert "otherwise the domain you know" in web


def test_a_known_domain_for_a_named_company_survives_grounding():
    """Komodor and Squadcast were named in listicles that printed no domains;
    the model's known domain is kept because the name is in the results."""
    hits = [SearchHit("Top incident tools", "https://blog.example/x", "Komodor and Squadcast lead the pack")]
    answer = GeminiSearchAnswer(companies=[
        GeminiCompany(name="Komodor", domain="komodor.com"),
        GeminiCompany(name="Squadcast", domain="squadcast.com"),
    ])
    kept = gemini_discovery.ground_in_hits(answer, hits)
    assert {c.domain for c in kept.companies} == {"komodor.com", "squadcast.com"}


@pytest.mark.asyncio
async def test_known_names_are_filtered_even_without_a_domain(db_session, monkeypatch):
    await _keys(db_session, gemini="g")

    async def grounded(*a, prompt, **kw):
        assert "Razorpay" in prompt
        return org_gemini.GroundedAnswer(data=_answer(
            GeminiCompany(name="Razorpay Software Pvt Ltd", domain=None),
            GeminiCompany(name="Fresh Co", domain="fresh.co")))

    monkeypatch.setattr(org_gemini, "generate_grounded_json", grounded)
    found = await gemini_discovery.search_companies(
        db_session, ORG, query="q", exclude_names=["Razorpay"])
    assert [r.name for r in found.records] == ["Fresh Co"]



@pytest.mark.parametrize("a, b, same", [
    ("Razorpay", "Razorpay Software Pvt Ltd", True),
    ("Gart Solutions", "gart solutions inc.", True),
    ("Komodor", "Komodor Technologies", True),
    ("Precision Manufacturing", "Precision Tools", False),   # never a prefix match
    ("Solutions", "Solutions", True),                         # never stripped to nothing
    ("Cloud Solutions", "Acme Cloud", False),
])
def test_company_match_key(a, b, same):
    assert (gemini_discovery.company_match_key(a) == gemini_discovery.company_match_key(b)) is same
    assert gemini_discovery.company_match_key(a)
