"""/settings/llm-key — the org's own Gemini key, as saved from Settings → API Keys."""

import pytest
from cryptography.fernet import Fernet

from app.config import get_settings
from app.models.user import UserRole
from app.org_context import APPREND_ORG_ID
from app.services import crypto, org_gemini

URL = "/api/v1/settings/llm-key"


@pytest.fixture(autouse=True)
def _crypto_key():
    settings = get_settings()
    prev = settings.token_encryption_key
    settings.token_encryption_key = Fernet.generate_key().decode()
    crypto._fernet.cache_clear()
    yield
    settings.token_encryption_key = prev
    crypto._fernet.cache_clear()


@pytest.mark.asyncio
async def test_save_then_read_back_and_discovery_sees_it(client, db_session):
    r = await client.put(URL, json={"api_key": "  AIzaSyTestKey1234  ", "provider": "gemini"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_key"] is True and body["provider"] == "gemini"
    assert "AIzaSyTestKey1234" not in body["key_hint"]

    r = await client.get(URL)
    assert r.status_code == 200
    assert r.json()["has_key"] is True

    # The key Discovery actually uses is the trimmed one.
    assert await org_gemini.get_org_gemini_key(db_session, APPREND_ORG_ID) == "AIzaSyTestKey1234"


@pytest.mark.asyncio
async def test_replacing_keeps_a_single_key(client, db_session):
    assert (await client.put(URL, json={"api_key": "first-key-aaaa"})).status_code == 200
    assert (await client.put(URL, json={"api_key": "second-key-bbbb"})).status_code == 200
    assert await org_gemini.get_org_gemini_key(db_session, APPREND_ORG_ID) == "second-key-bbbb"


@pytest.mark.asyncio
async def test_delete_removes_the_key(client, db_session):
    await client.put(URL, json={"api_key": "AIzaSyDeleteMe"})
    assert (await client.delete(URL)).status_code == 204
    assert (await client.get(URL)).json()["has_key"] is False
    assert await org_gemini.get_org_gemini_key(db_session, APPREND_ORG_ID) is None


@pytest.mark.asyncio
async def test_blank_key_rejected_and_nothing_stored(client, db_session):
    r = await client.put(URL, json={"api_key": "   "})
    assert r.status_code == 422
    assert (await client.get(URL)).json()["has_key"] is False


@pytest.mark.asyncio
async def test_unknown_provider_rejected(client):
    # Previously stored under "openai" and reported has_key=True, but GET and
    # Discovery only ever read "gemini".
    r = await client.put(URL, json={"api_key": "sk-something", "provider": "openai"})
    assert r.status_code == 422
    assert (await client.get(URL)).json()["has_key"] is False


@pytest.mark.asyncio
async def test_requires_a_bearer_token(unauth_client):
    r = await unauth_client.put(URL, json={"api_key": "AIzaSyNoAuth"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_non_admin_cannot_save(client, make_user, auth_headers):
    agent = await make_user("agent-llm@landline.app", role=UserRole.AGENT.value)
    r = await client.put(URL, json={"api_key": "AIzaSyAgent"}, headers=auth_headers(agent))
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_groq_and_openrouter_keys_are_saved_per_provider(client, db_session):
    assert (await client.put(URL, json={"api_key": "gem-key-1111"})).status_code == 200
    r = await client.put(URL, json={"api_key": "gsk_groq_2222", "provider": "groq"})
    assert r.status_code == 200 and r.json()["provider"] == "groq"
    r = await client.put(URL, json={"api_key": "sk-or-3333", "provider": "openrouter"})
    assert r.status_code == 200

    # The old no-param GET still means Gemini.
    assert (await client.get(URL)).json()["provider"] == "gemini"
    g = (await client.get(URL, params={"provider": "groq"})).json()
    assert g["has_key"] is True and "gsk_groq_2222" not in g["key_hint"]

    listing = (await client.get("/api/v1/settings/llm-keys")).json()
    # LLMs in fallback order (Gemini default, Claude optional), then web search.
    assert [k["provider"] for k in listing] == [
        "gemini", "anthropic", "groq", "openrouter", "tavily", "serper", "brave"]
    assert [k["kind"] for k in listing] == ["llm"] * 4 + ["search"] * 3
    assert {k["provider"] for k in listing if k["has_key"]} == {"gemini", "groq", "openrouter"}

    keys = await org_gemini.get_org_llm_keys(db_session, APPREND_ORG_ID)
    assert keys == {"gemini": "gem-key-1111", "groq": "gsk_groq_2222", "openrouter": "sk-or-3333"}


@pytest.mark.asyncio
async def test_deleting_groq_leaves_gemini(client, db_session):
    await client.put(URL, json={"api_key": "gem-key-1111"})
    await client.put(URL, json={"api_key": "gsk_groq_2222", "provider": "groq"})
    assert (await client.delete(URL, params={"provider": "groq"})).status_code == 204
    assert list(await org_gemini.get_org_llm_keys(db_session, APPREND_ORG_ID)) == ["gemini"]
    listing = (await client.get("/api/v1/settings/llm-keys")).json()
    assert [k["has_key"] for k in listing] == [True] + [False] * 6


@pytest.mark.asyncio
async def test_unknown_provider_rejected_on_get_and_delete(client):
    assert (await client.get(URL, params={"provider": "openai"})).status_code == 422
    assert (await client.delete(URL, params={"provider": "openai"})).status_code == 422
