"""In-app feedback widget endpoint."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_submit_and_list_feedback(client: AsyncClient):
    r = await client.post(
        "/api/v1/feedback",
        json={"page_name": "Campaigns", "feedback_text": "  Love the warm-up hint  "},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["page_name"] == "Campaigns"
    assert body["feedback_text"] == "Love the warm-up hint"
    assert body["user_id"]

    r = await client.get("/api/v1/feedback")
    assert r.status_code == 200, r.text
    assert any(f["id"] == body["id"] for f in r.json())


@pytest.mark.asyncio
async def test_feedback_rejects_empty_text(client: AsyncClient):
    r = await client.post("/api/v1/feedback", json={"page_name": "Inbox", "feedback_text": ""})
    assert r.status_code == 422
