"""Test double for the TypeSafe (Jev) API.

Patches ``typesafe_sdk.AsyncTypeSafeClient`` so the real SDK runs — request
building, question validation, response parsing — against an in-process
``httpx2.MockTransport`` instead of the network. Faking at the transport rather
than at our helpers means a wire-shape mismatch with the SDK fails here, not in
production.
"""

from __future__ import annotations

import json
from typing import Optional

import httpx2
import typesafe_sdk

from app.config import get_settings
from app.services import jev_service


class JevMock:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.answers: dict = {}
        self.status: int = 200
        self.raise_exc: Optional[Exception] = None

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.raise_exc is not None:
            raise self.raise_exc
        if self.status != 200:
            return httpx2.Response(self.status, json={"error": {"message": "boom"}})
        return httpx2.Response(
            200,
            json={
                "model": "jev-test",
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "answers": self.answers,
            },
        )

    @property
    def calls(self) -> int:
        return len(self.requests)


def choice_answer(label: str, confidence: float, probabilities: Optional[dict] = None) -> dict:
    return {
        "type": "choice",
        "choice": label,
        "confidence": confidence,
        "probabilities": probabilities or {label: confidence},
    }


def noul_answer(p: float) -> dict:
    return {"type": "noul", "noul": p}


def install(monkeypatch, *, api_key: str = "test-typesafe-key") -> JevMock:
    """Enable Jev with a mocked transport. Returns the mock to program/inspect."""
    mock = JevMock()
    real = typesafe_sdk.AsyncTypeSafeClient

    def _factory(**kwargs):
        return real(transport=httpx2.MockTransport(mock.handler), **kwargs)

    monkeypatch.setattr(typesafe_sdk, "AsyncTypeSafeClient", _factory)
    monkeypatch.setattr(get_settings(), "typesafe_api_key", api_key)
    jev_service.reset_cooldown()
    return mock
