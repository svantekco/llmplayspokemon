from __future__ import annotations

import types

import pytest

from pokemon_agent.agent.llm_client import OpenRouterClient
from pokemon_agent.config import OpenRouterConfig
import pokemon_agent.agent.llm_client as llm_client_module


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "choices": [{"message": {"content": '{"action":"MOVE_UP","repeat":1,"reason":"test"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "fake-model",
        }


class _FakeClient:
    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
        return _FakeResponse()


class _FakeUrlOpenResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    @property
    def headers(self):
        return self

    def get_content_charset(self):
        return "utf-8"

    def read(self) -> bytes:
        return (
            b'{"choices":[{"message":{"content":"{\\"action\\":\\"MOVE_UP\\",\\"repeat\\":1,\\"reason\\":\\"test\\"}"}}],'
            b'"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15},"model":"fake-model"}'
        )


def test_openrouter_client_waits_between_requests(monkeypatch):
    sleep_calls: list[float] = []
    monotonic_values = iter([100.0, 100.2])

    monkeypatch.setattr(llm_client_module, "httpx", types.SimpleNamespace(Client=_FakeClient))
    monkeypatch.setattr(llm_client_module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(llm_client_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key="test-key",
            min_request_interval_seconds=1.0,
            max_retries=0,
        )
    )

    client.complete([{"role": "user", "content": "first"}])
    client.complete([{"role": "user", "content": "second"}])

    assert sleep_calls == [pytest.approx(0.8)]


def test_openrouter_client_uses_urllib_when_httpx_missing(monkeypatch):
    monkeypatch.setattr(llm_client_module, "httpx", None)
    monkeypatch.setattr(llm_client_module.urllib.request, "urlopen", lambda request, timeout: _FakeUrlOpenResponse())

    client = OpenRouterClient(
        OpenRouterConfig(
            api_key="test-key",
            min_request_interval_seconds=0.0,
            max_retries=0,
        )
    )

    response = client.complete([{"role": "user", "content": "hello"}])

    assert response.model == "fake-model"
    assert response.usage is not None
    assert response.usage.total_tokens == 15


def test_openrouter_client_tracks_request_status(monkeypatch):
    snapshots = []
    tracked_client = None

    class _ObservedClient(_FakeClient):
        def post(self, url: str, json: dict, headers: dict) -> _FakeResponse:
            assert tracked_client is not None
            snapshots.append(tracked_client.snapshot_request_status())
            return super().post(url, json, headers)

    monkeypatch.setattr(llm_client_module, "httpx", types.SimpleNamespace(Client=_ObservedClient))

    tracked_client = OpenRouterClient(
        OpenRouterConfig(
            api_key="test-key",
            min_request_interval_seconds=0.0,
            max_retries=0,
        )
    )

    response = tracked_client.complete([{"role": "user", "content": "hello"}])

    assert response.model == "fake-model"
    assert snapshots
    assert snapshots[0] is not None
    assert snapshots[0].phase == "Sending request"
    assert tracked_client.snapshot_request_status() is not None
    assert tracked_client.snapshot_request_status().phase == "Received response"
