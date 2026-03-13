from __future__ import annotations

from dataclasses import dataclass
import json
import time
import urllib.error
import urllib.request
from pokemon_agent.config import OpenRouterConfig

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


@dataclass(slots=True)
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class CompletionResponse:
    content: str
    model: str | None = None
    usage: LLMUsage | None = None


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig) -> None:
        self.config = config
        self._last_request_started_at: float | None = None

    def complete(self, messages: list[dict]) -> CompletionResponse:
        if not self.config.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")

        payload = {
            "model": self.config.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": self.config.site_url,
            "X-Title": self.config.app_name,
        }
        url = f"{self.config.base_url}/chat/completions"
        last_error: Exception | None = None
        for _ in range(self.config.max_retries + 1):
            try:
                self._wait_for_min_interval()
                data = self._post_json(url, payload, headers)
                content = data["choices"][0]["message"]["content"]
                usage_data = data.get("usage") or {}
                usage = LLMUsage(
                    prompt_tokens=usage_data.get("prompt_tokens"),
                    completion_tokens=usage_data.get("completion_tokens"),
                    total_tokens=usage_data.get("total_tokens"),
                )
                if isinstance(content, list):
                    text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                    return CompletionResponse(
                        content="".join(text_parts).strip(),
                        model=data.get("model"),
                        usage=usage,
                    )
                return CompletionResponse(content=str(content).strip(), model=data.get("model"), usage=usage)
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(f"OpenRouter completion failed: {last_error}") from last_error

    def _wait_for_min_interval(self) -> None:
        if self.config.min_request_interval_seconds <= 0:
            return

        now = time.monotonic()
        if self._last_request_started_at is not None:
            elapsed = now - self._last_request_started_at
            remaining = self.config.min_request_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
                now += remaining
        self._last_request_started_at = now

    def _post_json(self, url: str, payload: dict, headers: dict) -> dict:
        if httpx is not None:
            with httpx.Client(timeout=self.config.timeout_seconds) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
