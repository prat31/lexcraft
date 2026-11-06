"""OpenAI-compatible HTTP backend (OpenCode Go, OpenRouter, vLLM, llama-server, ...)."""

from __future__ import annotations

import os
import random
import time
import uuid

import httpx

from lexcraft import __version__
from lexcraft.backends.base import Backend, Messages, ModelDescription

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class RemoteError(RuntimeError):
    pass


class RemoteBackend(Backend):
    name = "remote"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 180.0,
        max_retries: int = 4,
        extra_body: dict | None = None,
        session_id: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not key:
            raise RemoteError(f"Environment variable {api_key_env} is not set (check .env).")
        h = {
            "User-Agent": f"lexcraft/{__version__}",
            "Content-Type": "application/json",
            # Stable per-run session id; OpenCode Go requires it for routing/caching.
            "x-opencode-session": session_id or f"lexcraft-{uuid.uuid4().hex[:12]}",
        }
        if key:
            h["Authorization"] = f"Bearer {key}"
        h.update(headers or {})
        self.client = httpx.Client(timeout=timeout, headers=h)
        self.last_usage: dict = {}

    def describe(self) -> ModelDescription:
        return ModelDescription("remote", self.model, None, self.base_url)

    def chat(self, messages: Messages, max_tokens: int, temperature: float) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.extra_body,
        }
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                r = self.client.post(f"{self.base_url}/chat/completions", json=body)
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise RemoteError(f"{self.model}: network error: {e}") from e
            else:
                if r.status_code == 200:
                    data = r.json()
                    if "choices" not in data:
                        raise RemoteError(f"{self.model}: unexpected response: {r.text[:300]}")
                    return data
                if r.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    raise RemoteError(f"{self.model}: HTTP {r.status_code}: {r.text[:300]}")
                retry_after = r.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    delay = max(delay, float(retry_after))
            time.sleep(delay + random.uniform(0, 1))
            delay = min(delay * 2, 60)
        raise RemoteError("unreachable")

    def generate(self, messages: Messages, max_tokens: int, temperature: float) -> str:
        data = self.chat(messages, max_tokens, temperature)
        self.last_usage = data.get("usage", {}) or {}
        content = data["choices"][0]["message"].get("content") or ""
        return str(content)

    def close(self) -> None:
        self.client.close()
