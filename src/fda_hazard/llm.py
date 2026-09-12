"""Provider-agnostic chat client with on-disk response caching.

Groq, OpenRouter and Baseten all speak the OpenAI chat-completions dialect, so
they share one code path and differ only in base URL, env var and default
model. Anthropic's native API is kept as a fourth option because the original
brief called for it.

Every request is cached in SQLite under a hash of everything that could change
the response, so re-running an unchanged evaluation costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Literal

import httpx

ProviderName = Literal["groq", "openrouter", "baseten", "anthropic"]


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    env_var: str
    default_model: str
    dialect: Literal["openai", "anthropic"] = "openai"
    docs: str = ""


PROVIDERS: dict[str, Provider] = {
    "groq": Provider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        env_var="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        docs="https://console.groq.com/keys",
    ),
    "openrouter": Provider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        env_var="OPENROUTER_API_KEY",
        default_model="anthropic/claude-sonnet-4.5",
        docs="https://openrouter.ai/keys",
    ),
    "baseten": Provider(
        name="baseten",
        base_url="https://inference.baseten.co/v1",
        env_var="BASETEN_API_KEY",
        default_model="deepseek-ai/DeepSeek-V3-0324",
        docs="https://app.baseten.co/settings/api_keys",
    ),
    "anthropic": Provider(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-5",
        dialect="anthropic",
        docs="https://console.anthropic.com/settings/keys",
    ),
}

# Checked in this order when no provider is named explicitly.
_DETECT_ORDER = ("groq", "openrouter", "baseten", "anthropic")


class NoProviderConfigured(RuntimeError):
    """Raised when no provider key is present in the environment."""


def available_providers() -> list[str]:
    return [n for n in _DETECT_ORDER if os.environ.get(PROVIDERS[n].env_var)]


def resolve_provider(name: str | None = None) -> Provider:
    """Pick a provider, preferring an explicit name, else the first key found."""
    name = name or os.environ.get("LLM_PROVIDER") or None
    if name:
        if name not in PROVIDERS:
            raise ValueError(
                f"unknown provider {name!r}; choose from {sorted(PROVIDERS)}"
            )
        provider = PROVIDERS[name]
        if not os.environ.get(provider.env_var):
            raise NoProviderConfigured(
                f"provider {name!r} selected but {provider.env_var} is not set. "
                f"Get a key at {provider.docs}"
            )
        return provider

    found = available_providers()
    if not found:
        expected = ", ".join(PROVIDERS[n].env_var for n in _DETECT_ORDER)
        raise NoProviderConfigured(
            "no LLM provider key found in the environment. Set one of: "
            f"{expected} (see .env.example), or pass --provider."
        )
    return PROVIDERS[found[0]]


def resolve_model(provider: Provider, model: str | None = None) -> str:
    return model or os.environ.get("LLM_MODEL") or provider.default_model


@dataclass
class ChatResponse:
    text: str
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any]
    cache_hit: bool
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def stop_reason(self) -> str | None:
        if "choices" in self.raw:
            return self.raw["choices"][0].get("finish_reason")
        return self.raw.get("stop_reason")


def request_hash(payload: dict[str, Any], provider: str, model: str) -> str:
    """Stable hash over everything that could change the model's output."""
    blob = json.dumps(
        {"provider": provider, "model": model, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class LLMClient:
    """Chat client with a SQLite-backed response cache.

    The cache is keyed on the full request, so changing the prompt, the tool
    schemas, the model or the temperature all correctly miss the cache while an
    unchanged re-run is free.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        provider: Provider | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120.0,
        use_cache: bool = True,
    ) -> None:
        self.conn = conn
        self.provider = provider or resolve_provider()
        self.model = resolve_model(self.provider, model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self._client = httpx.Client(timeout=timeout)
        self.calls_made = 0
        self.cache_hits = 0

    # -- cache ------------------------------------------------------------

    def _cached(self, key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT response_json, latency_ms FROM llm_cache WHERE input_hash = ?",
            (key,),
        ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def _store(
        self, key: str, payload: dict[str, Any], response: dict[str, Any], ms: int
    ) -> None:
        usage = response.get("usage") or {}
        self.conn.execute(
            """INSERT OR REPLACE INTO llm_cache (
                   input_hash, created_at, provider, model, request_json,
                   response_json, prompt_tokens, completion_tokens, latency_ms)
               VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                self.provider.name,
                self.model,
                json.dumps(payload, sort_keys=True),
                json.dumps(response),
                usage.get("prompt_tokens") or usage.get("input_tokens"),
                usage.get("completion_tokens") or usage.get("output_tokens"),
                ms,
            ),
        )
        self.conn.commit()

    # -- request building -------------------------------------------------

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        if self.provider.dialect == "anthropic":
            system = "".join(
                m["content"] for m in messages if m["role"] == "system"
            )
            payload: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [m for m in messages if m["role"] != "system"],
            }
            if system:
                payload["system"] = system
            if tools:
                payload["tools"] = [
                    {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"],
                    }
                    for t in tools
                ]
                if tool_choice:
                    payload["tool_choice"] = (
                        {"type": "tool", "name": tool_choice}
                        if tool_choice != "auto"
                        else {"type": "auto"}
                    )
            return payload

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = (
                "auto"
                if tool_choice in (None, "auto")
                else {"type": "function", "function": {"name": tool_choice}}
            )
        return payload

    def _headers(self) -> dict[str, str]:
        key = os.environ[self.provider.env_var]
        if self.provider.dialect == "anthropic":
            return {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if self.provider.name == "openrouter":
            # OpenRouter attributes traffic with these; harmless elsewhere.
            headers["HTTP-Referer"] = "https://github.com/local/fda-recall-hazard"
            headers["X-Title"] = "fda-recall-hazard"
        return headers

    def _endpoint(self) -> str:
        suffix = "/messages" if self.provider.dialect == "anthropic" else "/chat/completions"
        return self.provider.base_url.rstrip("/") + suffix

    # -- parsing ----------------------------------------------------------

    @staticmethod
    def _parse(response: dict[str, Any], dialect: str) -> tuple[str, list[dict]]:
        if dialect == "anthropic":
            text_parts, calls = [], []
            for block in response.get("content", []):
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    calls.append(
                        {
                            "id": block["id"],
                            "name": block["name"],
                            "arguments": block.get("input") or {},
                        }
                    )
            return "".join(text_parts), calls

        message = (response.get("choices") or [{}])[0].get("message", {}) or {}
        calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            calls.append(
                {"id": call.get("id"), "name": fn.get("name"), "arguments": args or {}}
            )
        return message.get("content") or "", calls

    # -- public -----------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> ChatResponse:
        payload = self._build_payload(messages, tools, tool_choice)
        key = request_hash(payload, self.provider.name, self.model)

        if self.use_cache:
            hit = self._cached(key)
            if hit is not None:
                self.cache_hits += 1
                text, calls = self._parse(hit, self.provider.dialect)
                return ChatResponse(
                    text=text, tool_calls=calls, raw=hit, cache_hit=True, latency_ms=0
                )

        started = time.perf_counter()
        resp = self._client.post(
            self._endpoint(), headers=self._headers(), json=payload
        )
        ms = int((time.perf_counter() - started) * 1000)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"{self.provider.name} returned {resp.status_code}: {resp.text[:600]}"
            )
        body = resp.json()
        self.calls_made += 1
        if self.use_cache:
            self._store(key, payload, body, ms)

        text, calls = self._parse(body, self.provider.dialect)
        usage = body.get("usage") or {}
        return ChatResponse(
            text=text,
            tool_calls=calls,
            raw=body,
            cache_hit=False,
            latency_ms=ms,
            prompt_tokens=usage.get("prompt_tokens") or usage.get("input_tokens"),
            completion_tokens=usage.get("completion_tokens")
            or usage.get("output_tokens"),
        )

    def close(self) -> None:
        self._client.close()
