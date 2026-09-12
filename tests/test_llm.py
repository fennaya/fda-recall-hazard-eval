"""Tests for provider resolution, request shaping, response parsing and caching.

These run without a key and without network. They exist because the provider is
chosen at the last minute: whichever key gets populated, the request shape and
the cache behaviour have to be right on the first real run.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from fda_hazard.db import SCHEMA
from fda_hazard.llm import (
    PROVIDERS,
    LLMClient,
    NoProviderConfigured,
    available_providers,
    request_hash,
    resolve_model,
    resolve_provider,
)

ALL_KEYS = [p.env_var for p in PROVIDERS.values()] + ["LLM_PROVIDER", "LLM_MODEL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ALL_KEYS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    yield c
    c.close()


# -- provider resolution ---------------------------------------------------


def test_no_key_raises_a_helpful_error():
    with pytest.raises(NoProviderConfigured, match="GROQ_API_KEY"):
        resolve_provider()


@pytest.mark.parametrize("name", ["groq", "openrouter", "baseten", "anthropic"])
def test_each_provider_resolves_from_its_own_key(monkeypatch, name):
    monkeypatch.setenv(PROVIDERS[name].env_var, "test-key")
    assert resolve_provider().name == name


def test_detection_order_is_deterministic(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert resolve_provider().name == "openrouter"


def test_explicit_provider_overrides_detection(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("BASETEN_API_KEY", "k")
    assert resolve_provider("baseten").name == "baseten"


def test_llm_provider_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("BASETEN_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "baseten")
    assert resolve_provider().name == "baseten"


def test_selecting_a_provider_without_its_key_raises(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    with pytest.raises(NoProviderConfigured, match="BASETEN_API_KEY"):
        resolve_provider("baseten")


def test_unknown_provider_name_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_provider("mistral")


def test_available_providers_lists_only_configured(monkeypatch):
    assert available_providers() == []
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert available_providers() == ["groq"]


def test_model_resolution_precedence(monkeypatch):
    p = PROVIDERS["groq"]
    assert resolve_model(p) == p.default_model
    monkeypatch.setenv("LLM_MODEL", "from-env")
    assert resolve_model(p) == "from-env"
    assert resolve_model(p, "explicit") == "explicit"  # argument wins


# -- request shaping -------------------------------------------------------


def _client(conn, monkeypatch, name="groq"):
    monkeypatch.setenv(PROVIDERS[name].env_var, "test-key")
    return LLMClient(conn, provider=PROVIDERS[name], model="m")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_classification",
            "description": "submit",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "hi"},
]


def test_openai_payload_shape(conn, monkeypatch):
    c = _client(conn, monkeypatch, "groq")
    p = c._build_payload(MESSAGES, TOOLS, None)
    assert p["model"] == "m"
    assert p["messages"][0]["role"] == "system"  # system stays inline
    assert p["tools"] == TOOLS
    assert p["tool_choice"] == "auto"


def test_openai_forced_tool_choice(conn, monkeypatch):
    c = _client(conn, monkeypatch, "groq")
    p = c._build_payload(MESSAGES, TOOLS, "submit_classification")
    assert p["tool_choice"]["function"]["name"] == "submit_classification"


def test_anthropic_payload_hoists_system_and_rewrites_tools(conn, monkeypatch):
    c = _client(conn, monkeypatch, "anthropic")
    p = c._build_payload(MESSAGES, TOOLS, None)
    assert p["system"] == "sys"
    assert all(m["role"] != "system" for m in p["messages"])
    assert p["tools"][0]["name"] == "submit_classification"
    assert "input_schema" in p["tools"][0]


def test_endpoints_differ_by_dialect(conn, monkeypatch):
    assert _client(conn, monkeypatch, "groq")._endpoint().endswith("/chat/completions")
    assert _client(conn, monkeypatch, "anthropic")._endpoint().endswith("/messages")


def test_headers_use_the_right_auth_scheme(conn, monkeypatch):
    assert "Authorization" in _client(conn, monkeypatch, "groq")._headers()
    h = _client(conn, monkeypatch, "anthropic")._headers()
    assert "x-api-key" in h and "anthropic-version" in h


def test_openrouter_sends_attribution_headers(conn, monkeypatch):
    h = _client(conn, monkeypatch, "openrouter")._headers()
    assert "HTTP-Referer" in h and "X-Title" in h


# -- response parsing ------------------------------------------------------


def test_parse_openai_tool_call_with_string_arguments():
    body = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "submit_classification",
                                "arguments": '{"classification": "Class I"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    text, calls = LLMClient._parse(body, "openai")
    assert text == ""
    assert calls[0]["arguments"] == {"classification": "Class I"}


def test_parse_openai_handles_malformed_arguments():
    """A provider that emits truncated JSON must not crash the run."""
    body = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c", "function": {"name": "x", "arguments": "{bad"}}
                    ]
                }
            }
        ]
    }
    _, calls = LLMClient._parse(body, "openai")
    assert "_unparsed" in calls[0]["arguments"]


def test_parse_openai_plain_text():
    body = {"choices": [{"message": {"content": "Class II"}}]}
    text, calls = LLMClient._parse(body, "openai")
    assert text == "Class II" and calls == []


def test_parse_anthropic_blocks():
    body = {
        "content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "t1", "name": "find_precedents",
             "input": {"query": "x"}},
        ]
    }
    text, calls = LLMClient._parse(body, "anthropic")
    assert text == "thinking"
    assert calls[0]["name"] == "find_precedents"
    assert calls[0]["arguments"] == {"query": "x"}


# -- caching ---------------------------------------------------------------


def test_request_hash_is_stable_and_order_independent():
    a = request_hash({"b": 1, "a": 2}, "groq", "m")
    b = request_hash({"a": 2, "b": 1}, "groq", "m")
    assert a == b


@pytest.mark.parametrize(
    "provider,model,payload",
    [
        ("openrouter", "m", {"x": 1}),
        ("groq", "other", {"x": 1}),
        ("groq", "m", {"x": 2}),
    ],
)
def test_request_hash_changes_with_any_input(provider, model, payload):
    base = request_hash({"x": 1}, "groq", "m")
    assert request_hash(payload, provider, model) != base


def test_cache_hit_avoids_a_second_call(conn, monkeypatch):
    """A re-run of an unchanged evaluation must make zero API calls."""
    c = _client(conn, monkeypatch, "groq")
    payload = c._build_payload(MESSAGES, TOOLS, None)
    key = request_hash(payload, "groq", "m")
    body = {"choices": [{"message": {"content": "Class II"}}], "usage": {}}
    c._store(key, payload, body, 42)

    calls = {"n": 0}

    def explode(*a, **kw):  # a cache miss would hit this
        calls["n"] += 1
        raise AssertionError("network call made despite a warm cache")

    monkeypatch.setattr(c._client, "post", explode)
    resp = c.chat(MESSAGES, tools=TOOLS)
    assert resp.cache_hit is True
    assert resp.text == "Class II"
    assert calls["n"] == 0
    assert c.cache_hits == 1


def test_cache_stores_usage_counts(conn, monkeypatch):
    c = _client(conn, monkeypatch, "groq")
    body = {"choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5}}
    c._store("k1", {"a": 1}, body, 10)
    row = conn.execute("SELECT * FROM llm_cache WHERE input_hash='k1'").fetchone()
    assert row["prompt_tokens"] == 11 and row["completion_tokens"] == 5
    assert json.loads(row["response_json"]) == body


def test_changing_the_prompt_misses_the_cache(conn, monkeypatch):
    c = _client(conn, monkeypatch, "groq")
    p1 = c._build_payload(MESSAGES, TOOLS, None)
    c._store(request_hash(p1, "groq", "m"), p1, {"choices": []}, 1)
    p2 = c._build_payload(
        [{"role": "system", "content": "DIFFERENT"}, MESSAGES[1]], TOOLS, None
    )
    assert c._cached(request_hash(p2, "groq", "m")) is None


def test_no_cache_mode_does_not_read_the_cache(conn, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    c = LLMClient(conn, provider=PROVIDERS["groq"], model="m", use_cache=False)
    payload = c._build_payload(MESSAGES, None, None)
    c._store(request_hash(payload, "groq", "m"), payload,
             {"choices": [{"message": {"content": "cached"}}]}, 1)

    class FakeResp:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "fresh"}}], "usage": {}}

    monkeypatch.setattr(c._client, "post", lambda *a, **kw: FakeResp())
    assert c.chat(MESSAGES).text == "fresh"


def test_http_error_surfaces_the_provider_message(conn, monkeypatch):
    c = _client(conn, monkeypatch, "groq")

    class FakeResp:
        status_code = 401
        text = "invalid api key"

    monkeypatch.setattr(c._client, "post", lambda *a, **kw: FakeResp())
    with pytest.raises(RuntimeError, match="invalid api key"):
        c.chat(MESSAGES)
