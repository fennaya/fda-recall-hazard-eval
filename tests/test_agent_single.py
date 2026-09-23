"""Tests for the single-call architecture (Step 2/4).

No network, no API key. Covers: retrieval stays train-split-only (the same
leakage guarantee as the tool_loop path, since it's the same PrecedentIndex),
the precedent budget is respected, the output schema matches the tool_loop
path's AgentResult, and every result is tagged architecture='single_call' so
runs.record_run()'s no-mixing check has something to enforce.
"""

from __future__ import annotations

import json

import pytest

from fda_hazard.agent_single import (
    PRECEDENT_BUDGET_CHARS,
    SUBMIT_TOOL,
    build_case_context,
    classify_single,
)
from fda_hazard.llm import ChatResponse
from fda_hazard.retrieval import PrecedentIndex, build_index
from fda_hazard.splits import LeakageError, load_split

CUTOFF = "2025-01-01"


@pytest.fixture
def parts(corpus):
    index = build_index(corpus, CUTOFF)
    test = load_split(corpus, "test", CUTOFF)
    return corpus, index, test


class FakeClient:
    def __init__(self, args: dict | None = None, text: str = ""):
        self.args = args
        self.text = text
        self.calls = 0
        self.sent = []

    def chat(self, messages, tools=None, tool_choice=None):
        self.calls += 1
        self.sent.append({"messages": messages, "tools": tools, "tool_choice": tool_choice})
        if self.args is None:
            return ChatResponse(text=self.text, tool_calls=[], raw={}, cache_hit=False, latency_ms=5)
        return ChatResponse(
            text="",
            tool_calls=[{"id": "c1", "name": "submit_classification", "arguments": self.args}],
            raw={}, cache_hit=False, latency_ms=5,
        )


def submit(classification, confidence=0.8, reasoning="because", cited=None):
    return {
        "classification": classification, "confidence": confidence,
        "reasoning": reasoning, "precedents_cited": cited or [],
    }


# -- retrieval stays train-split-only ----------------------------------------


def test_single_call_context_never_includes_a_test_record(parts):
    conn, index, test = parts
    test_keys = {e.record_key for e in test}
    for ex in test:
        ctx = build_case_context(ex, index, conn, allow_network_lookups=False)
        for p in ctx.precedents_seen:
            assert p["record_key"] not in test_keys
        for key in test_keys:
            assert key not in ctx.precedents_text


def test_single_call_index_construction_shares_the_same_leakage_guarantee(corpus):
    """Same PrecedentIndex class, same constructor -- the existing leakage
    tests in test_retrieval.py already cover this path unchanged."""
    train = load_split(corpus, "train", CUTOFF)
    test = load_split(corpus, "test", CUTOFF)
    with pytest.raises(LeakageError, match="must be built from the train split"):
        PrecedentIndex(test, cutoff=CUTOFF)
    # Building it correctly and using it through build_case_context works.
    index = PrecedentIndex(train, cutoff=CUTOFF)
    ctx = build_case_context(test[0], index, corpus, allow_network_lookups=False)
    assert isinstance(ctx.precedents_text, str)


def test_single_call_excludes_the_case_being_classified(parts):
    conn, index, _ = parts
    train = load_split(conn, "train", CUTOFF)
    target = train[1]
    ctx = build_case_context(target, index, conn, allow_network_lookups=False)
    assert all(p["record_key"] != target.record_key for p in ctx.precedents_seen)


# -- precedent budget ----------------------------------------------------------


def test_precedent_text_respects_the_total_budget(parts):
    conn, index, test = parts
    ctx = build_case_context(test[0], index, conn, k=10, allow_network_lookups=False)
    assert ctx.precedent_chars_used <= PRECEDENT_BUDGET_CHARS
    assert len(ctx.precedents_text) <= PRECEDENT_BUDGET_CHARS + 50  # small formatting slack


def test_precedent_text_is_compact_not_raw_json(parts):
    """Compact summaries only: classification + a short excerpt, nothing else."""
    conn, index, test = parts
    ctx = build_case_context(test[0], index, conn, allow_network_lookups=False)
    assert "similarity" not in ctx.precedents_text
    assert "record_key" not in ctx.precedents_text  # no JSON keys leak into the text


def test_no_precedents_found_is_handled_not_an_error(parts):
    conn, index, test = parts
    ctx = build_case_context(test[0], index, conn, k=0, allow_network_lookups=False)
    assert ctx.precedents_text == "(none retrieved)"
    assert ctx.precedents_seen == []


# -- output schema parity with the tool_loop path ------------------------------


def test_submit_tool_schema_matches_the_tool_loop_path():
    from fda_hazard.agent import TOOLS

    tool_loop_submit = next(
        t for t in TOOLS if t["function"]["name"] == "submit_classification"
    )
    assert SUBMIT_TOOL["function"]["parameters"] == tool_loop_submit["function"]["parameters"]


def test_classify_single_result_has_the_same_shape_as_tool_loop(parts):
    conn, index, test = parts
    client = FakeClient(submit("Class I", 0.9, "sterility", cited=["A-1"]))
    result = classify_single(client, test[0], index, conn, allow_network_lookups=False)
    detail = result.to_detail(test[0].record_key)
    # Same keys record_run() reads regardless of architecture.
    assert set(detail) == {
        "record_key", "architecture", "confidence", "reasoning", "precedents",
        "tool_calls", "latency_ms", "cache_hit",
    }
    assert detail["architecture"] == "single_call"
    assert result.classification == "Class I"
    json.dumps(detail)  # must be serialisable, same as the tool_loop path


def test_classify_single_makes_exactly_one_model_call(parts):
    conn, index, test = parts
    client = FakeClient(submit("Class II"))
    classify_single(client, test[0], index, conn, allow_network_lookups=False)
    assert client.calls == 1


def test_classify_single_forces_the_submit_tool(parts):
    conn, index, test = parts
    client = FakeClient(submit("Class II"))
    classify_single(client, test[0], index, conn, allow_network_lookups=False)
    assert client.sent[0]["tool_choice"] == "submit_classification"
    assert client.sent[0]["tools"] == [SUBMIT_TOOL]


def test_classify_single_recovers_from_prose_answer(parts):
    conn, index, test = parts
    client = FakeClient(args=None, text="This is clearly a Class III labeling issue.")
    result = classify_single(client, test[0], index, conn, allow_network_lookups=False)
    assert result.classification == "Class III"
    assert result.architecture == "single_call"


def test_classify_single_never_raises_on_client_exception(parts):
    conn, index, test = parts

    class Explode:
        def chat(self, *a, **kw):
            raise RuntimeError("groq returned 500: boom")

    result = classify_single(Explode(), test[0], index, conn, allow_network_lookups=False)
    assert result.classification == "Class II"
    assert result.confidence == 0.0
    assert result.error is not None
    assert result.architecture == "single_call"


def test_classify_single_precedents_seen_recorded_for_the_dashboard(parts):
    conn, index, test = parts
    client = FakeClient(submit("Class II", cited=["A-1"]))
    result = classify_single(client, test[0], index, conn, allow_network_lookups=False)
    assert isinstance(result.precedents_seen, list)
