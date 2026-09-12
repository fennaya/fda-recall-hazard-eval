"""End-to-end agent loop against a scripted fake provider.

No network and no API key: the point is to prove the tool loop, the
precedent plumbing and the structured output all work, independently of which
provider is eventually configured.
"""

from __future__ import annotations

import json

import pytest

from fda_hazard.agent import HazardAgent
from fda_hazard.llm import PROVIDERS, ChatResponse
from fda_hazard.retrieval import build_index
from fda_hazard.splits import load_split

CUTOFF = "2025-01-01"


class FakeClient:
    """Replays a scripted list of responses and records what it was sent."""

    def __init__(self, script, dialect="openai"):
        self.script = list(script)
        self.provider = PROVIDERS["groq"] if dialect == "openai" else PROVIDERS["anthropic"]
        self.model = "fake-model"
        self.sent: list[list[dict]] = []
        self.calls_made = 0
        self.cache_hits = 0

    def chat(self, messages, tools=None, tool_choice=None):
        self.sent.append([dict(m) for m in messages])
        self.calls_made += 1
        if not self.script:
            raise AssertionError("fake client ran out of scripted responses")
        text, calls = self.script.pop(0)
        return ChatResponse(
            text=text, tool_calls=calls, raw={}, cache_hit=False, latency_ms=7
        )

    def close(self):
        pass


def submit(classification, confidence=0.8, reasoning="because", cited=None):
    return (
        "",
        [
            {
                "id": "c1",
                "name": "submit_classification",
                "arguments": {
                    "classification": classification,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "precedents_cited": cited or [],
                },
            }
        ],
    )


def call_tool(name, args):
    return ("", [{"id": "t1", "name": name, "arguments": args}])


@pytest.fixture
def agent_parts(corpus):
    index = build_index(corpus, CUTOFF)
    test = load_split(corpus, "test", CUTOFF)
    return corpus, index, test


def test_direct_submission(agent_parts):
    conn, index, test = agent_parts
    client = FakeClient([submit("Class I", 0.91, "sterility failure")])
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    result = agent.classify(test[0])
    assert result.classification == "Class I"
    assert result.confidence == 0.91
    assert result.error is None
    assert client.calls_made == 1


def test_precedent_tool_then_submit(agent_parts):
    conn, index, test = agent_parts
    client = FakeClient(
        [
            call_tool("find_precedents", {"query": "subpotent tablets", "k": 3}),
            submit("Class II", 0.7, "matches precedent", cited=["A-2"]),
        ]
    )
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    result = agent.classify(test[0])

    assert result.classification == "Class II"
    assert result.precedents_cited == ["A-2"]
    assert result.precedents_seen, "precedents should be recorded for the audit trail"
    assert result.tool_calls[0]["tool"] == "find_precedents"
    # The tool result must actually reach the model.
    assert any(m.get("role") == "tool" for m in client.sent[-1])


def test_precedents_reaching_the_model_are_train_only(agent_parts):
    conn, index, test = agent_parts
    test_keys = {e.record_key for e in test}
    client = FakeClient(
        [
            call_tool("find_precedents", {"query": "subpotent out of specification"}),
            submit("Class II"),
        ]
    )
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    result = agent.classify(test[0])

    for p in result.precedents_seen:
        assert p["record_key"] not in test_keys
    payload = json.dumps(client.sent[-1])
    for key in test_keys:
        assert f'"{key}"' not in payload


def test_agent_never_receives_its_own_record_as_precedent(agent_parts):
    """Even if the example somehow sat in the index, it must be excluded."""
    conn, index, _ = agent_parts
    train = load_split(conn, "train", CUTOFF)
    target = train[1]
    client = FakeClient(
        [call_tool("find_precedents", {"query": target.text()}), submit("Class II")]
    )
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    result = agent.classify(target)
    assert all(p["record_key"] != target.record_key for p in result.precedents_seen)


def test_drug_lookup_tool_without_network_is_not_fatal(agent_parts):
    conn, index, test = agent_parts
    client = FakeClient(
        [
            call_tool("lookup_drug_context", {"product_description": "Widget Tablets"}),
            submit("Class III", 0.4),
        ]
    )
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    result = agent.classify(test[0])
    assert result.classification == "Class III"
    assert result.tool_calls[0]["tool"] == "lookup_drug_context"


def test_unknown_tool_does_not_crash_the_loop(agent_parts):
    conn, index, test = agent_parts
    client = FakeClient([call_tool("nonexistent_tool", {}), submit("Class II")])
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    assert agent.classify(test[0]).classification == "Class II"


def test_prose_answer_is_recovered(agent_parts):
    conn, index, test = agent_parts
    client = FakeClient([("This is clearly a Class III labeling issue.", [])])
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    result = agent.classify(test[0])
    assert result.classification == "Class III"


def test_turn_limit_yields_a_scored_prediction_not_a_crash(agent_parts):
    """A model that loops forever must still produce a label, flagged as an
    error, rather than taking down the whole evaluation."""
    conn, index, test = agent_parts
    client = FakeClient(
        [call_tool("find_precedents", {"query": "x"}) for _ in range(6)]
    )
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    result = agent.classify(test[0])
    assert result.classification in ("Class I", "Class II", "Class III")
    assert result.error == "no_submission"


def test_anthropic_dialect_message_shape(agent_parts):
    conn, index, test = agent_parts
    client = FakeClient(
        [call_tool("find_precedents", {"query": "x"}), submit("Class II")],
        dialect="anthropic",
    )
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    agent.classify(test[0])
    last = client.sent[-1]
    assert last[-1]["role"] == "user"
    assert last[-1]["content"][0]["type"] == "tool_result"


def test_result_detail_is_json_serialisable(agent_parts):
    conn, index, test = agent_parts
    client = FakeClient(
        [call_tool("find_precedents", {"query": "x"}), submit("Class I", cited=["A-1"])]
    )
    agent = HazardAgent(client, index, conn, allow_network_lookups=False)
    detail = agent.classify(test[0]).to_detail("R-1")
    json.dumps(detail)  # must not raise
    assert detail["record_key"] == "R-1"
    assert detail["precedents"]["cited"] == ["A-1"]
