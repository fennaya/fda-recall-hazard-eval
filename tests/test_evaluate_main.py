"""End-to-end test of evaluate.main() with a fully stubbed LLM.

This exists because the real bug it catches doesn't show up in any unit test:
main() built its agents through a factory closure but the final report still
referenced a `client` variable that no longer existed, so every real run would
have crashed on the print statement right after finishing all 1,275 cases.
Only calling main() itself, start to finish, catches that.
"""

from __future__ import annotations

import os

import pytest

from fda_hazard import evaluate
from fda_hazard.llm import ChatResponse


@pytest.fixture(autouse=True)
def fake_groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def patched_db(monkeypatch, corpus, tmp_path):
    """Point every connect() call at the same in-memory-backed test corpus."""
    monkeypatch.setattr(evaluate, "connect", lambda *a, **kw: corpus)
    return corpus


def _install_stub_chat(monkeypatch):
    def fake_chat(self, messages, tools=None, tool_choice=None):
        return ChatResponse(
            text="",
            tool_calls=[
                {
                    "id": "s",
                    "name": "submit_classification",
                    "arguments": {
                        "classification": "Class II",
                        "confidence": 0.6,
                        "reasoning": "stub",
                        "precedents_cited": [],
                    },
                }
            ],
            raw={},
            cache_hit=False,
            latency_ms=1,
        )

    monkeypatch.setattr("fda_hazard.llm.LLMClient.chat", fake_chat, raising=True)


def test_main_runs_start_to_finish_without_crashing(patched_db, monkeypatch, capsys):
    """This must not raise -- it did, on the final print, before the fix."""
    _install_stub_chat(monkeypatch)
    rc = evaluate.main(["--workers", "1", "--no-lookups"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "saved as run" in out
    assert "api calls=" in out


def test_main_records_an_agent_run(patched_db, monkeypatch):
    _install_stub_chat(monkeypatch)
    evaluate.main(["--workers", "1", "--no-lookups"])
    row = patched_db.execute(
        "SELECT * FROM eval_runs WHERE system='agent' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert row["provider"] == "groq"


def test_main_with_sample_scores_fewer_than_the_full_split(patched_db, monkeypatch, capsys):
    _install_stub_chat(monkeypatch)
    rc = evaluate.main(["--workers", "1", "--no-lookups", "--sample", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NOT the full held-out set" in out


def test_main_rejects_limit_and_sample_together(patched_db, monkeypatch, capsys):
    rc = evaluate.main(["--limit", "1", "--sample", "1"])
    assert rc == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_main_with_no_provider_key_returns_error(patched_db, monkeypatch, capsys):
    for var in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "BASETEN_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    rc = evaluate.main([])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
