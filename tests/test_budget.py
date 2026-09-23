"""Tests for the hard token budget (Step 5).

No network. Covers: cumulative counting, the cap actually firing on a fake
low ceiling, cache hits never counting against it, and that BudgetExceeded
escapes classify()'s broad `except Exception` -- the whole point of it.
"""

from __future__ import annotations

import pytest

from fda_hazard.budget import BudgetExceeded, TokenBudget
from fda_hazard.llm import PROVIDERS, ChatResponse, LLMClient


# -- TokenBudget itself -------------------------------------------------------


def test_budget_accumulates_across_calls():
    b = TokenBudget(ceiling=None)
    b.add_call(100, 50)
    b.add_call(200, 25)
    assert b.spent == 375
    assert b.calls == 2


def test_budget_with_no_ceiling_never_raises():
    b = TokenBudget(ceiling=None)
    for _ in range(1000):
        b.add_call(10_000, 10_000)
    assert b.spent == 20_000_000  # just doesn't raise


def test_budget_fires_exactly_at_the_ceiling():
    """The test Step 5 asks for: the cap actually fires, using a fake low ceiling."""
    b = TokenBudget(ceiling=1000)
    b.add_call(400, 100)  # 500, under
    with pytest.raises(BudgetExceeded) as exc_info:
        b.add_call(400, 200)  # 1100, over
    assert exc_info.value.spent == 1100
    assert exc_info.value.ceiling == 1000


def test_budget_fires_on_exact_equality_not_just_overshoot():
    b = TokenBudget(ceiling=500)
    with pytest.raises(BudgetExceeded):
        b.add_call(500, 0)  # exactly at the ceiling


def test_budget_exceeded_is_not_a_plain_exception():
    """Must escape `except Exception` in both agents' resilience catch-alls,
    or the run would keep spending past the ceiling instead of stopping."""
    assert not issubclass(BudgetExceeded, Exception)
    assert issubclass(BudgetExceeded, BaseException)
    try:
        raise BudgetExceeded(spent=10, ceiling=5)
    except Exception:
        pytest.fail("BudgetExceeded must not be caught by `except Exception`")
    except BudgetExceeded:
        pass  # correct: only a specific catch sees it


def test_budget_logs_running_total_per_case(tmp_path):
    log = tmp_path / "budget.log"
    b = TokenBudget(ceiling=1000, log_path=log)
    b.add_call(100, 50)
    b.log_case(1, 10)
    b.add_call(200, 50)
    b.log_case(2, 10)
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    assert "case 1/10" in lines[0] and "tokens_spent=150" in lines[0]
    assert "case 2/10" in lines[1] and "tokens_spent=400" in lines[1]
    assert "40.0% of ceiling" in lines[1]


def test_budget_log_case_without_a_ceiling_omits_percentage(tmp_path):
    log = tmp_path / "budget.log"
    b = TokenBudget(ceiling=None, log_path=log)
    b.add_call(100, 0)
    b.log_case(1, 5)
    assert "%" not in log.read_text()


def test_budget_with_no_log_path_does_not_raise():
    b = TokenBudget(ceiling=None, log_path=None)
    b.add_call(10, 10)
    b.log_case(1, 1)  # must not raise, no file to write


# -- wired into LLMClient -----------------------------------------------------


def _client(conn, monkeypatch, budget=None):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return LLMClient(conn, provider=PROVIDERS["groq"], model="m", budget=budget)


def test_cache_hit_never_counts_against_the_budget(conn_db, monkeypatch):
    budget = TokenBudget(ceiling=100)
    c = _client(conn_db, monkeypatch, budget)
    payload = c._build_payload([{"role": "user", "content": "hi"}], None, None)
    body = {"choices": [{"message": {"content": "x"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 500}}
    from fda_hazard.llm import request_hash
    c._store(request_hash(payload, "groq", "m"), payload, body, 1)

    # A cached response of 1000 tokens must not trip a ceiling of 100.
    resp = c.chat([{"role": "user", "content": "hi"}])
    assert resp.cache_hit is True
    assert budget.spent == 0


def test_fresh_call_over_budget_raises_but_still_gets_cached(conn_db, monkeypatch):
    budget = TokenBudget(ceiling=100)
    c = _client(conn_db, monkeypatch, budget)

    class Big:
        status_code = 200
        headers = {}
        def json(self):
            return {"choices": [{"message": {"content": "x"}}],
                    "usage": {"prompt_tokens": 80, "completion_tokens": 50}}

    monkeypatch.setattr(c._client, "post", lambda *a, **kw: Big())
    with pytest.raises(BudgetExceeded):
        c.chat([{"role": "user", "content": "hi"}])
    assert budget.spent == 130
    # The response was still cached before the budget check ran -- a resume
    # after raising the ceiling reuses it for free rather than paying again.
    assert c.cache_hits == 0
    n = conn_db.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
    assert n == 1


@pytest.fixture
def conn_db():
    import sqlite3
    from fda_hazard.db import SCHEMA
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    yield c
    c.close()


# -- escapes the agents' resilience catch-alls --------------------------------


def test_budget_exceeded_escapes_hazard_agent_classify(corpus):
    from fda_hazard.agent import HazardAgent
    from fda_hazard.retrieval import build_index
    from fda_hazard.splits import load_split

    class BudgetBustingClient:
        provider = PROVIDERS["groq"]
        model = "fake"

        def chat(self, *a, **kw):
            raise BudgetExceeded(spent=999, ceiling=100)

        def close(self):
            pass

    index = build_index(corpus, "2025-01-01")
    test = load_split(corpus, "test", "2025-01-01")
    agent = HazardAgent(BudgetBustingClient(), index, corpus, allow_network_lookups=False)
    with pytest.raises(BudgetExceeded):
        agent.classify(test[0])  # must NOT come back as a scored fallback


def test_budget_exceeded_escapes_classify_single(corpus):
    from fda_hazard.agent_single import classify_single
    from fda_hazard.retrieval import build_index
    from fda_hazard.splits import load_split

    class BudgetBustingClient:
        def chat(self, *a, **kw):
            raise BudgetExceeded(spent=999, ceiling=100)

    index = build_index(corpus, "2025-01-01")
    test = load_split(corpus, "test", "2025-01-01")
    with pytest.raises(BudgetExceeded):
        classify_single(
            BudgetBustingClient(), test[0], index, corpus, allow_network_lookups=False
        )


def test_budget_log_creates_missing_parent_directory(tmp_path):
    """Regression: a real 20-case free-tier verification crashed and lost its
    result mid-run because log_case() didn't create its log directory (and,
    separately, because a /tmp path was passed to a native Windows process --
    this test covers the part budget.py itself is responsible for: never lose
    a result over an absent directory)."""
    log = tmp_path / "does" / "not" / "exist" / "budget.log"
    assert not log.parent.exists()
    b = TokenBudget(ceiling=None, log_path=log)
    b.add_call(10, 5)
    b.log_case(1, 1)  # must not raise
    assert log.exists()
    assert "tokens_spent=15" in log.read_text()
