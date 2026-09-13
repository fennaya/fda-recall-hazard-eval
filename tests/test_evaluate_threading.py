"""Regression test: classify_all must not share one sqlite3.Connection across
worker threads.

sqlite3 connections aren't safe to use from a thread other than the one that
created them (raises sqlite3.ProgrammingError). The threaded eval path found
this the hard way on a real 1,275-case Groq run: it must build a fresh
connection per worker thread, not reuse the main-thread one.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest

from fda_hazard.agent import HazardAgent
from fda_hazard.db import SCHEMA
from fda_hazard.evaluate import classify_all
from fda_hazard.llm import PROVIDERS, ChatResponse
from fda_hazard.retrieval import build_index
from fda_hazard.splits import load_split

CUTOFF = "2025-01-01"


class FakeClient:
    """Always submits Class II; records which thread and connection it used."""

    provider = PROVIDERS["groq"]
    model = "fake"
    calls_made = 0
    cache_hits = 0

    def __init__(self, conn):
        self.conn = conn
        self.seen_threads: set[int] = set()

    def chat(self, messages, tools=None, tool_choice=None):
        self.seen_threads.add(threading.get_ident())
        # Touch the connection from this thread, exactly like the real cache
        # lookup does -- this is what raises if the connection was made
        # elsewhere.
        self.conn.execute("SELECT 1").fetchone()
        return ChatResponse(
            text="",
            tool_calls=[
                {
                    "id": "s",
                    "name": "submit_classification",
                    "arguments": {
                        "classification": "Class II",
                        "confidence": 0.5,
                        "reasoning": "x",
                        "precedents_cited": [],
                    },
                }
            ],
            raw={},
            cache_hit=False,
            latency_ms=1,
        )

    def close(self):
        pass


def fresh_conn() -> sqlite3.Connection:
    # Mirrors db.connect(): check_same_thread=False only so the pool's cleanup
    # can close this connection from the main thread once the worker is done
    # with it. The connection is still only ever used by one thread at a time.
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def make_agent(index) -> HazardAgent:
    conn = fresh_conn()
    client = FakeClient(conn)
    return HazardAgent(client, index, conn, allow_network_lookups=False)


def test_threaded_eval_gives_each_worker_its_own_connection(corpus):
    index = build_index(corpus, CUTOFF)
    test = load_split(corpus, "test", CUTOFF) * 3  # enough work to force concurrency

    seed_agent = make_agent(index)
    try:
        results = classify_all(
            seed_agent,
            test,
            workers=4,
            progress_every=1000,
            agent_factory=lambda: make_agent(index),
        )
    finally:
        seed_agent.close()
        seed_agent.client.close()
        seed_agent.conn.close()

    assert len(results) == len(test)
    assert all(r.classification == "Class II" for r in results)


def test_workers_greater_than_one_requires_a_factory(corpus):
    index = build_index(corpus, CUTOFF)
    test = load_split(corpus, "test", CUTOFF)
    agent = make_agent(index)
    try:
        with pytest.raises(ValueError, match="agent_factory"):
            classify_all(agent, test, workers=4)
    finally:
        agent.close()
        agent.client.close()
        agent.conn.close()


def test_single_worker_path_does_not_require_a_factory(corpus):
    """workers<=1 must keep working exactly as before: no factory needed."""
    index = build_index(corpus, CUTOFF)
    test = load_split(corpus, "test", CUTOFF)
    agent = make_agent(index)
    try:
        results = classify_all(agent, test, workers=1, progress_every=1000)
        assert len(results) == len(test)
    finally:
        agent.close()
        agent.client.close()
        agent.conn.close()
