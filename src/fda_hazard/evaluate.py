"""Run the agent over the held-out test split and record the score.

Scoring happens in metrics.py. This module only collects predictions and hands
them over; it never computes a metric itself, and neither does the model.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .agent import PROMPT_VERSION, AgentResult, HazardAgent, prompt_sha
from .db import connect
from .llm import LLMClient, NoProviderConfigured, resolve_model, resolve_provider
from .metrics import format_report, score
from .retrieval import build_index
from .runs import record_run
from .splits import (
    SPLIT_CUTOFF,
    Example,
    class_counts,
    load_split,
    stratified_sample,
    verify_split,
)


@dataclass
class ClientStats:
    calls_made: int = 0
    cache_hits: int = 0


def classify_all(
    agent: HazardAgent,
    examples: list[Example],
    workers: int = 4,
    progress_every: int = 25,
    agent_factory: Callable[[], HazardAgent] | None = None,
    stats: ClientStats | None = None,
) -> list[AgentResult]:
    """Classify every example, preserving input order.

    A sqlite3.Connection cannot be shared across threads, and both LLMClient
    (response cache) and HazardAgent (drug-context cache) hold one. So the
    threaded path never reuses `agent` across workers: each worker thread lazily
    builds its own agent via `agent_factory` on first use, backed by its own
    connection, and those connections are closed when the pool is done. The
    single-threaded path (workers <= 1) just uses `agent` directly.
    """
    if workers > 1 and agent_factory is None:
        raise ValueError("agent_factory is required when workers > 1")

    results: list[AgentResult | None] = [None] * len(examples)
    done = 0
    started = time.perf_counter()

    if workers <= 1:
        for i, ex in enumerate(examples):
            results[i] = agent.classify(ex)
            done += 1
            if done % progress_every == 0:
                _progress(done, len(examples), started)
        _progress(done, len(examples), started)
        print()
        if stats is not None:
            stats.calls_made = agent.client.calls_made
            stats.cache_hits = agent.client.cache_hits
        return [r for r in results if r is not None]

    local = threading.local()
    made: list[HazardAgent] = []
    lock = threading.Lock()

    def worker_agent() -> HazardAgent:
        if not hasattr(local, "agent"):
            local.agent = agent_factory()
            with lock:
                made.append(local.agent)
        return local.agent

    def work(i: int) -> tuple[int, AgentResult]:
        return i, worker_agent().classify(examples[i])

    try:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in futures.as_completed(
                [pool.submit(work, i) for i in range(len(examples))]
            ):
                i, res = fut.result()
                results[i] = res
                done += 1
                if done % progress_every == 0:
                    _progress(done, len(examples), started)
    finally:
        for a in made:
            if stats is not None:
                stats.calls_made += a.client.calls_made
                stats.cache_hits += a.client.cache_hits
            a.close()
            a.client.close()
            a.conn.close()

    _progress(done, len(examples), started)
    print()
    return [r for r in results if r is not None]


def _progress(done: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    print(
        f"\r  {done}/{total} ({done / total:.0%})  {rate:.1f}/s  eta {eta / 60:.1f}m",
        end="",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default=None, help="groq|openrouter|baseten|anthropic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cutoff", default=SPLIT_CUTOFF)
    ap.add_argument("--limit", type=int, default=None,
                     help="score only the first N test cases by date (unbiased prefix "
                     "cut; useful for a quick sanity check, not a fair score)")
    ap.add_argument("--sample", type=int, default=None,
                     help="score a class-proportional random subsample of size N "
                     "instead of the full test split (mutually exclusive with --limit)")
    ap.add_argument("--sample-seed", type=int, default=20250101)
    ap.add_argument("--k", type=int, default=5, help="precedents retrieved per call")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-lookups", action="store_true", help="disable openFDA network lookups")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args(argv)

    conn = connect()
    try:
        provider = resolve_provider(args.provider)
    except NoProviderConfigured as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model = resolve_model(provider, args.model)
    print(f"provider={provider.name} model={model} prompt={PROMPT_VERSION}/{prompt_sha()}")

    if args.limit and args.sample:
        print("error: --limit and --sample are mutually exclusive", file=sys.stderr)
        return 2

    verify_split(conn, args.cutoff)
    index = build_index(conn, args.cutoff)
    test = load_split(conn, "test", args.cutoff)
    full_test_size = len(test)
    if args.limit:
        test = test[: args.limit]
        print(f"[note] scoring the first {len(test):,} of {full_test_size:,} test "
              f"cases by date, NOT the full held-out set")
    elif args.sample:
        test = stratified_sample(test, args.sample, seed=args.sample_seed)
        counts = class_counts(test)
        print(f"[note] scoring a class-proportional sample of {len(test):,} of "
              f"{full_test_size:,} test cases (seed={args.sample_seed}), NOT the "
              f"full held-out set: {counts}")
    print(f"precedent index: {len(index):,} train recalls")
    print(f"scoring {len(test):,} held-out test recalls\n")

    def make_agent(shared_conn) -> HazardAgent:
        c = LLMClient(
            shared_conn,
            provider=provider,
            model=model,
            temperature=args.temperature,
            use_cache=not args.no_cache,
        )
        return HazardAgent(
            c, index, shared_conn, default_k=args.k,
            allow_network_lookups=not args.no_lookups,
        )

    # The main-thread agent/connection serve the workers<=1 path directly.
    # Concurrent workers each get their own connection (see classify_all):
    # sqlite3 connections are not thread-safe to share, and both the response
    # cache and the drug-context cache write through them.
    agent = make_agent(conn)
    stats = ClientStats()

    started = time.perf_counter()
    try:
        results = classify_all(
            agent, test, workers=args.workers,
            agent_factory=(lambda: make_agent(connect())) if args.workers > 1 else None,
            stats=stats,
        )
    finally:
        agent.close()
        agent.client.close()
    duration = time.perf_counter() - started

    preds = [r.classification for r in results]
    metrics = score([e.classification for e in test], preds)
    details = [r.to_detail(e.record_key) for e, r in zip(test, results, strict=True)]

    errors = [r for r in results if r.error]
    if errors:
        print(f"[warn] {len(errors)} example(s) had agent-side problems "
              f"(e.g. {errors[0].error}); they are scored as predicted, not dropped\n")

    uid = record_run(
        conn,
        system="agent",
        metrics=metrics,
        examples=test,
        predictions=preds,
        cutoff=args.cutoff,
        prompt_version=PROMPT_VERSION,
        prompt_sha=prompt_sha(),
        model=model,
        provider=provider.name,
        config={
            "k": args.k,
            "temperature": args.temperature,
            "workers": args.workers,
            "lookups_enabled": not args.no_lookups,
            "limit": args.limit,
            "sample": args.sample,
            "sample_seed": args.sample_seed if args.sample else None,
            "full_test_size": full_test_size,
            "index_size": len(index),
        },
        duration_s=duration,
        details=details,
        notes=args.notes,
    )

    print(format_report(metrics, f"agent ({provider.name}/{model})"))
    print(
        f"\napi calls={stats.calls_made}  cache hits={stats.cache_hits}  "
        f"wall={duration / 60:.1f}m"
    )
    if args.sample or args.limit:
        print(f"[note] this run scored {len(test):,} of {full_test_size:,} test "
              f"cases, not the full held-out set")
    print(f"saved as run {uid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
