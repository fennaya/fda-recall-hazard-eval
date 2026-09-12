"""Run the agent over the held-out test split and record the score.

Scoring happens in metrics.py. This module only collects predictions and hands
them over; it never computes a metric itself, and neither does the model.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sys
import time

from .agent import PROMPT_VERSION, AgentResult, HazardAgent, prompt_sha
from .db import connect
from .llm import LLMClient, NoProviderConfigured, resolve_model, resolve_provider
from .metrics import format_report, score
from .retrieval import build_index
from .runs import record_run
from .splits import SPLIT_CUTOFF, Example, load_split, verify_split


def classify_all(
    agent: HazardAgent,
    examples: list[Example],
    workers: int = 4,
    progress_every: int = 25,
) -> list[AgentResult]:
    """Classify every example, preserving input order."""
    results: list[AgentResult | None] = [None] * len(examples)
    done = 0
    started = time.perf_counter()

    def work(i: int) -> tuple[int, AgentResult]:
        return i, agent.classify(examples[i])

    if workers <= 1:
        for i, ex in enumerate(examples):
            results[i] = agent.classify(ex)
            done += 1
            if done % progress_every == 0:
                _progress(done, len(examples), started)
    else:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for fut in futures.as_completed(
                [pool.submit(work, i) for i in range(len(examples))]
            ):
                i, res = fut.result()
                results[i] = res
                done += 1
                if done % progress_every == 0:
                    _progress(done, len(examples), started)

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
    ap.add_argument("--limit", type=int, default=None, help="score only the first N test cases")
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

    verify_split(conn, args.cutoff)
    index = build_index(conn, args.cutoff)
    test = load_split(conn, "test", args.cutoff)
    if args.limit:
        test = test[: args.limit]
    print(f"precedent index: {len(index):,} train recalls")
    print(f"scoring {len(test):,} held-out test recalls\n")

    client = LLMClient(
        conn,
        provider=provider,
        model=model,
        temperature=args.temperature,
        use_cache=not args.no_cache,
    )
    agent = HazardAgent(
        client, index, conn, default_k=args.k,
        allow_network_lookups=not args.no_lookups,
    )

    started = time.perf_counter()
    try:
        results = classify_all(agent, test, workers=args.workers)
    finally:
        agent.close()
        client.close()
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
            "index_size": len(index),
        },
        duration_s=duration,
        details=details,
        notes=args.notes,
    )

    print(format_report(metrics, f"agent ({provider.name}/{model})"))
    print(
        f"\napi calls={client.calls_made}  cache hits={client.cache_hits}  "
        f"wall={duration / 60:.1f}m"
    )
    print(f"saved as run {uid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
