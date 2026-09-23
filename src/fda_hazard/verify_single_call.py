"""Step 4: verify the single-call path on a small free-tier sample.

Deliberately separate from evaluate.py (which this task's constraints say not
to touch). Records its result as its own architecture='single_call' run via
runs.record_run() -- small n, clearly not the full held-out set -- and prints
the mean/median token cost plus a projection to the full 1,275 cases.

Usage: uv run -m fda_hazard.verify_single_call [--sample 20] [--budget N]
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time

from .agent_single import PRECEDENT_BUDGET_CHARS, SINGLE_CALL_PROMPT_VERSION, classify_single
from .budget import BudgetExceeded, TokenBudget
from .db import connect
from .llm import LLMClient, NoProviderConfigured, resolve_model, resolve_provider
from .metrics import format_report, score
from .retrieval import build_index
from .runs import record_run
from .splits import SPLIT_CUTOFF, class_counts, load_split, stratified_sample, verify_split


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=20, help="20-50 per Step 4")
    ap.add_argument("--sample-seed", type=int, default=20250101)
    ap.add_argument("--cutoff", default=SPLIT_CUTOFF)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--budget", type=int, default=None,
                     help="token ceiling (Step 5); default: none for this small check")
    ap.add_argument("--budget-log", default=None)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args(argv)

    if not (20 <= args.sample <= 50):
        print(f"error: --sample must be 20-50 per Step 4, got {args.sample}", file=sys.stderr)
        return 2

    conn = connect()
    try:
        provider = resolve_provider(args.provider)
    except NoProviderConfigured as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    model = resolve_model(provider, args.model)

    verify_split(conn, args.cutoff)  # leakage assertions, unchanged
    index = build_index(conn, args.cutoff)
    full_test = load_split(conn, "test", args.cutoff)
    sample = stratified_sample(full_test, args.sample, seed=args.sample_seed)
    counts = class_counts(sample)
    print(f"provider={provider.name} model={model} architecture=single_call "
          f"prompt={SINGLE_CALL_PROMPT_VERSION}")
    print(f"verifying {len(sample)} of {len(full_test):,} test cases "
          f"(seed={args.sample_seed}): {counts}")
    print(f"precedent budget: {PRECEDENT_BUDGET_CHARS} chars\n")

    budget = TokenBudget(ceiling=args.budget, log_path=args.budget_log)
    client = LLMClient(conn, provider=provider, model=model, budget=budget)

    results, per_case_tokens, exceeded = [], [], False
    started = time.perf_counter()
    for i, ex in enumerate(sample, 1):
        before = budget.spent
        try:
            r = classify_single(client, ex, index, conn, k=args.k, allow_network_lookups=False)
        except BudgetExceeded as exc:
            print(f"\n[budget] ceiling hit at case {i}/{len(sample)}: {exc}")
            exceeded = True
            break
        results.append(r)
        per_case_tokens.append(budget.spent - before)
        budget.log_case(i, len(sample))
        print(f"\r  {i}/{len(sample)}  spent={budget.spent:,} tok  calls={budget.calls}",
              end="", flush=True)
    duration = time.perf_counter() - started
    print()

    if not results:
        print("no results to score (budget exceeded before the first case completed)")
        return 1

    scored_examples = sample[: len(results)]
    preds = [r.classification for r in results]
    metrics = score([e.classification for e in scored_examples], preds)
    details = [r.to_detail(e.record_key) for e, r in zip(scored_examples, results, strict=True)]

    print(format_report(metrics, f"agent single_call ({provider.name}/{model}), n={len(results)}"))
    print()

    if per_case_tokens:
        mean_tok, med_tok = st.mean(per_case_tokens), st.median(per_case_tokens)
        print(f"tokens per case: mean={mean_tok:,.0f}  median={med_tok:,.0f}  "
              f"min={min(per_case_tokens):,}  max={max(per_case_tokens):,}")
        full_n = len(full_test)
        proj_tokens = mean_tok * full_n
        print(f"\nprojected full run ({full_n:,} cases): "
              f"{mean_tok:,.0f} tok/case x {full_n:,} = {proj_tokens:,.0f} tokens total")
        # unbiased/pareto rates from the earlier live probe: $2.5/M prompt, $7.5/M completion.
        # Applied to the blended mean since this script doesn't split prompt/completion
        # per case; report as a single blended-rate estimate, not a precise split.
        blended_rate_per_m = 5.0  # midpoint of $2.5-$7.5/M, stated as an estimate
        print(f"estimated cost at ~${blended_rate_per_m:.1f}/M tokens (blended, e.g. "
              f"unbiased/pareto): ${proj_tokens * blended_rate_per_m / 1_000_000:,.2f}")
        cases_per_sec = len(results) / duration if duration > 0 else 0
        if cases_per_sec > 0:
            proj_wall_s = full_n / cases_per_sec
            print(f"projected wall-clock at this sample's rate "
                  f"({cases_per_sec:.2f} cases/s): {proj_wall_s / 60:.1f} min "
                  f"({proj_wall_s / 3600:.2f} h)")

    uid = record_run(
        conn, system="agent", metrics=metrics, examples=scored_examples, predictions=preds,
        cutoff=args.cutoff, architecture="single_call",
        prompt_version=SINGLE_CALL_PROMPT_VERSION, model=model, provider=provider.name,
        config={"k": args.k, "sample": args.sample, "sample_seed": args.sample_seed,
                "precedent_budget_chars": PRECEDENT_BUDGET_CHARS,
                "token_budget_ceiling": args.budget, "budget_exceeded": exceeded,
                "full_test_size": len(full_test)},
        duration_s=duration, details=details,
        notes=(f"Step 4 verification: single_call architecture, n={len(results)} of "
               f"{len(full_test)}. Not the full held-out set."),
    )
    print(f"\nsaved as run {uid}")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
