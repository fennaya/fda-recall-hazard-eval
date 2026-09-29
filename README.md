# FDA drug recall hazard classification

Predict FDA's own hazard class (Class I / II / III) for a drug recall from
its product description and reason text. FDA's label is ground truth,
measured honestly on a held-out test set.

## The result

**The agent does not beat the TF-IDF + LR baseline on macro F1** (0.472 vs
0.583). That was the target from the start, and it lost.

| System | Accuracy | Macro F1 | Class I recall | Cost |
| --- | ---: | ---: | ---: | ---: |
| Majority class | 87.45% | 0.311 | 0.00% | 428 |
| **TF-IDF + LR** | 82.82% | **0.583** | 52.24% | 651 |
| TF-IDF + LR (balanced) | 68.08% | 0.560 | 85.07% | 1,707 |
| **Agent** (Groq, n=1,275) | 73.33% | 0.472 | 64.18% | 636 |

What it *does* do: at near-equal cost to TF-IDF (636 vs 651), it trades
precision for 12 more points of Class I recall (64.2% vs 52.2%) — but at
only 16.9% Class I precision, and it loses to TF-IDF-balanced on every axis
of a "first-pass screen" framing (fewer flags, more catches, less review
burden). Full breakdown, failure patterns, and a named fabricated-citation
finding: [`analysis/error_analysis.md`](analysis/error_analysis.md).

## Run it

```bash
cp .env.example .env    # add one provider key
uv sync
uv run -m fda_hazard.ingest
uv run pytest                  # 184 tests
uv run -m fda_hazard.baselines
uv run -m fda_hazard.evaluate  # scores the agent, needs a key
uv run python run_dashboard.py # http://127.0.0.1:8000
```

Groq's free tier is heavily rate-limited in practice — the full run took
~11 days of wall-clock, mostly waiting, not computing. Use `--sample N` or
`--limit N` for a quick check first.

## What's next

- **Test the label-drift hypothesis.** Half the errors read (15/30
  transcripts) show sound, precedent-grounded reasoning that still missed
  FDA's call. Leading guess: FDA's own classification practice shifted over
  time for some defect categories. Unconfirmed — needs FDA guidance history,
  not more model runs.
- **Build a cost-sensitive decision rule.** Threshold the agent's confidence
  against the cost matrix instead of taking its raw label. Needs a real
  validation split (doesn't exist yet — only train/test) to fit on honestly.
- **Run `single_call` at full scale.** A second, ~2x cheaper architecture
  exists (`agent_single.py`) and is tested, but only verified on 20 cases.
  Needs either a paid key (~$8 estimated) or another week on the free tier.
- **Try a second model/provider.** Every result above is one model
  (`openai/gpt-oss-120b`) on one provider (Groq). Whether these failure
  patterns are model-specific is untested.

## License

Code: [MIT](LICENSE). README, analysis, and written results: [CC BY
4.0](LICENSE-DATA). openFDA data keeps its own terms — see `LICENSE-DATA`.
Citation metadata: [`CITATION.cff`](CITATION.cff).

## Layout

```
src/fda_hazard/     ingest, split, metrics, baselines, retrieval, agent
                    (two architectures: agent.py, agent_single.py),
                    evaluate.py, runs.py, queries.py, app.py, templates/
analysis/           error_analysis.md (the full breakdown), distribution_buckets.py
scripts/            launch_detached.ps1 (survives the launching session ending)
tests/              184 tests
```
