# Changelog

## v1.0.0

Initial public release: an FDA drug recall hazard classification harness and
its measured, honestly-reported result.

**Headline result:** the agent (tool_loop architecture, Groq/openai-gpt-oss-120b,
full 1,275-case held-out set) does **not** beat the TF-IDF + LR baseline on
macro F1 (0.472 vs 0.583). At roughly the same cost (636 vs 651), it trades
precision for a 12-point Class I recall gain (64.2% vs 52.2%), at 16.9%
Class I precision. Full detail in `README.md` and `analysis/error_analysis.md`.

**Corpus:** 17,938 drug enforcement records from openFDA, ingested with raw
JSON preserved alongside parsed columns. Time-based train/test split (cutoff
2025-01-01) with verified no-leakage guarantees on both the split and on
precedent retrieval.

**Three scored systems, four runs:** majority-class baseline, TF-IDF + LR
(unweighted and class-balanced), and the agent -- each independently scored
by the same pure-Python, tested metrics code (accuracy, macro F1, a
cost-weighted metric, Class I recall with a Wilson interval).

**Two agent architectures**, both implemented, both tested, never mixed in
one score: `tool_loop` (the model decides when to call precedent-retrieval
and drug-context tools; scored on the full test set) and `single_call`
(retrieval done in Python before a single model call; verified on a 20-case
free-tier sample only, not run at full scale).

**Post-hoc error analysis** (`analysis/error_analysis.md`), entirely
additive over stored predictions, no re-run: six distinct false-positive
failure patterns identified from full transcripts (including one fabricated
citation, named as a finding), a mirrored false-negative pattern, a
triage-workload comparison across all four systems, and a distribution-breadth
error slice using a documented, tested bucketing rule.

**Dashboard:** FastAPI + HTMX + Plotly, server-rendered, every number traced
to a database row -- Overview, Runs, Errors, and a live single-case tester.

**Reliability infrastructure** built out over the course of a multi-day
free-tier run: per-case resilience (a single failure never kills the run),
retry/slow-call logging, a hard token budget with a kill switch that
deliberately escapes both agents' exception handling, and a detached-process
launcher for runs that must survive the launching session ending.

**184 tests.** Licensed: code under MIT (`LICENSE`), README/analysis/written
results under CC BY 4.0 (`LICENSE-DATA`); openFDA data and firm-submitted
text keep their own terms.
