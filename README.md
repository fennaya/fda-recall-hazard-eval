# FDA drug recall hazard classification

Predict FDA's own hazard classification (Class I / II / III) for a drug recall
from its product description and free-text reason, and justify the call. FDA's
label is ground truth. The deliverable is a measured score.

Corpus: **17,938** drug enforcement records from openFDA (export 2026-09-08).

---

## The headline

**The agent does not beat the TF-IDF + LR baseline on macro F1** (0.472 vs
0.583). That was the explicit target set at the start of this project, and on
the full 1,275-case held-out test set it lost. Everything else below is
detail on top of that result, not a way around it.

## Scores

Held-out test split: **1,275** recalls initiated on or after 2025-01-01.
Training/precedent pool: **16,662** recalls before that date.

| System | Accuracy | Macro F1 | Class I recall | Class I missed | Mean cost | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: majority class | 87.45% | 0.311 | 0.00% | 67 / 67 | 0.336 | 428 |
| **Baseline: TF-IDF + LR** | 82.82% | **0.583** | 52.24% | 32 / 67 | 0.511 | 651 |
| Baseline: TF-IDF + LR (balanced) | 68.08% | 0.560 | 85.07% | 10 / 67 | 1.339 | 1,707 |
| **Agent** (tool_loop, Groq/openai-gpt-oss-120b, n=1,275) | 73.33% | 0.472 | 64.18% (43/67) | 24 / 67 | 0.499 | 636 |

Every number above is read from the `eval_runs` table (`run_uid
009d14e75268` for the agent row). Nothing here is a projection.

### The supported finding

At roughly the same cost as the unweighted TF-IDF baseline (**636 vs 651**),
the agent trades away precision for a **12-point Class I recall gain**
(64.18% vs 52.24%). That precision trade is steep: the agent's Class I
precision is **16.9%** (43 of the 254 cases it flags as Class I are actually
Class I) -- it catches more true positives, but at the cost of a much larger
flagged set. Whether that trade is worth it depends entirely on what the
score is used for; see the triage-workload table below.

### The failure pattern (Step 1 of the error analysis, full detail in
[`analysis/error_analysis.md`](analysis/error_analysis.md))

Reading all 20 highest-confidence false positives and all 10
highest-confidence false negatives in full transcript, the original
"over-weighting scary language" hypothesis does not hold up as a single
explanation. Six distinct patterns emerged:

| Pattern | Count (of 20 FP) |
|---|---:|
| No precedent retrieval attempted at all | 5 |
| Blanket rule via a repeated, generic precedent set (one firm cluster) | 6 |
| Correct retrieval, sound reasoning, ground truth still diverged | 5 |
| Cherry-picked the minority precedent in a mixed set | 2 |
| Route/exposure-mismatched precedent | 1 |
| Fabricated (non-`record_key`) citation | 1 |

All **10 of 10** sampled false negatives show the fourth pattern's mirror
image: correct retrieval, near-unanimous concordant Class II precedent,
sound reasoning -- and still wrong, because FDA classified this specific
instance as Class I. Combined, **15 of the 30 transcripts read (half)** show
reasoning that was internally sound and precedent-grounded but still missed
FDA's actual call. That argues for a classification-boundary/consistency
limitation in how predictable FDA's own decisions are from text alone, not
simply a model bias toward alarming language.

### If this were used as a first-pass screen (Step 3)

"Cases reviewed per real Class I caught" is the inverse of Class I
precision, phrased as a review workload: if every case a system flagged as
Class I were opened by a reviewer, how many opens does it take to find one
real hit.

| System | Flagged Class I | Share of all cases | Class I recall | Reviewed per real catch |
|---|---:|---:|---:|---:|
| Majority class | 0 | 0.0% | 0.0% (0/67) | n/a |
| TF-IDF + LR | 99 | 7.8% | 52.2% (35/67) | 2.83 |
| TF-IDF + LR (balanced) | 145 | 11.4% | 85.1% (57/67) | 2.54 |
| **Agent (tool_loop)** | **254** | **19.9%** | **64.2% (43/67)** | **5.91** |

**In plain operational terms:** the agent asks a reviewer to open a fifth of
the entire test set to find 43 real Class I recalls -- almost 6 opens per
real hit. The **balanced TF-IDF baseline dominates the agent outright** on
this framing: higher recall (57 vs 43), at less than half the review burden
per catch (2.54 vs 5.91), for barely more than half the flagged share (11.4%
vs 19.9%). Even the unweighted TF-IDF baseline needs far fewer opens per real
hit (2.83 vs 5.91) despite lower recall. A quality or regulatory team using
either baseline as a first-pass screen would review less and catch more or
comparably.

### Supply chain / distribution-breadth slices (Step 4)

`distribution_pattern` is a native, already-populated field on every recall
(no join needed). Bucketed by a fixed, documented rule (nationwide by
keyword; else count of US state/territory codes matched) -- full detail and
8 unit tests in `analysis/distribution_buckets.py`:

| Bucket | n | Share | FP rate | FN rate |
|---|---:|---:|---:|---:|
| nationwide | 1,125 | 88.2% | 17.4% | 37.5% |
| multi_state | 80 | 6.3% | 23.4% | 0.0% (n=1) |
| single_state | 61 | 4.8% | 14.8% | 0.0% (n=1) |
| other | 9 | 0.7% | 42.9% (n=7) | 0.0% (n=1) |

**No clear, sample-size-supported pattern.** False-positive rates are
broadly similar (14.8-23.4%) across the three buckets with usable sample
size. The false-negative comparison isn't usable outside `nationwide`: 64 of
the corpus's 67 true Class I recalls are nationwide-distributed, leaving
exactly one true Class I case in each other bucket. The one real finding
here is about the data, not the agent: Class I recalls in this corpus are
overwhelmingly nationwide.

### Reading the baselines

There is no single "the baseline" number, and which one you compare against
changes the bar:

- **Majority class** wins on accuracy (87.45%) by predicting Class II every
  time. It catches **zero** of 67 Class I recalls. Accuracy alone is a useless
  headline metric on this corpus.
- **TF-IDF + LR** is the strongest all-round baseline: best macro F1 (0.583),
  and the agent did not beat it.
- **TF-IDF + LR (balanced)** catches 85% of Class I recalls, dominates the
  agent on the triage-workload framing above, and does it at less than 3x the
  majority baseline's cost.

### Class I is reported separately

Missing a Class I recall -- one with a reasonable probability of serious
injury or death -- is the expensive error. The test split contains only 67
Class I recalls, so Class I recall is reported with a 95% Wilson interval on
the dashboard; the point estimate alone overstates what the data supports.

### Cost-weighted metric

`COST[truth][predicted]`, defined in `metrics.py`:

| truth \ predicted | Class I | Class II | Class III |
| --- | ---: | ---: | ---: |
| **Class I** | 0 | **5** | 10 |
| **Class II** | **1** | 0 | 5 |
| **Class III** | 1 | 1 | 0 |

Predicting Class II when the truth is Class I costs **5x** the reverse
mistake. This matrix is an **assumption**, not something derived from FDA
policy or measured consequence data -- see Limitations.

---

## Limitations

- **Single model, single provider, single run.** All agent numbers above are
  one run of `openai/gpt-oss-120b` via Groq. No comparison run on a second
  model or provider exists; whether these failure patterns are specific to
  this model or general to the tool-calling architecture is untested.
- **No confidence-based threshold tuning was possible.** The agent emits a
  `confidence` field, but it was not used in scoring and no threshold was
  fit on it -- every prediction shown here is the agent's raw discrete
  classification. A cost-sensitive decision rule would need either a
  validation split to fit a threshold on, or would have to be justified
  independent of the test set; neither was done here.
- **The cost matrix is an assumption.** The 5x asymmetry (missing Class I
  costs more than over-calling it) reflects a stated design principle, not a
  measured or FDA-published cost. Changing the matrix changes `total_cost`
  and `mean_cost` for every system, including the triage-workload framing's
  interpretation.
- **The 2 `no_submission` cases are scored as Class II.** Per Step 2's
  analysis (`analysis/error_analysis.md`), one is an agent-side loop, the
  other is an unrelated Groq daily-quota exhaustion -- both fell back to a
  default Class II guess rather than a real classification, and are counted
  in the reported metrics as such (not dropped).
- **A second, cheaper architecture exists but was only verified on n=20.**
  `agent_single.py` (retrieval done in Python, one model call instead of an
  average 2.07) was built and tested, and a 20-case free-tier check showed
  roughly half the token cost per case -- but it has not been run on the
  full 1,275-case set, and its own n=20 metrics are not a reliable
  comparison point. See `analysis/error_analysis.md`'s Step 3/4 note and
  `agent_single.py`'s docstring for detail. The two architectures' results
  are never combined into one score -- see the `architecture` column and
  `runs.record_run()`'s no-mixing check.

### What would fix it (future work, not done here)

A cost-sensitive decision rule -- reclassifying a prediction based on the
agent's own confidence plus the cost matrix, fit on a genuine validation
split rather than the test set -- or a calibrated confidence output the
rule could threshold against, are the natural next steps implied by the
triage-workload finding above. Neither is implemented in this repository.

---

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is provisioned by uv.

```bash
cp .env.example .env    # then fill in the keys below
uv sync
uv run -m fda_hazard.ingest      # pull the corpus into data/recalls.db
uv run -m fda_hazard.splits      # verify the temporal split invariants
uv run pytest                    # 184 tests
uv run -m fda_hazard.baselines   # score and record both baselines
uv run -m fda_hazard.evaluate    # score and record the agent (needs a key)
uv run python run_dashboard.py   # dashboard at http://127.0.0.1:8000
```

Every command is idempotent. Re-running `ingest` updates in place; re-running
`evaluate` with an unchanged prompt and model is served entirely from the
response cache and makes zero API calls.

### Configure a provider

Set **one** key in `.env`. The harness auto-detects which one is populated, in
the order groq → openrouter → baseten → anthropic:

| Provider | Env var | Default model |
| --- | --- | --- |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4.5` |
| Baseten | `BASETEN_API_KEY` | `deepseek-ai/DeepSeek-V3-0324` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |

Override with `LLM_PROVIDER` / `LLM_MODEL`, or `--provider` / `--model`.
`OPENFDA_API_KEY` raises openFDA rate limits; the bulk download path does not
need it.

**A note on free-tier throughput:** Groq's free tier is heavily rate-limited
in practice -- the full 1,275-case run above took roughly 11 days of
wall-clock time, dominated by provider-side retry waits, not by tokens per
case (mean 2,864 tokens/case; see the retry/slow-call log lines this harness
prints). Budget accordingly, or use `fda_hazard.budget.TokenBudget` (see
below) to cap spend on a paid key.

Useful flags:

```bash
uv run -m fda_hazard.evaluate --limit 50          # smoke-test on 50 cases first
uv run -m fda_hazard.evaluate --sample 150        # class-proportional partial run
uv run -m fda_hazard.evaluate --workers 1 --k 5   # workers>1 can worsen rate-limit contention
uv run -m fda_hazard.evaluate --no-lookups        # skip openFDA network lookups
uv run -m fda_hazard.verify_single_call --sample 20   # try the single-call architecture, small n
```

Start with `--limit 50` or `--sample` before committing to a full run,
especially on a paid key -- see `fda_hazard.budget` for a hard token ceiling
you can attach before spending anything.

---

## How it works

### 1. Corpus

`ingest.py` pulls the full corpus from the bulk download at
open.fda.gov/data/downloads (one 3.82 MB partition), falling back to
date-sliced pagination if that is stale or unreachable. The fallback slices by
year so `skip` never approaches the API's 26,000 ceiling, and raises rather
than silently truncating if a year ever exceeds it.

Field names were confirmed against a live sample record before the schema was
written. Each record is stored as parsed columns **plus** its untouched
`raw_json`, so nothing is lost.

Two records ship with no usable `recall_number` (one `""`, one the literal
string `"N/A"`). They are otherwise complete, so the primary key is a surrogate
(`record_key`) that falls back to a content hash — under a natural key those two
would collide and overwrite each other.

**Class distribution (n=17,937 labeled; one record is "Not Yet Classified"):**

| Class | Count | Share |
| --- | ---: | ---: |
| Class II | 14,484 | 80.7% |
| Class I | 1,744 | 9.7% |
| Class III | 1,709 | 9.5% |

Imbalance is **8.5:1** largest-to-smallest. Class II is 8.3x Class I and 8.5x
Class III; Class I and Class III are near-identical in size, so the whole skew
is Class II against the other two.

### 2. Split

Time-based, never random. Recalls cluster hard by `event_id` (17,938 records
share only 4,668 event_ids) and by firm, so a random split would put sibling
recalls of the same incident — often with verbatim-identical reason text — on
both sides and inflate the score.

The cut is at **2025-01-01** on `recall_initiation_date`, the date the event
occurred. `report_date` and `center_classification_date` are both downstream of
FDA's own classification decision, so splitting on them would order examples by
when the label was assigned rather than when the event happened.

`splits.verify_split()` asserts, and is run before every training and scoring
step:

- no `record_key` on both sides,
- **no `event_id` spans the cutoff** (currently 0 do),
- every date actually respects the cutoff.

122 firms appear on both sides. That is expected and is not leakage — firms
recur across years, and excluding them would bias the test split toward
one-time recallers.

### 3. Baselines

Fit on train, scored on test. Majority class, and TF-IDF (1-2 grams over
`reason_for_recall`) into multinomial logistic regression, in both unweighted
and `class_weight='balanced'` form. Seed fixed at 20250101.

### 4. Agent — two architectures, never mixed

Both live in the repo; both are runnable; a scored run must declare which one
it is (`architecture` column on `eval_runs`/`predictions`), and
`runs.record_run()` raises if a run's own predictions disagree on that.

**`agent.py` (`architecture='tool_loop'`)** — the agent decides when to call
tools, in up to 4 turns:

- **`lookup_drug_context`** — resolves a drug name from the free-text product
  description and queries the openFDA **label** and **NDC** endpoints for
  indication, route, pharmacologic class and boxed-warning status. The
  embedded `openfda` block is populated for only 3,261 of 17,938 recalls
  (18%), so this resolves from text rather than keying off that block, and
  "no match" is a normal, frequent outcome the agent must reason past.
- **`find_precedents`** — the k most similar historical recalls with their FDA
  classifications.
- **`submit_classification`** — the only structured output the harness reads:
  `{classification, confidence, reasoning, precedents_cited}`.

**`agent_single.py` (`architecture='single_call'`)** — precedent retrieval and
drug-context lookup happen in deterministic Python *before* the model is
called, using the identical `PrecedentIndex` (same leakage guarantee, same
tests cover it unchanged); the model gets one turn with everything already
provided and must submit. Built to test whether the tool-calling loop's
turn-by-turn context resending (measured ~1.4x per case) was the throughput
bottleneck -- it wasn't the dominant one; see `analysis/error_analysis.md`
and the Limitations section above for what was verified and what wasn't.

Both prompts quote FDA's own 21 CFR 7.3(m) definitions and direct the agent
to identify the physical defect, reason about its clinical consequence for
the patient, then weigh precedent.

**Precedent retrieval can never touch the test split**, enforced in code and
shared by both architectures since both call the same `PrecedentIndex`:

- `PrecedentIndex` is built from a list of train `Example`s and holds no
  database reference, so there is no code path by which a test record could be
  returned.
- Its constructor raises `LeakageError` if handed any example dated on or after
  the cutoff, so passing the wrong split is an exception, not a silent
  inflation.
- The example being classified is always excluded from its own precedent set.

Verified exhaustively on the real corpus: across all 1,275 test queries at k=5,
**zero** test records were returned as precedent.

**Resilience:** both `HazardAgent.classify()` and `classify_single()` never
raise on a per-case failure -- any exception becomes a scored, low-confidence
Class II fallback with the error preserved, so one bad case cannot take down
a run spanning days. A hard token budget (`fda_hazard.budget.TokenBudget`) can
be attached to an `LLMClient` to cap total spend across a run and terminate
before a ceiling is crossed; it deliberately escapes both agents' broad
exception handling (see `budget.py`'s docstring) so a cost cap can never be
silently absorbed as a per-case fallback.

### 5. Eval

`evaluate.py` collects predictions and hands them to `metrics.py`. Results go to
`eval_runs` (git SHA, whether the tree was dirty, prompt version, prompt hash,
model, provider, architecture, timestamp, full metrics JSON) and one row per
example to `predictions` (truth, prediction, cost, confidence, reasoning,
precedents seen and cited, tool calls).

Every model response is cached in SQLite under a hash of the full request —
prompt, tools, model, temperature — so an unchanged re-run is free and a changed
prompt correctly misses the cache.

### 6. Dashboard

FastAPI + HTMX + Plotly, server-rendered, no React and no build step. Dark and
light themes, tabular numerals, no emoji.

- **Overview** — current score against all baselines, class distribution,
  confusion matrix heatmap, accuracy over time by recall year.
- **Runs** — every eval run with its score, prompt version, git SHA, and deltas
  against the previous run of the same system.
- **Errors** — every misclassification, sorted by cost, showing the reason text,
  the agent's reasoning, the precedents it cited, and FDA's actual class. Class I
  misses are marked. Filterable by truth class, sortable without a page reload.
- **Single case** — paste a recall, run the agent live, see its reasoning and
  precedents.

---

## Constraints, and how they are held

**The model never computes a metric.** All scoring lives in `metrics.py`: pure
Python, no ML library, no model. The agent only ever emits a class label. Its
`confidence` is stored and displayed but never enters any score.

**Every claim in the dashboard traces to a row in the database.** All reads go
through `queries.py`; templates receive values, never expressions. Nothing is
recomputed at render time. `tests/test_runs.py` asserts that stored headline
metrics match the scorer and that per-row costs sum to the run total.

**A scored run can never blend two agent architectures.** `record_run()`
resolves every example's declared architecture and raises before writing
anything if they disagree — tested directly in `test_runs.py`.

**Tested.** 184 pytest tests covering the cost matrix and its asymmetry, macro
F1 by hand, confusion matrices, Wilson intervals, split invariants, the
event_id-spanning leak, retrieval's test-split ban (shared by both
architectures), the agent loop against a scripted fake provider, provider
resolution and request shaping for both API dialects, cache hit/miss
behaviour, run persistence, the no-mixing-architectures check, the token
budget (including that it escapes both agents' exception handling), and the
distribution-bucketing rule. Metric tests use hand-computed expected values,
not regenerated output.

```bash
uv run pytest
```

### Known limitations

(See also the dedicated **Limitations** section above, which covers the
agent's scored result specifically.)

- The test split holds only 67 Class I recalls. Class I recall has a wide
  confidence interval; treat single-run movements of a few points as noise.
- `recall_initiation_date` spans 2006-2026, but the test window is 2025-2026
  only. A model is being asked to generalise forward roughly one year.
- Class distribution differs across the split: Class II is 80.2% of train but
  87.5% of test, which flatters the majority baseline on test specifically.
- Drug-context lookups depend on a heuristic drug-name parse of firm-written
  free text. It recovers the active ingredient for typical descriptions but will
  miss unusual ones, and returns no match rather than a wrong one.
- Groq's free tier's actual throughput ceiling (see Reproduce) means a full
  run is a multi-day commitment; the `single_call` architecture and
  `TokenBudget` exist specifically to make a paid-key run affordable and
  boundable, but neither has been exercised at full scale.

## Layout

```
src/fda_hazard/
  db.py               schema, connection, LABEL_CLASSES
  ingest.py           bulk download + paginated fallback
  splits.py           temporal split and leakage assertions
  metrics.py          all scoring: pure Python, no model
  baselines.py        majority and TF-IDF + LR
  retrieval.py        train-only precedent index, shared by both architectures
  drug_context.py     openFDA label/NDC lookups, cached
  llm.py              provider abstraction + response cache + retry/slow-call logging
  budget.py           hard token ceiling, escapes both agents' exception handling
  agent.py            tool_loop architecture: prompt, tools, agent loop
  agent_single.py     single_call architecture: Python-side retrieval, one call
  evaluate.py         test-split evaluation driver (tool_loop)
  verify_single_call.py  small-sample verification driver (single_call)
  runs.py             run and prediction persistence, no-mixing enforcement
  queries.py          every dashboard read
  app.py              FastAPI routes
  templates/          Jinja2 + HTMX
analysis/
  error_analysis.md          Steps 1-4 of the post-hoc error analysis
  distribution_buckets.py    Step 4's documented bucketing rule
scripts/
  launch_detached.ps1        launches a run as a process independent of the
                              calling shell (survives the shell/session ending)
tests/                184 tests
```
