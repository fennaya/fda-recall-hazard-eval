# FDA drug recall hazard classification

Predict FDA's own hazard classification (Class I / II / III) for a drug recall
from its product description and free-text reason, and justify the call. FDA's
label is ground truth. The deliverable is a measured score.

Corpus: **17,938** drug enforcement records from openFDA (export 2026-09-08).

---

## Scores

Held-out test split: **1,275** recalls initiated on or after 2025-01-01.
Training/precedent pool: **16,662** recalls before that date.

| System | Accuracy | Macro F1 | Class I recall | Class I missed | Mean cost | Total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: majority class | 87.45% | 0.311 | 0.00% | 67 / 67 | 0.336 | 428 |
| Baseline: TF-IDF + LR | 82.82% | 0.583 | 52.24% | 32 / 67 | 0.511 | 651 |
| Baseline: TF-IDF + LR (balanced) | 68.08% | 0.560 | 85.07% | 10 / 67 | 1.339 | 1,707 |
| **Agent** | not yet run | - | - | - | - | - |

The agent has **not been scored yet** — it needs an LLM provider key (see
[Configure a provider](#configure-a-provider)). Once run, its row appears here
and on the dashboard. Nothing in this README is a projection; every number above
is read from the `eval_runs` table.

### Reading the baselines

There is no single "the baseline" number, and which one you compare against
changes the bar:

- **Majority class** wins on accuracy (87.45%) by predicting Class II every
  time. It catches **zero** of 67 Class I recalls. Accuracy alone is a useless
  headline metric on this corpus.
- **TF-IDF + LR** is the strongest all-round baseline: best macro F1 (0.583) and
  the lowest cost of any system that actually discriminates.
- **TF-IDF + LR (balanced)** catches 85% of Class I recalls but pays for it —
  it over-predicts Class III, and its cost is 4x the majority baseline's.

**The agent has to beat TF-IDF + LR on macro F1 (0.583) and on cost (651) while
not losing Class I recall (52.24%)**, or it is not worth its cost.

### Class I is reported separately

Missing a Class I recall — one with a reasonable probability of serious injury
or death — is the expensive error. The test split contains only 67 Class I
recalls, so Class I recall is reported with a 95% Wilson interval; the point
estimate alone overstates what the data supports.

### Cost-weighted metric

`COST[truth][predicted]`, defined in `metrics.py`:

| truth \ predicted | Class I | Class II | Class III |
| --- | ---: | ---: | ---: |
| **Class I** | 0 | **5** | 10 |
| **Class II** | **1** | 0 | 5 |
| **Class III** | 1 | 1 | 0 |

Predicting Class II when the truth is Class I costs **5x** the reverse mistake.
Under-calling hazard severity is penalised more than over-calling everywhere in
the matrix, and the asymmetry is asserted in the tests.

---

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is provisioned by uv.

```bash
cp .env.example .env    # then fill in the keys below
uv sync
uv run -m fda_hazard.ingest      # pull the corpus into data/recalls.db
uv run -m fda_hazard.splits      # verify the temporal split invariants
uv run pytest                    # 121 tests
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
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4.5` |
| Baseten | `BASETEN_API_KEY` | `deepseek-ai/DeepSeek-V3-0324` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-5` |

Override with `LLM_PROVIDER` / `LLM_MODEL`, or `--provider` / `--model`.
`OPENFDA_API_KEY` raises openFDA rate limits; the bulk download path does not
need it.

Useful flags:

```bash
uv run -m fda_hazard.evaluate --limit 50          # smoke-test on 50 cases first
uv run -m fda_hazard.evaluate --provider groq --model llama-3.3-70b-versatile
uv run -m fda_hazard.evaluate --workers 8 --k 8   # parallelism, precedents per call
uv run -m fda_hazard.evaluate --no-lookups        # skip openFDA network lookups
```

Start with `--limit 50` to confirm the provider and model behave before paying
for the full 1,275-case run.

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

### 4. Agent

`agent.py`, provider-agnostic, structured output via tool calling. Given a
recall it may call:

- **`lookup_drug_context`** — resolves a drug name from the free-text product
  description and queries the openFDA **label** and **NDC** endpoints for
  indication, route, pharmacologic class and boxed-warning status. Note that the
  embedded `openfda` block is populated for only 3,261 of 17,938 recalls (18%),
  so this resolves from text rather than keying off that block, and "no match"
  is a normal, frequent outcome the agent must reason past.
- **`find_precedents`** — the k most similar historical recalls with their FDA
  classifications.
- **`submit_classification`** — the only structured output the harness reads:
  `{classification, confidence, reasoning, precedents_cited}`.

The prompt quotes FDA's own 21 CFR 7.3(m) definitions and directs the agent to
identify the physical defect, reason about its clinical consequence for the
patient, then weigh precedent.

**Precedent retrieval can never touch the test split**, enforced in code:

- `PrecedentIndex` is built from a list of train `Example`s and holds no
  database reference, so there is no code path by which a test record could be
  returned.
- Its constructor raises `LeakageError` if handed any example dated on or after
  the cutoff, so passing the wrong split is an exception, not a silent
  inflation.
- The example being classified is always excluded from its own precedent set.

Verified exhaustively on the real corpus: across all 1,275 test queries at k=5,
**zero** test records were returned as precedent.

### 5. Eval

`evaluate.py` collects predictions and hands them to `metrics.py`. Results go to
`eval_runs` (git SHA, whether the tree was dirty, prompt version, prompt hash,
model, provider, timestamp, full metrics JSON) and one row per example to
`predictions` (truth, prediction, cost, confidence, reasoning, precedents seen
and cited, tool calls).

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

**Tested.** 121 pytest tests covering the cost matrix and its asymmetry, macro
F1 by hand, confusion matrices, Wilson intervals, split invariants, the
event_id-spanning leak, retrieval's test-split ban, the agent loop against a
scripted fake provider, provider resolution and request shaping for both API
dialects, cache hit/miss behaviour, and run persistence. Metric tests use
hand-computed expected values, not regenerated output.

```bash
uv run pytest
```

### Known limitations

- The test split holds only 67 Class I recalls. Class I recall has a wide
  confidence interval; treat single-run movements of a few points as noise.
- `recall_initiation_date` spans 2006-2026, but the test window is 2025-2026
  only. A model is being asked to generalise forward roughly one year.
- Class distribution differs across the split: Class II is 80.2% of train but
  87.5% of test, which flatters the majority baseline on test specifically.
- Drug-context lookups depend on a heuristic drug-name parse of firm-written
  free text. It recovers the active ingredient for typical descriptions but will
  miss unusual ones, and returns no match rather than a wrong one.

## Layout

```
src/fda_hazard/
  db.py            schema, connection, LABEL_CLASSES
  ingest.py        bulk download + paginated fallback
  splits.py        temporal split and leakage assertions
  metrics.py       all scoring: pure Python, no model
  baselines.py     majority and TF-IDF + LR
  retrieval.py     train-only precedent index
  drug_context.py  openFDA label/NDC lookups, cached
  llm.py           provider abstraction + response cache
  agent.py         prompt, tools, agent loop
  evaluate.py      test-split evaluation driver
  runs.py          run and prediction persistence
  queries.py       every dashboard read
  app.py           FastAPI routes
  templates/       Jinja2 + HTMX
tests/             121 tests
```
