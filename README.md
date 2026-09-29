# Can an AI agent triage FDA drug recalls?

[![DOI](https://zenodo.org/badge/DOI/DOI_HERE.svg)](https://doi.org/DOI_HERE)

Not better than a simple baseline. I checked.

When a drug is recalled, FDA assigns a hazard class. **Class I** means the
product can cause serious harm or death, so it has to be pulled fast and
deep, down to pharmacies and patients. Class II and III are less severe.
Getting the class wrong means either under-reacting to a dangerous product
or flooding a quality team with false alarms.

I built an LLM agent that reads a recall, pulls similar past recalls as
precedent, and predicts the class. Then I scored it against FDA's own labels
on 1,275 recalls it had never seen.

## Result

| System | Macro F1 | Class I recall | Class I precision | Flagged as Class I | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always guess Class II | 0.311 | 0% | n/a | 0 | 428 |
| Keyword model (TF-IDF + LR) | **0.583** | 52.2% | 35.4% | 99 | 651 |
| Keyword model, reweighted | 0.560 | **85.1%** | 39.3% | 145 | 1,707 |
| Agent (gpt-oss-120b on Groq) | 0.472 | 64.2% | 16.9% | 254 | 636 |

Lower cost is better. Cost weights a missed Class I higher than a false alarm, but not enough: under my weights, always guessing Class II is the cheapest system, even though it misses every Class I recall. Read the cost column with that in mind. The weights are my assumption, stated in the code.

## What I found

1. **The agent lost.** A plain keyword model beats it on the main metric.
2. **As a screen, it's worse too.** The reweighted keyword model flags fewer
   cases (145 vs 254) and catches more real Class I recalls (85% vs 64%).
   A reviewer opens 2.5 cases per real catch instead of 5.9. It does cost
   more overall under my weights, because it makes other mistakes.
3. **It made up a source once.** For one false positive, it cited
   "Sterility failure of injectable anesthetic (e.g., lidocaine,
   bupivacaine) classified as Class I by FDA" as its precedent. That is
   not a real record ID from the corpus, and the transcript shows no
   precedent search was ever run before the citation was written.
4. **Sometimes by my reading, the agent reasoned soundly and FDA disagreed with its own past.**
   In all 10 missed Class I recalls, the agent found near-unanimous Class II
   precedent and followed it. FDA still said Class I. Either FDA's practice
   changed over time, or its labels are inconsistent. I don't know which yet.

Full breakdown, with every failure pattern counted:
[`analysis/error_analysis.md`](analysis/error_analysis.md)

## Limits

- One model, one provider, one run.
- The cost weights are an assumption.
- The 30 errors I read closely were the most costly ones, not a random sample.
- 2 cases got no answer (one agent loop, one Groq quota cutoff). Both scored as Class II.
- Data: openFDA enforcement records, snapshot of 2026-09-08.

## Run it

```bash
cp .env.example .env    # add one provider key
uv sync
uv run -m fda_hazard.ingest
uv run pytest           # 184 tests
uv run -m fda_hazard.baselines
uv run -m fda_hazard.evaluate
uv run python run_dashboard.py   # http://127.0.0.1:8000
```

On Groq's free tier the full run took about 11 days. Try `--sample 20` first.

## Next

- Test whether FDA's classification drifted over time, or is just noisy.
- Use the agent's confidence plus the cost weights to set a better threshold.
- Try a second model.

## License

Code: [MIT](LICENSE). Text and analysis: [CC BY 4.0](LICENSE-DATA).
openFDA data keeps its own terms. Cite via [`CITATION.cff`](CITATION.cff).
Author: Aya Emssaad.
