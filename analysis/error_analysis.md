# Error analysis: agent (tool_loop, Groq/openai-gpt-oss-120b), full run

Run `009d14e75268`, n=1,275, the complete held-out test set. All content below
is read from stored `predictions` rows for that run and from the `recalls`
table it joins to. **No model was re-run and no reported number changes.**
Counts quoted here (199 false positives, 24 false negatives, 254 flagged
Class I, etc.) were independently re-derived from the stored predictions and
checked against the confusion matrix already reported for this run; they
match exactly.

Any grouping choice below (confidence as a tie-break, the failure-mode labels,
the distribution-breadth rule) is a **post-hoc descriptive categorisation**
applied uniformly after the fact. None of it was chosen by looking at which
choice would produce a better number, and none of it changes accuracy, macro
F1, cost, or Class I recall as already reported.

---

## Step 1: false-positive and false-negative transcripts

### False positives (truth Class II, predicted Class I) -- 199 total, top 20 read

The cost matrix assigns this cell a uniform unit cost of 1, so "highest-cost"
doesn't distinguish among them; the 20 read here are the 20 with the agent's
own highest stated confidence, as a documented, non-outcome-based tie-break.

Reading all 20 full transcripts (reasoning, precedents retrieved, precedents
cited, tool-call sequence) gives six distinct patterns, not one:

| Pattern | Count | Description |
|---|---:|---|
| **No precedent retrieval attempted** | 5 | Model classified from an internal "sterility failure -> Class I" prior without ever calling `find_precedents`. |
| **Blanket rule via a repeated, generic precedent set** | 6 | Same firm (GenoGenix LLC compounded injectables), same defect text ("Lack of Assurance of Sterility"), same retrieval query, same 5-precedent set every time (4x Class I + 1x an unrelated Class III labeling case) -- the model cites the same Class I precedent (`D-0250-2023`) verbatim across all 6, never engaging with product-specific detail. |
| **Correct retrieval, defensible reasoning, ground truth diverged** | 5 | Precedent was genuinely on-point (subpotent fentanyl; hair found in injectable vials from the same manufacturer) and mostly/unanimously Class I in the retrieved set -- the analogy was reasonable, but FDA's actual call on *this* instance was Class II. |
| **Cherry-picked the minority precedent** | 2 | Retrieved set was mixed (e.g. 3 Class I + 2 Class II, or 4 Class II + 1 Class I) and the model's cited reasoning follows only the more severe outcome, without addressing the contrary majority. |
| **Route/exposure-mismatched precedent** | 1 | Benzene in a topical acne gel (dermal absorption) justified by citing benzene-in-hand-sanitizer precedent (inhalation/high-volume-use exposure) -- same contaminant, different exposure route, not addressed. |
| **Fabricated citation** | 1 | `precedents_cited` contains a paraphrased sentence, not a real `record_key`, and no `find_precedents` call was made in the transcript at all. |

**Evidence, one example per pattern** (record key, quoted phrase):

- *No retrieval* -- `D-0481-2025` (Semaglutide/Cyanocobalamin injectable): `tool_calls_json` is a single `submit_classification` call, no `find_precedents` anywhere. Reasoning: *"FDA guidance and historical precedents consistently place sterility failures for injectable drugs in Hazard Class I"* -- asserted, not retrieved.
- *Blanket rule* -- `D-0052-2026`, `D-0054-2026`, `D-0055-2026`, `D-0059-2026`, `D-0061-2026`, `D-0089-2026` (all GenoGenix LLC, all "Lack of Assurance of Sterility"): every one queries `"sterility injection recall Class I"`, gets back the identical 3-5 record set, and cites `D-0250-2023` as the deciding precedent verbatim. The retrieved set itself contains a Class III case (`D-1139-2017`, a labeling miscode) that none of the six reasoning texts mention or distinguish from.
- *Diverged despite sound reasoning* -- `D-0548-2026` (subpotent fentanyl citrate): retrieved 5/5 Class I fentanyl-subpotency precedents, reasoned *"FDA has historically classified subpotent fentanyl injection recalls as Class I (see precedents D-1154-2016, D-1157-2016, ...)"* -- a real, close, unanimous precedent match; the true label here was Class II regardless.
- *Cherry-picked minority* -- `D-0005-2026` (hand sanitizer, cGMP/methanol risk): retrieved set was `["Class II","Class II","Class III","Class I","Class II"]` (4 of 5 not Class I); reasoning cites only the single Class I hit (`D-0298-2022`) and does not mention the four contrary results.
- *Route mismatch* -- `D-0273-2025` (benzene in Zapzyt acne gel, topical): cites `D-0002-2023`, described in its own precedent record as *"Antica Farmacista Hand Sanitizer ... product found to contain benzene"* -- a different route of exposure, not discussed.
- *Fabricated citation* -- `D-0498-2025`: `precedents_cited: ["Sterility failure of injectable anesthetic (e.g., lidocaine, bupivacaine) classified as Class I by FDA"]` -- not a `record_key` format, and `tool_calls_json` shows no `find_precedents` call preceding it.

### False negatives (truth Class I, predicted Class II) -- 24 total, top 10 read

All 10 of the highest-confidence misses show the **same** pattern, and it is
close to the mirror image of the "diverged despite sound reasoning" FP
pattern above:

| Pattern | Count | Description |
|---|---:|---|
| **Correct retrieval, unanimous/near-unanimous concordant precedent, reasoning sound, ground truth diverged** | 10 / 10 | Every case calls `find_precedents`, gets back 4/5 or 5/5 Class II matches on a genuinely similar defect (microbial contamination of a *non-sterile* product: nasal sprays, topical antiseptics, an infant oral swab, IV particulate matter), reasons soundly from that evidence, and is wrong because FDA classified this specific instance as Class I. |

No false negative in this top-10 sample shows a missing retrieval call, a
fabricated citation, or a route mismatch -- the defect pattern here is
different in kind from the false-positive side.

**Evidence:**
- `D-0288-2026` (ReBoost Nasal Spray, Achromobacter contamination): retrieved `["Class II","Class II","Class II","Class II","Class II"]` (5/5), reasoning: *"FDA precedents for similar nasal spray microbial contamination ... were classified as Class II, reflecting the same risk profile."* Truth: Class I.
- `D-0611-2025`, `D-0612-2025`, `D-0613-2025` (three DermaRite Industries products, all *Burkholderia cepacia* contamination): each retrieves 4-5/5 Class II precedent, each reasons that the topical/non-sterile route limits systemic risk. All three truth: Class I.
- `D-0787-2026` (Cefazolin injection, particulate matter): retrieved `["Class II","Class II","Class II","Class II","Class II"]` (5/5), reasoning: *"Particulates in injectable solutions can cause embolic events, phlebitis, or infusion reactions, but these are generally temporary or reversible."* Truth: Class I.

### Was "over-weighting scary language" the right explanation?

**No, or at best it is a partial description of one of six false-positive
patterns and none of the false-negative pattern.** The original hypothesis
implied a single, simple bias (alarming wording pushes the model toward
Class I). The evidence instead shows:

1. The largest single false-positive cluster (6/20, the GenoGenix group) is
   better described as **a blanket textual rule applied without regard to
   product-specific detail, reinforced by a retrieval query generic enough to
   return the same precedent set for six different products.**
2. A further 5/20 false positives involve **no precedent lookup at all** --
   not "scary language overriding evidence," but reasoning from an unexamined
   prior with no evidence consulted.
3. Another 5/20 false positives, and **all 10/10** of the sampled false
   negatives, show **sound reasoning consistent with genuinely on-point
   retrieved precedent** -- these are not reasoning failures in an obvious
   sense; they are cases where FDA's own historical classification for this
   defect category does not fully predict FDA's classification of the new
   instance from text alone. This is the single largest pattern across both
   error types combined (15 of 30 transcripts read) and argues for a
   **classification-boundary/consistency limitation**, not an agent bias.
4. Only 2/20 false positives clearly fit "the model saw alarming language and
   ignored contrary evidence" (the cherry-picking pattern), and 1/20 fits a
   route-mismatch variant of the same idea.

---

## Step 2: the 2 `no_submission` cases

`D-0353-2026` and `D-0372-2026` are unrelated: `D-0353-2026` is an
agent-side loop -- it called `find_precedents` three times with
near-identical rewordings of the same query plus one `lookup_drug_context`
call, never called `submit_classification`, and was cut off at the 4-turn
limit. `D-0372-2026` never got a model response at all: Groq's own error
message shows its **daily** token quota was exhausted ("Rate limit reached
... tokens per day (TPD): Limit 200000, Used 199434, Requested 1481")
and all 12 retry attempts failed with 429, an infrastructure limit rather
than anything about this specific input.
