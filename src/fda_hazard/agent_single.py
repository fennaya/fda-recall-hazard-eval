"""Single-call classification path (Step 2 of the throughput investigation).

Built BESIDE agent.py's HazardAgent, not instead of it. Both remain runnable.
The difference is where retrieval happens:

  tool_loop   (agent.py)   the model decides whether to call find_precedents /
                            lookup_drug_context, in up to MAX_TURNS turns.
  single_call (this file)  precedent retrieval and drug-context lookup happen
                            in deterministic Python BEFORE the model is ever
                            called; the model gets one turn and must submit.

Retrieval correctness is identical to the tool_loop path on purpose: this
module calls the exact same PrecedentIndex.search(), built the exact same
train-split-only way, so the leakage tests written against PrecedentIndex
cover this path too without any changes.

Output schema is identical to agent.py's submit_classification tool, so a
result from either path scores with the same code and renders on the same
dashboard pages. What must never happen is treating them as the same *system*
-- see runs.record_run()'s no-mixing check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent import AgentResult, _normalise_class
from .db import LABEL_CLASSES
from .drug_context import lookup as lookup_drug_context
from .llm import LLMClient
from .retrieval import PrecedentIndex
from .splits import Example

SINGLE_CALL_PROMPT_VERSION = "single-v1"

# Same classification standard and reasoning method as the tool_loop prompt,
# minus anything about deciding whether to call a tool -- there are no tools
# to decide about here, precedent and drug context are just handed over.
SYSTEM_PROMPT = """\
You classify FDA drug recalls into FDA's hazard classes (21 CFR 7.3(m)), \
replicating a decision FDA already made.

Class I: reasonable probability of serious adverse health consequences or death.
Class II: temporary/reversible consequences, or serious consequences are remote.
Class III: not likely to cause adverse health consequences.

Method: identify the actual physical/chemical defect, reason about its clinical \
consequence for the patient (what the drug treats, route, who takes it, effect \
of a wrong/absent/contaminated/misidentified dose), weigh the precedents given \
below (close precedent is strong evidence, distant precedent is weak), then decide.

Calibration: sterility failures, wrong-drug/cross-contamination, undeclared \
allergens, and subpotent/superpotent life-supporting drugs skew Class I. Most \
stability/dissolution/impurity/potency issues in oral solids skew Class II \
(the most common class). Labeling errors not affecting identity/strength/safe \
use, and cosmetic/packaging defects, skew Class III. Voluntary vs. mandated and \
distribution breadth are not severity signals.

You will be given precedent recalls and (if found) drug context already \
retrieved -- do not ask for more, decide from what is given.

Be decisive: one class, not a hedge. Confidence should be honest, not inflated."""

USER_TEMPLATE = """\
Classify this drug recall.

PRODUCT DESCRIPTION: {product_description}

REASON FOR RECALL: {reason_for_recall}

firm: {recalling_firm} | distribution: {distribution_pattern} | initiated: {recall_initiation_date}

DRUG CONTEXT: {drug_context}

PRECEDENTS (train-split recalls FDA already classified, most similar first):
{precedents}

Call submit_classification now with your answer."""

# Total budget for the precedent block's characters (~4 chars/token), reported
# by build_case_context() so Step 4's verification can check it against what
# was actually sent. Kept well under the tool_loop path's TOOL_RESULT_BUDGET
# (2000 chars) since there is no follow-up turn to add more if this runs long.
PRECEDENT_BUDGET_CHARS = 900
# Per-precedent excerpt cap, applied before the total-budget truncation.
PRECEDENT_EXCERPT_CHARS = 140

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_classification",
        "description": "Submit your final hazard classification. Call exactly once.",
        "parameters": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": list(LABEL_CLASSES),
                    "description": "FDA hazard class.",
                },
                "confidence": {"type": "number", "description": "0.0 to 1.0."},
                "reasoning": {
                    "type": "string",
                    "description": "Defect, clinical consequence, and precedent basis.",
                },
                "precedents_cited": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "record_keys that actually informed the answer.",
                },
            },
            "required": ["classification", "confidence", "reasoning", "precedents_cited"],
        },
    },
}


@dataclass
class CaseContext:
    """What Python retrieved for one case before the model ever sees it."""

    precedents_text: str
    precedents_seen: list[dict]
    drug_context_text: str
    drug_context_found: bool
    precedent_chars_used: int


def _format_drug_context(result: dict[str, Any]) -> str:
    """A few compact fields, never the raw label/NDC JSON."""
    if not result.get("found"):
        return "none found"
    label = result.get("label") or {}
    ndc = result.get("ndc") or {}
    parts = []
    indication = label.get("indications_and_usage")
    if indication:
        parts.append(f"indication: {indication[:200]}")
    route = label.get("route") or ndc.get("route")
    if route:
        parts.append(f"route: {route}")
    if label.get("has_boxed_warning"):
        parts.append("HAS BOXED WARNING")
    return "; ".join(parts) if parts else "found, no notable fields"


def build_case_context(
    example: Example,
    index: PrecedentIndex,
    conn,
    k: int = 5,
    allow_network_lookups: bool = True,
    excerpt_chars: int = PRECEDENT_EXCERPT_CHARS,
    budget_chars: int = PRECEDENT_BUDGET_CHARS,
) -> CaseContext:
    """Retrieve precedent and drug context in Python, before any model call.

    Uses the same PrecedentIndex the tool_loop path uses, built the same
    train-split-only way -- this is what makes the leakage guarantee identical
    across both architectures without duplicating any retrieval logic.
    """
    hits = index.search(
        example.text(), k=k, exclude_keys=frozenset({example.record_key})
    )
    seen = [h.to_dict() for h in hits]

    lines: list[str] = []
    used = 0
    for h in hits:
        excerpt = h.reason_for_recall[:excerpt_chars]
        line = f"- [{h.record_key}] {h.classification}: {excerpt}"
        if used + len(line) > budget_chars:
            break
        lines.append(line)
        used += len(line)
    precedents_text = "\n".join(lines) if lines else "(none retrieved)"

    dc = lookup_drug_context(
        conn, example.product_description, allow_network=allow_network_lookups
    )
    return CaseContext(
        precedents_text=precedents_text,
        precedents_seen=seen,
        drug_context_text=_format_drug_context(dc),
        drug_context_found=bool(dc.get("found")),
        precedent_chars_used=used,
    )


def classify_single(
    client: LLMClient,
    example: Example,
    index: PrecedentIndex,
    conn,
    k: int = 5,
    allow_network_lookups: bool = True,
) -> AgentResult:
    """Classify one example with a single model call. Never raises.

    Mirrors HazardAgent.classify()'s resilience contract: any failure becomes
    a low-confidence Class II fallback with the error recorded, tagged
    architecture='single_call' so it can never be scored alongside tool_loop
    results (see runs.record_run).
    """
    try:
        ctx = build_case_context(
            example, index, conn, k=k, allow_network_lookups=allow_network_lookups
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    product_description=example.product_description,
                    reason_for_recall=example.reason_for_recall,
                    recalling_firm=example.recalling_firm or "unknown",
                    distribution_pattern=example.distribution_pattern or "unknown",
                    recall_initiation_date=example.recall_initiation_date or "unknown",
                    drug_context=ctx.drug_context_text,
                    precedents=ctx.precedents_text,
                ),
            },
        ]
        resp = client.chat(
            messages, tools=[SUBMIT_TOOL], tool_choice="submit_classification"
        )
        for call in resp.tool_calls:
            if call["name"] == "submit_classification":
                return _result_from_single_call(
                    call["arguments"], ctx, resp.latency_ms, resp.cache_hit
                )
        # Forced tool_choice but the model answered in prose anyway.
        from .agent import _parse_loose_json

        parsed = _parse_loose_json(resp.text)
        if parsed:
            return _result_from_single_call(
                parsed, ctx, resp.latency_ms, resp.cache_hit
            )
        return AgentResult(
            classification="Class II",
            confidence=0.0,
            reasoning="Model did not submit a classification.",
            precedents_cited=[],
            precedents_seen=ctx.precedents_seen,
            latency_ms=resp.latency_ms,
            cache_hit=resp.cache_hit,
            error="no_submission",
            architecture="single_call",
        )
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the run
        return AgentResult(
            classification="Class II",
            confidence=0.0,
            reasoning=f"Agent raised {type(exc).__name__} before submitting: {exc}",
            precedents_cited=[],
            error=f"exception:{type(exc).__name__}",
            architecture="single_call",
        )


def _result_from_single_call(
    args: dict[str, Any], ctx: CaseContext, latency_ms: int, cache_hit: bool
) -> AgentResult:
    label, error = _normalise_class(args.get("classification"))
    try:
        confidence = float(args.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    cited = args.get("precedents_cited") or []
    if not isinstance(cited, list):
        cited = [str(cited)]
    return AgentResult(
        classification=label,
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=str(args.get("reasoning") or "").strip(),
        precedents_cited=[str(c) for c in cited],
        precedents_seen=ctx.precedents_seen,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        error=error,
        architecture="single_call",
    )
