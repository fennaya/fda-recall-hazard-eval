"""The hazard-classification agent.

Given a recall's product description and reason, predict FDA's own hazard
class. The agent decides when to look up drug context and when to pull
precedent; it must finish by calling submit_classification, which is the only
structured output the harness reads.

The agent never sees a metric, never scores itself, and never sees the test
split's labels -- precedent comes from PrecedentIndex, which is train-only by
construction.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .db import LABEL_CLASSES
from .drug_context import lookup as lookup_drug_context
from .llm import LLMClient
from .retrieval import PrecedentIndex
from .splits import Example

PROMPT_VERSION = "v1"

# FDA's own definitions, quoted so the agent reasons against the actual
# standard rather than a vibe about severity. 21 CFR 7.3(m).
SYSTEM_PROMPT = """\
You classify FDA drug recalls into FDA's own hazard classes. You are replicating \
a decision FDA already made; your output is graded against their classification.

The three classes, per 21 CFR 7.3(m):

Class I  - a reasonable probability that use of, or exposure to, the product \
will cause SERIOUS ADVERSE HEALTH CONSEQUENCES OR DEATH.
Class II - use of, or exposure to, the product may cause temporary or medically \
reversible adverse health consequences, or the probability of serious adverse \
health consequences is REMOTE.
Class III - use of, or exposure to, the product is NOT LIKELY to cause adverse \
health consequences.

How to decide:

1. Identify the actual defect. Not the paperwork around it - the physical or \
chemical failure in the product as distributed.
2. Reason about the clinical consequence of that defect for the patient who \
takes the affected unit. Consider: what the drug treats, how it is \
administered, who takes it, and what happens if the dose is wrong, absent, \
contaminated, or misidentified.
3. Weigh the precedents you retrieve. FDA is highly consistent for recurring \
defect types, so a close precedent is strong evidence. A distant precedent is \
weak evidence - say so rather than following it.
4. Choose the class that matches the clinical consequence.

Calibration notes drawn from how FDA actually classifies:

- Sterility failures in injectables, cross-contamination with a different \
active ingredient, undeclared allergens, and life-supporting drugs that are \
subpotent or superpotent tend toward Class I.
- Most stability, dissolution, impurity, and moderate potency failures in oral \
solid dosage forms tend toward Class II. Class II is by far the most common \
class overall.
- Labeling errors that do not affect identity, strength, or safe use, cosmetic \
or packaging defects, and minor documentation failures tend toward Class III.
- A recall being voluntary, or nationwide, says little about hazard. Do not use \
distribution breadth as a severity signal.

Be decisive. Give one class, not a hedge. Set confidence honestly: low \
confidence is useful information, an inflated one is not."""

USER_TEMPLATE = """\
Classify this drug recall.

PRODUCT DESCRIPTION:
{product_description}

REASON FOR RECALL:
{reason_for_recall}

ADDITIONAL CONTEXT:
recalling firm: {recalling_firm}
distribution pattern: {distribution_pattern}
initiation: {recall_initiation_date}

Use your tools if they would change your answer, then call \
submit_classification."""


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLS = [
    _tool(
        "lookup_drug_context",
        "Look up a drug in the openFDA label and NDC endpoints to get its "
        "indication, route of administration, pharmacologic class, and whether "
        "it carries a boxed warning. Use this when the clinical consequence of "
        "the defect depends on what the drug is for or how it is given. Often "
        "returns no match, which is normal - proceed on the recall text.",
        {
            "type": "object",
            "properties": {
                "product_description": {
                    "type": "string",
                    "description": "The product description text to resolve a drug from.",
                }
            },
            "required": ["product_description"],
        },
    ),
    _tool(
        "find_precedents",
        "Retrieve the most similar historical recalls that FDA has already "
        "classified, with their classifications. These come only from recalls "
        "predating the evaluation period. Use this to check how FDA has treated "
        "this defect type before.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Defect description to find precedent for. Describe "
                    "the defect and product type, not the firm name.",
                },
                "k": {
                    "type": "integer",
                    "description": "How many precedents to return (1-10).",
                },
            },
            "required": ["query"],
        },
    ),
    _tool(
        "submit_classification",
        "Submit your final hazard classification. You must call this exactly "
        "once to finish.",
        {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": list(LABEL_CLASSES),
                    "description": "FDA hazard class.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence, 0.0 to 1.0.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this class: the defect, its clinical "
                    "consequence, and how the precedents bear on it.",
                },
                "precedents_cited": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "record_keys of precedents that actually "
                    "informed your answer. Empty if none did.",
                },
            },
            "required": ["classification", "confidence", "reasoning", "precedents_cited"],
        },
    ),
]

MAX_TURNS = 6


@dataclass
class AgentResult:
    classification: str
    confidence: float
    reasoning: str
    precedents_cited: list[str]
    precedents_seen: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    latency_ms: int = 0
    cache_hit: bool = True
    error: str | None = None

    def to_detail(self, record_key: str) -> dict[str, Any]:
        return {
            "record_key": record_key,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "precedents": {
                "cited": self.precedents_cited,
                "seen": self.precedents_seen,
            },
            "tool_calls": self.tool_calls,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
        }


def prompt_sha() -> str:
    """Hash of the prompt text, so a run records exactly which prompt produced it."""
    blob = SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(TOOLS, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class HazardAgent:
    def __init__(
        self,
        client: LLMClient,
        index: PrecedentIndex,
        conn: sqlite3.Connection,
        default_k: int = 5,
        allow_network_lookups: bool = True,
    ) -> None:
        self.client = client
        self.index = index
        self.conn = conn
        self.default_k = default_k
        self.allow_network_lookups = allow_network_lookups
        self._http = httpx.Client(timeout=30)

    # -- tool dispatch ----------------------------------------------------

    def _run_tool(
        self, name: str, args: dict[str, Any], example: Example
    ) -> tuple[dict[str, Any], list[dict]]:
        """Execute a tool call. Returns (result, precedents_seen)."""
        if name == "lookup_drug_context":
            desc = args.get("product_description") or example.product_description
            return (
                lookup_drug_context(
                    self.conn,
                    desc,
                    client=self._http,
                    allow_network=self.allow_network_lookups,
                ),
                [],
            )

        if name == "find_precedents":
            query = args.get("query") or example.text()
            try:
                k = int(args.get("k") or self.default_k)
            except (TypeError, ValueError):
                k = self.default_k
            k = max(1, min(k, 10))
            # Structural guard: the example being classified can never be
            # returned as its own precedent, even if it somehow sits in train.
            hits = self.index.search(
                query, k=k, exclude_keys=frozenset({example.record_key})
            )
            seen = [h.to_dict() for h in hits]
            return (
                {
                    "count": len(seen),
                    "note": "All precedents predate the evaluation period.",
                    "precedents": seen,
                },
                seen,
            )

        return ({"error": f"unknown tool {name!r}"}, [])

    # -- message plumbing -------------------------------------------------

    def _append_tool_result(
        self,
        messages: list[dict],
        response_text: str,
        call: dict,
        result: dict,
    ) -> None:
        if self.client.provider.dialect == "anthropic":
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        ([{"type": "text", "text": response_text}] if response_text else [])
                        + [
                            {
                                "type": "tool_use",
                                "id": call["id"],
                                "name": call["name"],
                                "input": call["arguments"],
                            }
                        ]
                    ),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": serialise_tool_result(result),
                        }
                    ],
                }
            )
            return

        messages.append(
            {
                "role": "assistant",
                "content": response_text or None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": serialise_tool_result(result),
            }
        )

    # -- main loop --------------------------------------------------------

    def classify(self, example: Example) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    product_description=example.product_description,
                    reason_for_recall=example.reason_for_recall,
                    recalling_firm=example.recalling_firm or "unknown",
                    distribution_pattern=example.distribution_pattern or "unknown",
                    recall_initiation_date=example.recall_initiation_date or "unknown",
                ),
            },
        ]

        tool_log: list[dict] = []
        precedents_seen: list[dict] = []
        total_ms = 0
        all_cached = True

        for turn in range(MAX_TURNS):
            # On the last turn force the answer, so a model that keeps calling
            # tools still produces a classification instead of timing out.
            force = "submit_classification" if turn == MAX_TURNS - 1 else None
            resp = self.client.chat(messages, tools=TOOLS, tool_choice=force)
            total_ms += resp.latency_ms
            all_cached = all_cached and resp.cache_hit

            if not resp.tool_calls:
                # Some models answer in prose instead of calling the tool.
                parsed = _parse_loose_json(resp.text)
                if parsed:
                    return _result_from_args(
                        parsed, precedents_seen, tool_log, total_ms, all_cached
                    )
                messages.append({"role": "assistant", "content": resp.text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Call submit_classification now with your answer.",
                    }
                )
                continue

            for call in resp.tool_calls:
                if call["name"] == "submit_classification":
                    tool_log.append({"turn": turn, "tool": call["name"], "args": call["arguments"]})
                    return _result_from_args(
                        call["arguments"], precedents_seen, tool_log, total_ms, all_cached
                    )

                result, seen = self._run_tool(call["name"], call["arguments"], example)
                precedents_seen.extend(seen)
                tool_log.append(
                    {
                        "turn": turn,
                        "tool": call["name"],
                        "args": call["arguments"],
                        "result_summary": _summarise(result),
                    }
                )
                self._append_tool_result(messages, resp.text, call, result)

        return AgentResult(
            classification="Class II",
            confidence=0.0,
            reasoning="Agent did not submit a classification within the turn limit.",
            precedents_cited=[],
            precedents_seen=precedents_seen,
            tool_calls=tool_log,
            latency_ms=total_ms,
            cache_hit=all_cached,
            error="no_submission",
        )

    def close(self) -> None:
        self._http.close()


TOOL_RESULT_BUDGET = 6000


def serialise_tool_result(result: dict[str, Any], budget: int = TOOL_RESULT_BUDGET) -> str:
    """Serialise a tool result to JSON that fits in `budget` and still parses.

    Slicing the serialised string would hand the model truncated, invalid JSON,
    which is worse than sending less data. Instead: shorten the long free-text
    fields first, then drop whole precedents from the end until it fits.
    """
    def shrink(obj: dict[str, Any], text_cap: int) -> dict[str, Any]:
        out = dict(obj)
        for field in ("reason_for_recall", "product_description"):
            value = out.get(field)
            if isinstance(value, str) and len(value) > text_cap:
                out[field] = value[:text_cap] + "..."
        return out

    payload = dict(result)
    precedents = payload.get("precedents")

    for cap in (600, 400, 250, 150):
        if not isinstance(precedents, list):
            break
        payload["precedents"] = [shrink(p, cap) for p in precedents]
        blob = json.dumps(payload)
        if len(blob) <= budget:
            return blob

    if isinstance(precedents, list):
        trimmed = [shrink(p, 150) for p in precedents]
        while trimmed:
            trimmed.pop()
            payload["precedents"] = trimmed
            payload["truncated"] = True
            blob = json.dumps(payload)
            if len(blob) <= budget:
                return blob

    blob = json.dumps(payload)
    if len(blob) <= budget:
        return blob
    # Nothing structured left to drop: return valid JSON describing the problem
    # rather than an unparseable fragment.
    return json.dumps({"error": "tool result too large to serialise", "truncated": True})


def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Compact tool result for the stored audit trail."""
    if "precedents" in result:
        return {
            "count": result.get("count"),
            "classes": [p["classification"] for p in result["precedents"]],
            "keys": [p["record_key"] for p in result["precedents"]],
        }
    return {k: v for k, v in result.items() if k in ("found", "query", "reason")}


def _parse_loose_json(text: str) -> dict[str, Any] | None:
    """Recover a classification from prose when a model skips the tool call."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict) and obj.get("classification"):
                return obj
        except json.JSONDecodeError:
            pass
    # Longest label first and anchored on a word boundary: a plain substring
    # test matches "Class I" inside "Class III", which would turn the least
    # severe class into the most severe one.
    for label in sorted(LABEL_CLASSES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", text, re.IGNORECASE):
            return {
                "classification": label,
                "confidence": 0.3,
                "reasoning": text.strip()[:2000],
                "precedents_cited": [],
                "_recovered": True,
            }
    return None


def _normalise_class(value: Any) -> tuple[str, str | None]:
    """Map a model's class string onto a canonical label."""
    if not isinstance(value, str):
        return "Class II", f"non-string classification {value!r}"
    text = value.strip()
    for label in LABEL_CLASSES:
        if text.lower() == label.lower():
            return label, None
    compact = text.lower().replace("class", "").replace(".", "").strip()
    roman = {"i": "Class I", "ii": "Class II", "iii": "Class III",
             "1": "Class I", "2": "Class II", "3": "Class III"}
    if compact in roman:
        return roman[compact], None
    return "Class II", f"unparseable classification {value!r}"


def _result_from_args(
    args: dict[str, Any],
    precedents_seen: list[dict],
    tool_log: list[dict],
    latency_ms: int,
    cache_hit: bool,
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
        precedents_seen=precedents_seen,
        tool_calls=tool_log,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        error=error,
    )
