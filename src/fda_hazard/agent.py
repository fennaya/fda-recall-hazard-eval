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

PROMPT_VERSION = "v2"

# FDA's own definitions, quoted so the agent reasons against the actual
# standard rather than a vibe about severity. 21 CFR 7.3(m).
#
# Kept deliberately tight: this text, the tool schemas, and every tool result
# are resent in full on every turn of the conversation, so their size is
# multiplied by turn count. On a rate-limited free tier that multiplication is
# what actually determines whether a case finishes or stalls, not the prompt's
# absolute length in isolation.
SYSTEM_PROMPT = """\
You classify FDA drug recalls into FDA's hazard classes (21 CFR 7.3(m)), \
replicating a decision FDA already made.

Class I: reasonable probability of serious adverse health consequences or death.
Class II: temporary/reversible consequences, or serious consequences are remote.
Class III: not likely to cause adverse health consequences.

Method: identify the actual physical/chemical defect, reason about its clinical \
consequence for the patient (what the drug treats, route, who takes it, effect \
of a wrong/absent/contaminated/misidentified dose), weigh any precedent \
(close precedent is strong evidence, distant precedent is weak), then decide.

Calibration: sterility failures, wrong-drug/cross-contamination, undeclared \
allergens, and subpotent/superpotent life-supporting drugs skew Class I. Most \
stability/dissolution/impurity/potency issues in oral solids skew Class II \
(the most common class). Labeling errors not affecting identity/strength/safe \
use, and cosmetic/packaging defects, skew Class III. Voluntary vs. mandated and \
distribution breadth are not severity signals.

Be decisive: one class, not a hedge. Confidence should be honest, not inflated."""

USER_TEMPLATE = """\
Classify this drug recall.

PRODUCT DESCRIPTION: {product_description}

REASON FOR RECALL: {reason_for_recall}

firm: {recalling_firm} | distribution: {distribution_pattern} | initiated: {recall_initiation_date}

Use tools only if they would change your answer, then call submit_classification."""


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# Fewer turns and fewer precedents per call bound how much the conversation can
# grow: every prior turn (system prompt, tool schemas, and every tool result) is
# resent on each subsequent call, so cost compounds with both knobs.
MAX_TURNS = 4
MAX_PRECEDENTS_PER_CALL = 5

# Tool descriptions are kept short for the same reason as the system prompt:
# they are resent on every turn.
TOOLS = [
    _tool(
        "lookup_drug_context",
        "Look up the drug in openFDA label/NDC data (indication, route, boxed "
        "warning). Often finds nothing - that's normal, proceed on the recall text.",
        {
            "type": "object",
            "properties": {
                "product_description": {
                    "type": "string",
                    "description": "Product description to resolve a drug from.",
                }
            },
            "required": ["product_description"],
        },
    ),
    _tool(
        "find_precedents",
        "Retrieve similar past recalls FDA already classified (train-split only, "
        "predates this evaluation). Use to check how FDA has treated this defect before.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Defect/product type to match, not the firm name.",
                },
                "k": {
                    "type": "integer",
                    "description": f"Precedents to return (1-{MAX_PRECEDENTS_PER_CALL}).",
                },
            },
            "required": ["query"],
        },
    ),
    _tool(
        "submit_classification",
        "Submit your final hazard classification. Call exactly once to finish.",
        {
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
    ),
]


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
    #: 'tool_loop' (this class) or 'single_call' (agent_single.py). Carried
    #: through to record_run() so a run can never silently blend the two.
    architecture: str = "tool_loop"

    def to_detail(self, record_key: str) -> dict[str, Any]:
        return {
            "record_key": record_key,
            "architecture": self.architecture,
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
            k = max(1, min(k, MAX_PRECEDENTS_PER_CALL))
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
        """Classify one example. Never raises.

        A run scores 1,275+ cases over many hours; a provider-side quirk on any
        single one of them (a malformed response, a model that ignores a forced
        tool_choice and gets hard-rejected by the API, anything) must not take
        the whole evaluation down with it. Every failure mode becomes a scored
        low-confidence Class II prediction with the error recorded, the same way
        an exhausted turn limit already does, so the case is still measured
        rather than silently missing from the results.
        """
        try:
            return self._classify_inner(example)
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch everything
            return AgentResult(
                classification="Class II",
                confidence=0.0,
                reasoning=f"Agent raised {type(exc).__name__} before submitting: {exc}",
                precedents_cited=[],
                error=f"exception:{type(exc).__name__}",
            )

    def _classify_inner(self, example: Example) -> AgentResult:
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
            try:
                resp = self.client.chat(messages, tools=TOOLS, tool_choice=force)
            except RuntimeError as exc:
                if force and _is_tool_choice_mismatch(exc):
                    # The model tried a different tool anyway and the provider
                    # hard-rejected the mismatch instead of coercing it. Retrying
                    # unforced lets the model's actual intended call go through
                    # (handled by the normal tool-dispatch path below) instead of
                    # failing this case outright.
                    resp = self.client.chat(messages, tools=TOOLS, tool_choice=None)
                else:
                    raise
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


# Small on purpose: this text is resent in full on every subsequent turn of the
# conversation, so it is the single biggest lever on tokens-per-case on a
# rate-limited provider.
TOOL_RESULT_BUDGET = 2000


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


def _is_tool_choice_mismatch(exc: RuntimeError) -> bool:
    """True if a provider rejected a response because the model called a
    different tool than the one an earlier turn forced via tool_choice.

    Observed on Groq/gpt-oss-120b: forcing tool_choice does not always stop the
    model from attempting a different tool, and the API hard-rejects the
    mismatch (400 tool_use_failed) rather than coercing or ignoring it.
    """
    text = str(exc).lower()
    return "tool_use_failed" in text or "does not match request.tool_choice" in text


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
