"""Look up drug context (indication, route, population) from openFDA.

The embedded `openfda` block is populated for only 3,261 of 17,938 recalls
(18%), so this resolves context from the product description text against the
label and NDC endpoints instead. "Nothing found" is a normal, frequent outcome
and is reported as such rather than raised -- the agent is expected to reason
without it most of the time.

Results are cached in SQLite, so an eval re-run makes no network calls.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from typing import Any

import httpx

from .ingest import api_key

LABEL_URL = "https://api.fda.gov/drug/label.json"
NDC_URL = "https://api.fda.gov/drug/ndc.json"

# Dosage forms, units and marketing boilerplate that pollute a drug-name guess.
_NOISE = re.compile(
    r"\b(tablets?|capsules?|injection|solution|suspension|cream|ointment|gel|"
    r"syrup|elixir|powder|vials?|ampules?|bottles?|mg|mcg|ml|g|kg|iu|usp|nf|"
    r"rx only|otc|sterile|oral|topical|ophthalmic|intravenous|lot|exp|ndc|"
    r"unit dose|per|each|count|ct)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^A-Za-z0-9\s\-]")


def extract_drug_name(product_description: str, max_terms: int = 4) -> str:
    """Best-effort drug name from a free-text product description.

    These strings are written by firms, not to a schema: the name is usually the
    first few words before a strength or dosage form, so we strip units and
    boilerplate and keep the leading terms.
    """
    head = product_description.split(",")[0]
    head = _PUNCT.sub(" ", head)
    head = re.sub(r"\b\d+(\.\d+)?\b", " ", head)
    head = _NOISE.sub(" ", head)
    terms = [t for t in head.split() if len(t) > 2]
    return " ".join(terms[:max_terms]).strip()


def _cache_get(conn: sqlite3.Connection, key: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT result_json FROM drug_context_cache WHERE query_hash = ?", (key,)
    ).fetchone()
    return json.loads(row["result_json"]) if row else None


def _cache_put(
    conn: sqlite3.Connection, key: str, query: str, result: dict[str, Any]
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO drug_context_cache
           (query_hash, query, created_at, result_json)
           VALUES (?, ?, datetime('now'), ?)""",
        (key, query, json.dumps(result)),
    )
    conn.commit()


def _first(value: Any, limit: int = 600) -> str | None:
    """openFDA returns most label fields as single-element lists of long text."""
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    text = " ".join(str(value).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _query_label(
    client: httpx.Client, name: str, key: str | None
) -> dict[str, Any] | None:
    params: dict[str, Any] = {
        "search": f'openfda.generic_name:"{name}" OR openfda.brand_name:"{name}"',
        "limit": 1,
    }
    if key:
        params["api_key"] = key
    r = client.get(LABEL_URL, params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    rec = results[0]
    of = rec.get("openfda") or {}
    return {
        "source": "drug/label",
        "brand_name": _first(of.get("brand_name"), 120),
        "generic_name": _first(of.get("generic_name"), 120),
        "route": of.get("route"),
        "indications_and_usage": _first(rec.get("indications_and_usage")),
        "warnings": _first(rec.get("boxed_warning") or rec.get("warnings")),
        "has_boxed_warning": bool(rec.get("boxed_warning")),
        "pediatric_use": _first(rec.get("pediatric_use"), 300),
        "pregnancy": _first(rec.get("pregnancy"), 300),
    }


def _query_ndc(
    client: httpx.Client, name: str, key: str | None
) -> dict[str, Any] | None:
    params: dict[str, Any] = {
        "search": f'generic_name:"{name}" OR brand_name:"{name}"',
        "limit": 1,
    }
    if key:
        params["api_key"] = key
    r = client.get(NDC_URL, params=params, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    rec = results[0]
    return {
        "source": "drug/ndc",
        "generic_name": rec.get("generic_name"),
        "brand_name": rec.get("brand_name"),
        "dosage_form": rec.get("dosage_form"),
        "route": rec.get("route"),
        "marketing_category": rec.get("marketing_category"),
        "product_type": rec.get("product_type"),
        "pharm_class": (rec.get("pharm_class") or [])[:4],
        "dea_schedule": rec.get("dea_schedule"),
    }


def lookup(
    conn: sqlite3.Connection,
    product_description: str,
    client: httpx.Client | None = None,
    allow_network: bool = True,
) -> dict[str, Any]:
    """Resolve drug context for a product description.

    Always returns a dict. `found` is False when openFDA has nothing, which is
    the common case and not an error.
    """
    name = extract_drug_name(product_description)
    if not name:
        return {"found": False, "reason": "no drug name could be parsed", "query": ""}

    key = hashlib.sha256(name.lower().encode("utf-8")).hexdigest()[:32]
    cached = _cache_get(conn, key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    if not allow_network:
        return {"found": False, "reason": "network disabled", "query": name}

    owns_client = client is None
    client = client or httpx.Client(timeout=30)
    api = api_key()
    result: dict[str, Any] = {"query": name, "found": False}
    try:
        label = _query_label(client, name, api)
        ndc = _query_ndc(client, name, api)
        if label or ndc:
            result = {
                "query": name,
                "found": True,
                "label": label,
                "ndc": ndc,
            }
        else:
            result = {"query": name, "found": False, "reason": "no openFDA match"}
    except (httpx.HTTPError, ValueError) as exc:
        # A lookup failure must not fail the classification; the agent is
        # expected to proceed on the recall text alone.
        result = {"query": name, "found": False, "reason": f"lookup error: {exc}"}
    finally:
        if owns_client:
            client.close()

    _cache_put(conn, key, name, result)
    result["cache_hit"] = False
    return result
