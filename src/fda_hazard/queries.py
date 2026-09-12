"""Every read the dashboard performs.

The rule: each number rendered in the UI comes from a row selected here. No
metric is computed at render time, and no metric comes from a model. Templates
receive values, never expressions.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .db import LABEL_CLASSES
from .metrics import COST_MATRIX, wilson_interval

BASELINE_SYSTEMS = ("baseline_majority", "baseline_tfidf", "baseline_tfidf_balanced")

SYSTEM_LABELS = {
    "agent": "Agent",
    "baseline_majority": "Baseline: majority class",
    "baseline_tfidf": "Baseline: TF-IDF + LR",
    "baseline_tfidf_balanced": "Baseline: TF-IDF + LR (balanced)",
}


def run_metrics(row: sqlite3.Row) -> dict[str, Any]:
    return json.loads(row["metrics_json"])


def latest(conn: sqlite3.Connection, system: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM eval_runs WHERE system = ? ORDER BY created_at DESC, id DESC"
        " LIMIT 1",
        (system,),
    ).fetchone()


def scoreboard(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Latest run for the agent and each baseline, side by side."""
    out = []
    for system in ("agent", *BASELINE_SYSTEMS):
        row = latest(conn, system)
        if row is None:
            continue
        m = run_metrics(row)
        tp = m["per_class"]["Class I"]["true_positives"]
        support = m["class1_support"]
        lo, hi = wilson_interval(tp, support)
        out.append(
            {
                "system": system,
                "label": SYSTEM_LABELS.get(system, system),
                "run_uid": row["run_uid"],
                "created_at": row["created_at"],
                "model": row["model"],
                "provider": row["provider"],
                "n": row["n_examples"],
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "total_cost": row["total_cost"],
                "mean_cost": row["mean_cost"],
                "class1_recall": row["class1_recall"],
                "class1_precision": row["class1_precision"],
                "class1_missed": row["class1_missed"],
                "class1_support": support,
                "class1_ci": (lo, hi),
                "is_agent": system == "agent",
            }
        )
    return out


def corpus_class_distribution(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT classification AS cls, COUNT(*) AS n
           FROM recalls
           WHERE classification IN (?,?,?)
           GROUP BY cls ORDER BY n DESC""",
        LABEL_CLASSES,
    ).fetchall()
    total = sum(r["n"] for r in rows) or 1
    return [
        {"cls": r["cls"], "n": r["n"], "share": r["n"] / total} for r in rows
    ]


def split_distribution(conn: sqlite3.Connection, cutoff: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, op in (("train", "<"), ("test", ">=")):
        rows = conn.execute(
            f"""SELECT classification AS cls, COUNT(*) AS n
                FROM recalls
                WHERE classification IN (?,?,?)
                  AND recall_initiation_date_iso IS NOT NULL
                  AND recall_initiation_date_iso {op} ?
                GROUP BY cls ORDER BY cls""",
            (*LABEL_CLASSES, cutoff),
        ).fetchall()
        out[name] = {r["cls"]: r["n"] for r in rows}
        out[f"{name}_total"] = sum(r["n"] for r in rows)
    return out


def accuracy_by_year(conn: sqlite3.Connection, run_uid: str) -> list[dict[str, Any]]:
    """Per-year accuracy for one run, joined back to the recall dates."""
    rows = conn.execute(
        """SELECT substr(r.recall_initiation_date_iso, 1, 4) AS yr,
                  COUNT(*) AS n,
                  SUM(p.correct) AS n_correct,
                  SUM(p.cost) AS cost
           FROM predictions p
           JOIN recalls r ON r.record_key = p.record_key
           WHERE p.run_uid = ?
           GROUP BY yr ORDER BY yr""",
        (run_uid,),
    ).fetchall()
    return [
        {
            "year": r["yr"],
            "n": r["n"],
            "accuracy": r["n_correct"] / r["n"] if r["n"] else 0.0,
            "mean_cost": r["cost"] / r["n"] if r["n"] else 0.0,
        }
        for r in rows
    ]


def all_runs(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM eval_runs ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def run_by_uid(conn: sqlite3.Connection, run_uid: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM eval_runs WHERE run_uid = ?", (run_uid,)
    ).fetchone()


def previous_run(conn: sqlite3.Connection, row: sqlite3.Row) -> sqlite3.Row | None:
    """The run immediately preceding this one for the same system."""
    return conn.execute(
        """SELECT * FROM eval_runs
           WHERE system = ? AND (created_at < ? OR (created_at = ? AND id < ?))
           ORDER BY created_at DESC, id DESC LIMIT 1""",
        (row["system"], row["created_at"], row["created_at"], row["id"]),
    ).fetchone()


def run_diff(
    conn: sqlite3.Connection, row: sqlite3.Row
) -> dict[str, Any] | None:
    """Change in each headline metric against the previous run of the same system."""
    prev = previous_run(conn, row)
    if prev is None:
        return None
    fields = (
        ("accuracy", True),
        ("macro_f1", True),
        ("class1_recall", True),
        ("mean_cost", False),  # lower is better
        ("total_cost", False),
    )
    deltas = {}
    for name, higher_is_better in fields:
        now, before = row[name], prev[name]
        if now is None or before is None:
            continue
        delta = now - before
        deltas[name] = {
            "now": now,
            "before": before,
            "delta": delta,
            "better": (delta > 0) if higher_is_better else (delta < 0),
            "unchanged": abs(delta) < 1e-12,
        }
    return {
        "prev_uid": prev["run_uid"],
        "prev_created_at": prev["created_at"],
        # Either signal counts: the text can change under a fixed version
        # label, and the label can be bumped without the text changing.
        "prompt_changed": (
            prev["prompt_sha"] != row["prompt_sha"]
            or prev["prompt_version"] != row["prompt_version"]
        ),
        "model_changed": prev["model"] != row["model"],
        "sha_changed": prev["git_sha"] != row["git_sha"],
        "prev_prompt_version": prev["prompt_version"],
        "prev_model": prev["model"],
        "deltas": deltas,
    }


def errors(
    conn: sqlite3.Connection,
    run_uid: str,
    sort: str = "cost",
    truth: str | None = None,
    predicted: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Every misclassification in a run, with the full audit trail."""
    order = {
        "cost": "p.cost DESC, r.recall_initiation_date_iso DESC",
        "confidence": "p.confidence DESC, p.cost DESC",
        "date": "r.recall_initiation_date_iso DESC",
        "truth": "p.truth, p.cost DESC",
    }.get(sort, "p.cost DESC")

    where = ["p.run_uid = ?", "p.correct = 0"]
    params: list[Any] = [run_uid]
    if truth:
        where.append("p.truth = ?")
        params.append(truth)
    if predicted:
        where.append("p.predicted = ?")
        params.append(predicted)
    params.append(limit)

    rows = conn.execute(
        f"""SELECT p.*, r.product_description, r.reason_for_recall,
                   r.recalling_firm, r.recall_initiation_date_iso AS init_date,
                   r.distribution_pattern, r.recall_number
            FROM predictions p
            JOIN recalls r ON r.record_key = p.record_key
            WHERE {" AND ".join(where)}
            ORDER BY {order} LIMIT ?""",
        params,
    ).fetchall()
    return [_error_row(r) for r in rows]


def _error_row(r: sqlite3.Row) -> dict[str, Any]:
    precedents = json.loads(r["precedents_json"]) if r["precedents_json"] else {}
    seen = precedents.get("seen") or []
    cited = set(precedents.get("cited") or [])
    return {
        "record_key": r["record_key"],
        "recall_number": r["recall_number"],
        "truth": r["truth"],
        "predicted": r["predicted"],
        "cost": r["cost"],
        "confidence": r["confidence"],
        "reasoning": r["reasoning"],
        "product_description": r["product_description"],
        "reason_for_recall": r["reason_for_recall"],
        "recalling_firm": r["recalling_firm"],
        "init_date": r["init_date"],
        "distribution_pattern": r["distribution_pattern"],
        "precedents": [
            {**p, "cited": p.get("record_key") in cited} for p in seen
        ],
        "n_cited": len(cited),
        "tool_calls": json.loads(r["tool_calls_json"]) if r["tool_calls_json"] else [],
        "is_class1_miss": r["truth"] == "Class I",
    }


def error_summary(conn: sqlite3.Connection, run_uid: str) -> dict[str, Any]:
    rows = conn.execute(
        """SELECT truth, predicted, COUNT(*) AS n, SUM(cost) AS cost
           FROM predictions WHERE run_uid = ? AND correct = 0
           GROUP BY truth, predicted ORDER BY cost DESC""",
        (run_uid,),
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) AS n, SUM(cost) AS cost FROM predictions WHERE run_uid = ?",
        (run_uid,),
    ).fetchone()
    return {
        "buckets": [
            {
                "truth": r["truth"],
                "predicted": r["predicted"],
                "n": r["n"],
                "cost": r["cost"],
                "unit_cost": COST_MATRIX[r["truth"]][r["predicted"]],
            }
            for r in rows
        ],
        "n_errors": sum(r["n"] for r in rows),
        "n_total": total["n"] if total else 0,
        "total_cost": total["cost"] if total else 0,
    }
