"""Persist evaluation runs and their per-example predictions.

Everything the dashboard shows traces back to a row written here. A metric is
never recomputed at render time and never comes from a model.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from .metrics import Metrics, cost_of
from .splits import Example


def git_sha() -> tuple[str | None, bool]:
    """Current commit and whether the working tree is dirty.

    A dirty tree means the recorded SHA does not fully describe the code that
    produced the score, so it is stored alongside rather than silently ignored.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if sha.returncode != 0:
            return None, False
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        return sha.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, False


def record_run(
    conn: sqlite3.Connection,
    *,
    system: str,
    metrics: Metrics,
    examples: Sequence[Example],
    predictions: Sequence[str],
    cutoff: str,
    split: str = "test",
    prompt_version: str | None = None,
    prompt_sha: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    config: dict[str, Any] | None = None,
    duration_s: float | None = None,
    details: Sequence[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> str:
    """Write one eval run plus a row per scored example. Returns the run uid."""
    if len(examples) != len(predictions):
        raise ValueError(
            f"{len(examples)} examples but {len(predictions)} predictions"
        )
    if metrics.n != len(examples):
        raise ValueError(
            f"metrics scored {metrics.n} but {len(examples)} examples were passed"
        )

    uid = uuid.uuid4().hex[:12]
    sha, dirty = git_sha()
    conn.execute(
        """INSERT INTO eval_runs (
               run_uid, created_at, system, git_sha, git_dirty, prompt_version,
               prompt_sha, model, provider, split, split_cutoff, n_examples,
               accuracy, macro_f1, total_cost, mean_cost, class1_recall,
               class1_precision, class1_f1, class1_missed, metrics_json,
               config_json, duration_s, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            uid,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            system,
            sha,
            int(dirty),
            prompt_version,
            prompt_sha,
            model,
            provider,
            split,
            cutoff,
            len(examples),
            metrics.accuracy,
            metrics.macro_f1,
            metrics.total_cost,
            metrics.mean_cost,
            metrics.class1_recall,
            metrics.class1_precision,
            metrics.class1_f1,
            metrics.class1_missed,
            json.dumps(metrics.to_dict()),
            json.dumps(config or {}),
            duration_s,
            notes,
        ),
    )

    detail_by_key: dict[str, dict[str, Any]] = {}
    if details:
        detail_by_key = {d["record_key"]: d for d in details}

    rows = []
    for ex, pred in zip(examples, predictions, strict=True):
        d = detail_by_key.get(ex.record_key, {})
        rows.append(
            (
                uid,
                ex.record_key,
                ex.classification,
                pred,
                int(pred == ex.classification),
                cost_of(ex.classification, pred),
                d.get("confidence"),
                d.get("reasoning"),
                json.dumps(d["precedents"]) if d.get("precedents") is not None else None,
                json.dumps(d["tool_calls"]) if d.get("tool_calls") is not None else None,
                d.get("latency_ms"),
                int(bool(d.get("cache_hit"))),
            )
        )
    conn.executemany(
        """INSERT INTO predictions (
               run_uid, record_key, truth, predicted, correct, cost, confidence,
               reasoning, precedents_json, tool_calls_json, latency_ms, cache_hit)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return uid


def latest_run(conn: sqlite3.Connection, system: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM eval_runs WHERE system = ? ORDER BY created_at DESC, id DESC "
        "LIMIT 1",
        (system,),
    ).fetchone()


def all_runs(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM eval_runs ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)
    ).fetchall()
