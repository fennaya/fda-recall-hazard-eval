"""SQLite schema and connection helpers for the FDA recall corpus."""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "recalls.db"

# Field names verified against a live sample record from
# https://api.fda.gov/drug/enforcement.json (see README).
# Every scalar field the endpoint returns gets its own column; the untouched
# record is also kept in `raw_json` so nothing is lost.
SCALAR_FIELDS: tuple[str, ...] = (
    "recall_number",
    "event_id",
    "status",
    "classification",
    "product_type",
    "recalling_firm",
    "address_1",
    "address_2",
    "city",
    "state",
    "postal_code",
    "country",
    "voluntary_mandated",
    "initial_firm_notification",
    "distribution_pattern",
    "product_description",
    "product_quantity",
    "reason_for_recall",
    "code_info",
    "recall_initiation_date",
    "center_classification_date",
    "termination_date",
    "report_date",
)

# Dates arrive as YYYYMMDD strings. We keep the raw string and add an ISO
# column so SQLite date comparisons (and the time split) are lexicographic.
DATE_FIELDS: tuple[str, ...] = (
    "recall_initiation_date",
    "center_classification_date",
    "termination_date",
    "report_date",
)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS recalls (
    -- recall_number is the natural key but openFDA ships at least one record
    -- without it, so the PK is a surrogate (see ingest.record_key).
    record_key TEXT PRIMARY KEY,
    {",\n    ".join(f"{f} TEXT" for f in SCALAR_FIELDS)},
    {",\n    ".join(f"{f}_iso TEXT" for f in DATE_FIELDS)},
    openfda_json TEXT,
    raw_json TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_recalls_number
    ON recalls(recall_number) WHERE recall_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_recalls_class ON recalls(classification);
CREATE INDEX IF NOT EXISTS idx_recalls_event ON recalls(event_id);
CREATE INDEX IF NOT EXISTS idx_recalls_init_iso ON recalls(recall_initiation_date_iso);
CREATE INDEX IF NOT EXISTS idx_recalls_firm ON recalls(recalling_firm);

CREATE TABLE IF NOT EXISTS ingest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    source TEXT NOT NULL,
    records_seen INTEGER,
    records_written INTEGER,
    api_total INTEGER,
    export_date TEXT,
    notes TEXT
);

-- Every evaluation, agent or baseline, lands here. The dashboard reads only
-- these tables: no number shown in the UI is computed anywhere else.
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uid TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    system TEXT NOT NULL,              -- 'agent' | 'baseline_majority' | 'baseline_tfidf'
    -- 'tool_loop' (model decides when to call find_precedents/lookup_drug_context)
    -- or 'single_call' (retrieval done in Python, one structured-output call).
    -- Two different systems; a score must never blend predictions from both.
    architecture TEXT NOT NULL DEFAULT 'tool_loop',
    git_sha TEXT,
    git_dirty INTEGER NOT NULL DEFAULT 0,
    prompt_version TEXT,
    prompt_sha TEXT,
    model TEXT,
    provider TEXT,
    split TEXT NOT NULL,
    split_cutoff TEXT NOT NULL,
    n_examples INTEGER NOT NULL,
    accuracy REAL,
    macro_f1 REAL,
    total_cost REAL,
    mean_cost REAL,
    class1_recall REAL,
    class1_precision REAL,
    class1_f1 REAL,
    class1_missed INTEGER,
    metrics_json TEXT NOT NULL,        -- full metric payload incl. confusion matrix
    config_json TEXT,
    duration_s REAL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_system ON eval_runs(system, created_at);

-- One row per scored example. The Errors page is a filter over this table.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uid TEXT NOT NULL REFERENCES eval_runs(run_uid) ON DELETE CASCADE,
    record_key TEXT NOT NULL REFERENCES recalls(record_key),
    architecture TEXT NOT NULL DEFAULT 'tool_loop',
    truth TEXT NOT NULL,
    predicted TEXT NOT NULL,
    correct INTEGER NOT NULL,
    cost REAL NOT NULL,
    confidence REAL,
    reasoning TEXT,
    precedents_json TEXT,
    tool_calls_json TEXT,
    latency_ms INTEGER,
    cache_hit INTEGER NOT NULL DEFAULT 0,
    UNIQUE(run_uid, record_key)
);

CREATE INDEX IF NOT EXISTS idx_pred_run ON predictions(run_uid);
CREATE INDEX IF NOT EXISTS idx_pred_cost ON predictions(run_uid, cost DESC);
CREATE INDEX IF NOT EXISTS idx_pred_wrong ON predictions(run_uid, correct);

-- Model responses keyed by a hash of everything that could change the output,
-- so re-running an unchanged eval costs nothing.
CREATE TABLE IF NOT EXISTS llm_cache (
    input_hash TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_ms INTEGER
);

-- Cache for openFDA label/NDC context lookups the agent makes as a tool.
CREATE TABLE IF NOT EXISTS drug_context_cache (
    query_hash TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    created_at TEXT NOT NULL,
    result_json TEXT NOT NULL
);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a connection with sane defaults and the schema applied.

    A sqlite3.Connection is not safe to use concurrently from multiple threads.
    The evaluation driver runs one worker thread per concurrent LLM call, and
    gives each its own connection via this function, so no connection is ever
    touched by two threads at once. check_same_thread=False only relaxes
    sqlite3's same-thread assertion for the one case that still crosses a
    thread boundary: the pool closing a worker's connection from the main
    thread after that worker is done with it. busy_timeout lets concurrent
    writes (mostly to llm_cache) queue instead of raising "database is locked".
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent ALTERs for columns added after a database already existed.

    CREATE TABLE IF NOT EXISTS does nothing to a table that already exists, so
    a new column has to be added by hand for any database created before it
    was introduced. Safe to run against a database another connection (e.g. a
    long-running background eval) has open: ADD COLUMN with a DEFAULT doesn't
    rewrite existing rows or touch what an in-flight INSERT is doing.
    """
    for table in ("eval_runs", "predictions"):
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if "architecture" not in cols:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN architecture TEXT NOT NULL "
                "DEFAULT 'tool_loop'"
            )
    conn.commit()


def to_iso(yyyymmdd: str | None) -> str | None:
    """Convert an openFDA YYYYMMDD date string to YYYY-MM-DD, or None."""
    if not yyyymmdd:
        return None
    s = str(yyyymmdd).strip()
    if len(s) != 8 or not s.isdigit():
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


# The three hazard classes FDA assigns. Anything else ("Not Yet Classified",
# missing) is excluded from training and evaluation.
LABEL_CLASSES: tuple[str, ...] = ("Class I", "Class II", "Class III")
