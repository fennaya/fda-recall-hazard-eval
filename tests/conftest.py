"""Shared fixtures: a small in-memory corpus that exercises the split and
retrieval invariants without touching the real 18k-row database."""

from __future__ import annotations

import json
import sqlite3

import pytest

from fda_hazard.db import SCHEMA


def make_record(
    record_key: str,
    classification: str,
    init_date: str,
    event_id: str,
    reason: str = "Subpotent product",
    description: str = "Widget Tablets 10 mg",
    firm: str = "Acme Pharma",
) -> dict:
    return {
        "record_key": record_key,
        "recall_number": record_key,
        "event_id": event_id,
        "classification": classification,
        "recall_initiation_date_iso": init_date,
        "reason_for_recall": reason,
        "product_description": description,
        "recalling_firm": firm,
        "distribution_pattern": "Nationwide",
        "voluntary_mandated": "Voluntary: Firm initiated",
        "status": "Terminated",
    }


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    yield c
    c.close()


def insert(conn: sqlite3.Connection, records: list[dict]) -> None:
    for rec in records:
        cols = list(rec) + ["raw_json", "ingested_at"]
        vals = list(rec.values()) + [json.dumps(rec), "2026-01-01T00:00:00+00:00"]
        conn.execute(
            f"INSERT INTO recalls ({','.join(cols)}) "
            f"VALUES ({','.join('?' for _ in cols)})",
            vals,
        )
    conn.commit()


@pytest.fixture
def corpus(conn: sqlite3.Connection) -> sqlite3.Connection:
    """A clean corpus: events sit entirely on one side of the 2025-01-01 cut."""
    records = [
        make_record("A-1", "Class I", "2023-05-01", "e100", "Contains undeclared penicillin"),
        make_record("A-2", "Class II", "2023-06-01", "e101", "Subpotent, out of specification"),
        make_record("A-3", "Class III", "2023-07-01", "e102", "Label typo on carton"),
        make_record("A-4", "Class II", "2024-02-01", "e103", "Dissolution failure at 12 months"),
        make_record("A-5", "Class I", "2024-09-01", "e104", "Sterility failure in injectable"),
        make_record("B-1", "Class II", "2025-03-01", "e200", "Subpotent, out of specification"),
        make_record("B-2", "Class I", "2025-04-01", "e201", "Contains undeclared penicillin"),
        make_record("B-3", "Class III", "2025-05-01", "e202", "Label typo on carton"),
    ]
    insert(conn, records)
    return conn
