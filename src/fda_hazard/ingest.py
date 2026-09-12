"""Pull the full openFDA drug enforcement corpus into SQLite.

Primary path is the bulk download from open.fda.gov/data/downloads, which
ships the whole corpus as one zipped JSON and avoids the API's 26k skip
ceiling entirely. The date-sliced pagination fallback exists for when the
bulk export is stale or unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from typing import Any, Iterator

import requests

from .db import (
    DATE_FIELDS,
    DB_PATH,
    LABEL_CLASSES,
    SCALAR_FIELDS,
    connect,
    to_iso,
)

API_URL = "https://api.fda.gov/drug/enforcement.json"
DOWNLOAD_MANIFEST = "https://api.fda.gov/download.json"

# The API refuses skip values at/above 26000, so each date slice must stay
# comfortably under that. 1000 keeps individual responses small.
PAGE_LIMIT = 1000
SKIP_CEILING = 25000


def api_key() -> str | None:
    return os.environ.get("OPENFDA_API_KEY") or None


def _params(extra: dict[str, Any]) -> dict[str, Any]:
    p = dict(extra)
    key = api_key()
    if key:
        p["api_key"] = key
    return p


def api_total(session: requests.Session) -> int:
    """Total record count the API reports, used to verify ingest completeness."""
    r = session.get(API_URL, params=_params({"limit": 1}), timeout=60)
    r.raise_for_status()
    return int(r.json()["meta"]["results"]["total"])


# --------------------------------------------------------------------------
# Source A: bulk download (preferred)
# --------------------------------------------------------------------------


def bulk_partitions(session: requests.Session) -> tuple[list[str], str | None]:
    r = session.get(DOWNLOAD_MANIFEST, timeout=60)
    r.raise_for_status()
    enf = r.json()["results"]["drug"]["enforcement"]
    return [p["file"] for p in enf["partitions"]], enf.get("export_date")


def iter_bulk(session: requests.Session) -> Iterator[dict[str, Any]]:
    urls, export_date = bulk_partitions(session)
    print(f"[bulk] {len(urls)} partition(s), export_date={export_date}", flush=True)
    for url in urls:
        print(f"[bulk] downloading {url}", flush=True)
        resp = session.get(url, timeout=600)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if not name.endswith(".json"):
                    continue
                with zf.open(name) as fh:
                    payload = json.load(fh)
                yield from payload.get("results", [])


# --------------------------------------------------------------------------
# Source B: date-sliced pagination (fallback)
# --------------------------------------------------------------------------


def iter_paginated(
    session: requests.Session, start_year: int = 2000, end_year: int | None = None
) -> Iterator[dict[str, Any]]:
    """Page the API one year at a time so `skip` never nears the cap.

    Slicing on report_date keeps each window's count well under SKIP_CEILING.
    If a single year ever exceeded it we would be silently truncating, so that
    case raises instead.
    """
    end_year = end_year or datetime.now(timezone.utc).year
    for year in range(start_year, end_year + 1):
        search = f"report_date:[{year}0101+TO+{year}1231]"
        skip = 0
        year_total: int | None = None
        while True:
            if skip >= SKIP_CEILING:
                raise RuntimeError(
                    f"year {year} exceeds the skip ceiling ({year_total} records); "
                    "split this window into months"
                )
            r = session.get(
                API_URL,
                params=_params({"search": search, "limit": PAGE_LIMIT, "skip": skip}),
                timeout=120,
            )
            if r.status_code == 404:  # openFDA's "no matches" response
                break
            r.raise_for_status()
            body = r.json()
            if year_total is None:
                year_total = int(body["meta"]["results"]["total"])
                print(f"[page] {year}: {year_total} records", flush=True)
            results = body.get("results", [])
            if not results:
                break
            yield from results
            skip += len(results)
            if skip >= (year_total or 0):
                break


# --------------------------------------------------------------------------
# Parse + persist
# --------------------------------------------------------------------------

_COLUMNS = (
    ["record_key"]
    + list(SCALAR_FIELDS)
    + [f"{f}_iso" for f in DATE_FIELDS]
    + ["openfda_json", "raw_json", "ingested_at"]
)
_INSERT = (
    f"INSERT INTO recalls ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)}) "
    "ON CONFLICT(record_key) DO UPDATE SET "
    + ", ".join(f"{c}=excluded.{c}" for c in _COLUMNS if c != "record_key")
)


def record_key(rec: dict[str, Any]) -> str:
    """Stable primary key for a record.

    Uses recall_number when present. openFDA ships at least one otherwise
    complete record without one, so those fall back to a deterministic hash of
    the record's identifying content -- dropping them would be data loss, and a
    content hash keeps re-ingests idempotent.
    """
    number = _blank_to_none(rec.get("recall_number"))
    if number:
        return number
    # Placeholders ("", "N/A") are not identifiers: two such records would
    # collide on the PK and silently overwrite each other.
    ident = "|".join(
        str(rec.get(f) or "")
        for f in ("event_id", "recalling_firm", "product_description", "report_date")
    )
    return "synthetic:" + hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16]


def _blank_to_none(value: Any) -> Any:
    """openFDA uses "", " " and "N/A" interchangeably with an absent field.

    Normalising them to NULL keeps `IS NULL` checks honest and lets the partial
    unique index on recall_number actually do its job.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return None if s == "" or s.upper() == "N/A" else s
    return value


def parse_record(rec: dict[str, Any], now: str) -> tuple[Any, ...]:
    """Flatten one API record into a row, preserving the original JSON."""
    row: list[Any] = [record_key(rec)]
    row += [_blank_to_none(rec.get(f)) for f in SCALAR_FIELDS]
    row += [to_iso(rec.get(f)) for f in DATE_FIELDS]
    openfda = rec.get("openfda") or {}
    row.append(json.dumps(openfda, sort_keys=True) if openfda else None)
    row.append(json.dumps(rec, sort_keys=True))
    row.append(now)
    return tuple(row)


def ingest(
    conn: sqlite3.Connection, records: Iterator[dict[str, Any]]
) -> tuple[int, int]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen = 0
    synthetic = 0
    batch: list[tuple[Any, ...]] = []
    for rec in records:
        seen += 1
        if not _blank_to_none(rec.get("recall_number")):
            synthetic += 1
        batch.append(parse_record(rec, now))
        if len(batch) >= 2000:
            conn.executemany(_INSERT, batch)
            conn.commit()
            batch.clear()
    if batch:
        conn.executemany(_INSERT, batch)
        conn.commit()
    if synthetic:
        print(
            f"[note] {synthetic} record(s) had no recall_number; kept under a "
            "synthetic content-hash key",
            flush=True,
        )
    written = conn.execute("SELECT COUNT(*) FROM recalls").fetchone()[0]
    return seen, written


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def class_distribution(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT COALESCE(NULLIF(TRIM(classification), ''), '(missing)') AS cls,
               COUNT(*) AS n
        FROM recalls
        GROUP BY cls
        ORDER BY n DESC
        """
    ).fetchall()


def print_distribution(conn: sqlite3.Connection) -> None:
    rows = class_distribution(conn)
    total = sum(r["n"] for r in rows)
    if not total:
        print("no rows")
        return
    print(f"\nClass distribution (n={total:,})")
    print(f"{'class':<12}{'count':>9}{'share':>9}")
    print("-" * 30)
    for r in rows:
        print(f"{r['cls']:<12}{r['n']:>9,}{r['n'] / total:>8.1%}")
    counts = {r["cls"]: r["n"] for r in rows}
    # Imbalance is only meaningful across the three real hazard classes;
    # unclassified/missing rows are excluded from modelling entirely.
    labeled = {c: counts[c] for c in LABEL_CLASSES if c in counts}
    unlabeled = sum(n for c, n in counts.items() if c not in LABEL_CLASSES)
    if labeled:
        lab_total = sum(labeled.values())
        print(f"\nLabeled subset used for modelling: n={lab_total:,}")
        print(
            f"  imbalance (largest:smallest) = "
            f"{max(labeled.values()) / min(labeled.values()):.1f}:1"
        )
        for a, b in (("Class II", "Class I"), ("Class II", "Class III")):
            if labeled.get(a) and labeled.get(b):
                print(f"  {a} / {b} = {labeled[a] / labeled[b]:.1f}x")
        print(f"  majority-class baseline would score {max(labeled.values()) / lab_total:.1%}")
    if unlabeled:
        print(f"  excluded as unlabeled: {unlabeled:,}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument(
        "--source",
        choices=("bulk", "paginate", "auto"),
        default="auto",
        help="auto tries bulk, falls back to date-sliced pagination",
    )
    args = ap.parse_args(argv)

    if not api_key():
        print("[warn] OPENFDA_API_KEY unset: rate limited", flush=True)

    conn = connect(args.db)
    session = requests.Session()
    session.headers["User-Agent"] = "fda-recall-hazard/0.1 (research harness)"

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    total = api_total(session)
    print(f"[api] reports {total:,} total records", flush=True)

    export_date: str | None = None
    source_used = args.source
    if args.source in ("bulk", "auto"):
        try:
            _, export_date = bulk_partitions(session)
            seen, written = ingest(conn, iter_bulk(session))
            source_used = "bulk"
        except Exception as exc:  # noqa: BLE001 - the fallback is the point
            if args.source == "bulk":
                raise
            print(f"[bulk] failed ({exc}); falling back to pagination", flush=True)
            seen, written = ingest(conn, iter_paginated(session))
            source_used = "paginate"
    else:
        seen, written = ingest(conn, iter_paginated(session))

    conn.execute(
        """INSERT INTO ingest_runs
           (started_at, finished_at, source, records_seen, records_written,
            api_total, export_date, notes)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            started,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source_used,
            seen,
            written,
            total,
            export_date,
            None,
        ),
    )
    conn.commit()

    print(f"\n[done] source={source_used} seen={seen:,} rows_in_db={written:,}")
    if written < total:
        print(f"[warn] db has {total - written:,} fewer rows than the API total")
    print_distribution(conn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
