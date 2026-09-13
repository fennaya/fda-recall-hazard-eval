"""Time-based train/test split.

A random split would leak: recalls cluster hard by event_id (17,938 records
share only 4,668 event_ids) and by firm, so the same underlying incident would
land on both sides and the score would be inflated. The split is therefore
strictly temporal, and the no-event_id-spanning property is asserted in code
rather than assumed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, Literal

from .db import LABEL_CLASSES

# Everything strictly before this date is train/dev; everything on or after is
# the held-out test set.
SPLIT_CUTOFF = "2025-01-01"

# The split is on recall_initiation_date: the date the event actually happened.
# report_date and center_classification_date are both downstream of FDA's own
# classification decision, so splitting on them would order examples by when
# the label was assigned rather than by when the event occurred.
SPLIT_DATE_COLUMN = "recall_initiation_date_iso"

Split = Literal["train", "test"]


class LeakageError(RuntimeError):
    """Raised when a split invariant is violated. Never catch this."""


@dataclass(frozen=True)
class Example:
    """One labeled recall, as the model and the baselines both see it."""

    record_key: str
    event_id: str | None
    recalling_firm: str | None
    product_description: str
    reason_for_recall: str
    classification: str
    recall_initiation_date: str | None
    distribution_pattern: str | None
    voluntary_mandated: str | None
    status: str | None

    @property
    def year(self) -> int | None:
        if not self.recall_initiation_date:
            return None
        return int(self.recall_initiation_date[:4])

    def text(self) -> str:
        """The text field used for retrieval and for the TF-IDF baseline."""
        return f"{self.product_description}\n\n{self.reason_for_recall}"


_SELECT = f"""
SELECT record_key, event_id, recalling_firm, product_description,
       reason_for_recall, classification, {SPLIT_DATE_COLUMN} AS init_date,
       distribution_pattern, voluntary_mandated, status
FROM recalls
WHERE classification IN ({",".join("?" for _ in LABEL_CLASSES)})
  AND {SPLIT_DATE_COLUMN} IS NOT NULL
  AND product_description IS NOT NULL
  AND reason_for_recall IS NOT NULL
"""


def _row_to_example(row: sqlite3.Row) -> Example:
    return Example(
        record_key=row["record_key"],
        event_id=row["event_id"],
        recalling_firm=row["recalling_firm"],
        product_description=row["product_description"],
        reason_for_recall=row["reason_for_recall"],
        classification=row["classification"],
        recall_initiation_date=row["init_date"],
        distribution_pattern=row["distribution_pattern"],
        voluntary_mandated=row["voluntary_mandated"],
        status=row["status"],
    )


def load_split(
    conn: sqlite3.Connection, split: Split, cutoff: str = SPLIT_CUTOFF
) -> list[Example]:
    """Load one side of the temporal split, ordered deterministically."""
    if split == "train":
        clause = f"AND {SPLIT_DATE_COLUMN} < ?"
    elif split == "test":
        clause = f"AND {SPLIT_DATE_COLUMN} >= ?"
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown split {split!r}")
    sql = f"{_SELECT} {clause} ORDER BY {SPLIT_DATE_COLUMN}, record_key"
    rows = conn.execute(sql, (*LABEL_CLASSES, cutoff)).fetchall()
    return [_row_to_example(r) for r in rows]


def verify_split(
    conn: sqlite3.Connection, cutoff: str = SPLIT_CUTOFF
) -> dict[str, object]:
    """Check every split invariant. Raises LeakageError on any violation.

    This runs before training and before evaluation, so a future data refresh
    that introduces a boundary-spanning event fails loudly instead of quietly
    inflating the score.
    """
    train = load_split(conn, "train", cutoff)
    test = load_split(conn, "test", cutoff)

    if not train or not test:
        raise LeakageError(f"empty split at cutoff {cutoff}: {len(train)}/{len(test)}")

    # 1. No record appears on both sides.
    train_keys = {e.record_key for e in train}
    test_keys = {e.record_key for e in test}
    overlap = train_keys & test_keys
    if overlap:
        raise LeakageError(f"{len(overlap)} record_keys in both splits")

    # 2. No event_id spans the boundary. This is the one that would actually
    #    bite: sibling recalls of one incident share a reason verbatim.
    train_events = {e.event_id for e in train if e.event_id}
    test_events = {e.event_id for e in test if e.event_id}
    shared = train_events & test_events
    if shared:
        raise LeakageError(
            f"{len(shared)} event_ids span the split cutoff {cutoff}: "
            f"{sorted(shared)[:5]}"
        )

    # 3. Dates actually respect the cutoff.
    bad_train = [e.record_key for e in train if (e.recall_initiation_date or "") >= cutoff]
    bad_test = [e.record_key for e in test if (e.recall_initiation_date or "") < cutoff]
    if bad_train or bad_test:
        raise LeakageError(f"date cutoff violated: {bad_train[:3]} {bad_test[:3]}")

    return {
        "cutoff": cutoff,
        "n_train": len(train),
        "n_test": len(test),
        "n_train_events": len(train_events),
        "n_test_events": len(test_events),
        "shared_firms": len(
            {e.recalling_firm for e in train if e.recalling_firm}
            & {e.recalling_firm for e in test if e.recalling_firm}
        ),
        "train_class_counts": class_counts(train),
        "test_class_counts": class_counts(test),
    }


def class_counts(examples: Iterable[Example]) -> dict[str, int]:
    counts = {c: 0 for c in LABEL_CLASSES}
    for e in examples:
        counts[e.classification] = counts.get(e.classification, 0) + 1
    return counts


def stratified_sample(
    examples: list[Example], n: int, seed: int = 20250101
) -> list[Example]:
    """A deterministic, class-proportional subsample of `examples`.

    A plain --limit prefix cut is biased here: class share drifts across the
    test window's own timeline (Class II is 80.2% of train but 87.5% of test
    overall), so truncating to "the first N by date" would not even match the
    test split's own distribution, let alone the corpus's. Sampling within each
    class independently keeps the sample representative regardless of n.

    Raises rather than silently returning fewer than requested if `examples` is
    empty, so a caller never mistakes "no data" for "a valid small sample".
    """
    import random

    if not examples:
        raise ValueError("cannot sample from zero examples")
    if n <= 0:
        raise ValueError(f"sample size must be positive, got {n}")
    if n >= len(examples):
        return sorted(examples, key=lambda e: e.record_key)

    by_class: dict[str, list[Example]] = {}
    for e in examples:
        by_class.setdefault(e.classification, []).append(e)

    total = len(examples)
    rng = random.Random(seed)
    sample: list[Example] = []
    remaining = n
    classes = sorted(by_class)  # deterministic iteration order
    for i, cls in enumerate(classes):
        pool = by_class[cls]
        if i == len(classes) - 1:
            take = min(remaining, len(pool))  # last class absorbs rounding
        else:
            take = min(len(pool), round(n * len(pool) / total))
            take = min(take, remaining)
        sample.extend(rng.sample(pool, take))
        remaining -= take

    return sorted(sample, key=lambda e: e.record_key)


def main() -> int:
    from .db import connect

    conn = connect()
    info = verify_split(conn)
    print(f"Split cutoff: {info['cutoff']} (on {SPLIT_DATE_COLUMN})")
    print(f"  train  n={info['n_train']:>6,}  events={info['n_train_events']:,}")
    print(f"  test   n={info['n_test']:>6,}  events={info['n_test_events']:,}")
    print("  event_ids spanning the split: 0 (verified)")
    print(f"  firms appearing on both sides: {info['shared_firms']:,} (expected; not leakage)")
    for name in ("train", "test"):
        counts = info[f"{name}_class_counts"]
        total = sum(counts.values())
        parts = ", ".join(f"{k}={v:,} ({v / total:.1%})" for k, v in counts.items())
        print(f"  {name} classes: {parts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
