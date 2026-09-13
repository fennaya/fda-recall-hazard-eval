"""Tests for the temporal split and its leakage guarantees."""

from __future__ import annotations

import pytest

from fda_hazard.splits import LeakageError, load_split, verify_split
from tests.conftest import insert, make_record

CUTOFF = "2025-01-01"


def test_split_sizes_and_sides(corpus):
    train = load_split(corpus, "train", CUTOFF)
    test = load_split(corpus, "test", CUTOFF)
    assert len(train) == 5
    assert len(test) == 3
    assert all(e.recall_initiation_date < CUTOFF for e in train)
    assert all(e.recall_initiation_date >= CUTOFF for e in test)


def test_cutoff_boundary_is_inclusive_on_the_test_side(conn):
    insert(
        conn,
        [
            make_record("X-1", "Class II", "2024-12-31", "e1"),
            make_record("X-2", "Class II", "2025-01-01", "e2"),
        ],
    )
    train = load_split(conn, "train", CUTOFF)
    test = load_split(conn, "test", CUTOFF)
    assert [e.record_key for e in train] == ["X-1"]
    assert [e.record_key for e in test] == ["X-2"]


def test_verify_split_passes_on_clean_corpus(corpus):
    info = verify_split(corpus, CUTOFF)
    assert info["n_train"] == 5
    assert info["n_test"] == 3


def test_event_id_spanning_the_split_is_caught(conn):
    """The failure this whole module exists to prevent."""
    insert(
        conn,
        [
            make_record("S-1", "Class I", "2024-12-01", "shared-event"),
            make_record("S-2", "Class I", "2025-02-01", "shared-event"),
        ],
    )
    with pytest.raises(LeakageError, match="event_ids span the split"):
        verify_split(conn, CUTOFF)


def test_unlabeled_records_are_excluded(conn):
    insert(
        conn,
        [
            make_record("U-1", "Not Yet Classified", "2023-01-01", "e1"),
            make_record("U-2", "Class II", "2023-01-02", "e2"),
            make_record("U-3", "Class II", "2025-06-01", "e3"),
        ],
    )
    train = load_split(conn, "train", CUTOFF)
    assert [e.record_key for e in train] == ["U-2"]


def test_records_without_a_date_are_excluded(conn):
    rec = make_record("D-1", "Class II", "2023-01-01", "e1")
    rec["recall_initiation_date_iso"] = None
    insert(conn, [rec, make_record("D-2", "Class II", "2023-01-02", "e2")])
    train = load_split(conn, "train", CUTOFF)
    assert [e.record_key for e in train] == ["D-2"]


def test_empty_split_raises_rather_than_scoring_nothing(conn):
    insert(conn, [make_record("T-1", "Class II", "2023-01-01", "e1")])
    with pytest.raises(LeakageError, match="empty split"):
        verify_split(conn, CUTOFF)


def test_load_split_is_deterministic(corpus):
    a = [e.record_key for e in load_split(corpus, "train", CUTOFF)]
    b = [e.record_key for e in load_split(corpus, "train", CUTOFF)]
    assert a == b == sorted(a, key=lambda k: k)


def test_example_text_combines_description_and_reason(corpus):
    e = load_split(corpus, "train", CUTOFF)[0]
    assert e.product_description in e.text()
    assert e.reason_for_recall in e.text()


def test_example_year(corpus):
    e = load_split(corpus, "test", CUTOFF)[0]
    assert e.year == 2025


# -- stratified sampling ----------------------------------------------------


def test_stratified_sample_preserves_class_proportions(conn):
    records = (
        [make_record(f"II-{i}", "Class II", "2025-06-01", f"e2-{i}") for i in range(80)]
        + [make_record(f"I-{i}", "Class I", "2025-06-01", f"e1-{i}") for i in range(10)]
        + [make_record(f"III-{i}", "Class III", "2025-06-01", f"e3-{i}") for i in range(10)]
    )
    insert(conn, records)
    test = load_split(conn, "test", CUTOFF)
    from fda_hazard.splits import stratified_sample

    sample = stratified_sample(test, 20, seed=1)
    counts = {c: sum(1 for e in sample if e.classification == c) for c in
              ("Class I", "Class II", "Class III")}
    assert len(sample) == 20
    # 80/10/10 split of 100 scaled to 20 -> roughly 16/2/2.
    assert counts["Class II"] >= 14
    assert counts["Class I"] >= 1
    assert counts["Class III"] >= 1


def test_stratified_sample_is_deterministic(conn):
    records = [make_record(f"R-{i}", "Class II", "2025-06-01", f"e-{i}") for i in range(30)]
    insert(conn, records)
    test = load_split(conn, "test", CUTOFF)
    from fda_hazard.splits import stratified_sample

    a = [e.record_key for e in stratified_sample(test, 10, seed=7)]
    b = [e.record_key for e in stratified_sample(test, 10, seed=7)]
    assert a == b


def test_stratified_sample_different_seeds_differ(conn):
    records = [make_record(f"R-{i}", "Class II", "2025-06-01", f"e-{i}") for i in range(30)]
    insert(conn, records)
    test = load_split(conn, "test", CUTOFF)
    from fda_hazard.splits import stratified_sample

    a = {e.record_key for e in stratified_sample(test, 10, seed=1)}
    b = {e.record_key for e in stratified_sample(test, 10, seed=2)}
    assert a != b


def test_stratified_sample_n_greater_than_available_returns_all(corpus):
    from fda_hazard.splits import stratified_sample

    test = load_split(corpus, "test", CUTOFF)
    sample = stratified_sample(test, 1000)
    assert len(sample) == len(test)
    assert {e.record_key for e in sample} == {e.record_key for e in test}


def test_stratified_sample_only_draws_from_the_given_examples(corpus):
    from fda_hazard.splits import stratified_sample

    train = load_split(corpus, "train", CUTOFF)
    test_keys = {e.record_key for e in load_split(corpus, "test", CUTOFF)}
    sample = stratified_sample(train, 3, seed=1)
    assert all(e.record_key not in test_keys for e in sample)


def test_stratified_sample_rejects_zero_or_negative_n(corpus):
    from fda_hazard.splits import stratified_sample

    test = load_split(corpus, "test", CUTOFF)
    with pytest.raises(ValueError, match="positive"):
        stratified_sample(test, 0)
    with pytest.raises(ValueError, match="positive"):
        stratified_sample(test, -5)


def test_stratified_sample_rejects_empty_input():
    from fda_hazard.splits import stratified_sample

    with pytest.raises(ValueError, match="zero examples"):
        stratified_sample([], 5)
