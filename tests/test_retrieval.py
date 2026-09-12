"""Tests for precedent retrieval, focused on the test-split ban.

Leakage here would be invisible in the score -- it would just make the agent
look good -- so these are the tests that matter most.
"""

from __future__ import annotations

import pytest

from fda_hazard.retrieval import PrecedentIndex, build_index
from fda_hazard.splits import LeakageError, load_split

CUTOFF = "2025-01-01"


def test_index_contains_only_train_records(corpus):
    index = build_index(corpus, CUTOFF)
    test_keys = {e.record_key for e in load_split(corpus, "test", CUTOFF)}
    assert len(index) == 5
    for key in test_keys:
        assert not index.contains(key)


def test_search_never_returns_a_test_record(corpus):
    """The headline guarantee. A test recall whose reason is verbatim identical
    to a train recall must still only ever surface the train one."""
    index = build_index(corpus, CUTOFF)
    test_keys = {e.record_key for e in load_split(corpus, "test", CUTOFF)}
    for ex in load_split(corpus, "test", CUTOFF):
        hits = index.search(ex.text(), k=10)
        assert hits, "expected at least one precedent"
        for hit in hits:
            assert hit.record_key not in test_keys


def test_building_an_index_from_the_test_split_raises(corpus):
    """Enforced in code, not by convention: passing the wrong split is an error."""
    test = load_split(corpus, "test", CUTOFF)
    with pytest.raises(LeakageError, match="must be built from the train split"):
        PrecedentIndex(test, cutoff=CUTOFF)


def test_index_rejects_a_single_contaminated_example(corpus):
    train = load_split(corpus, "train", CUTOFF)
    test = load_split(corpus, "test", CUTOFF)
    with pytest.raises(LeakageError, match="at/after the cutoff"):
        PrecedentIndex(train + test[:1], cutoff=CUTOFF)


def test_identical_text_retrieves_the_train_twin(corpus):
    """B-1's reason is verbatim identical to A-2's, so A-2 must rank first."""
    index = build_index(corpus, CUTOFF)
    test = {e.record_key: e for e in load_split(corpus, "test", CUTOFF)}
    hits = index.search(test["B-1"].text(), k=3)
    assert hits[0].record_key == "A-2"
    assert hits[0].classification == "Class II"
    assert hits[0].similarity > 0.5


def test_exclude_keys_filters_self(corpus):
    index = build_index(corpus, CUTOFF)
    train = {e.record_key: e for e in load_split(corpus, "train", CUTOFF)}
    hits = index.search(train["A-2"].text(), k=5, exclude_keys=frozenset({"A-2"}))
    assert all(h.record_key != "A-2" for h in hits)


def test_k_is_respected(corpus):
    index = build_index(corpus, CUTOFF)
    assert len(index.search("subpotent tablets", k=2)) <= 2
    assert index.search("subpotent tablets", k=0) == []


def test_precedents_carry_their_classification(corpus):
    index = build_index(corpus, CUTOFF)
    for hit in index.search("penicillin contamination", k=3):
        assert hit.classification in ("Class I", "Class II", "Class III")
        assert hit.recall_initiation_date < CUTOFF


def test_zero_similarity_results_are_dropped(corpus):
    index = build_index(corpus, CUTOFF)
    assert index.search("zzzz qqqq xxxx nonsense token", k=5) == []


def test_empty_index_raises(corpus):
    with pytest.raises(ValueError, match="zero examples"):
        PrecedentIndex([], cutoff=CUTOFF)


def test_precedent_to_dict_is_json_shaped(corpus):
    index = build_index(corpus, CUTOFF)
    d = index.search("sterility failure injectable", k=1)[0].to_dict()
    assert set(d) == {
        "record_key", "classification", "product_description",
        "reason_for_recall", "recall_initiation_date", "similarity",
    }
