"""Tests for run persistence and the dashboard's queries.

The project claim is that every number shown in the UI traces to a row in the
database. These tests check that the stored rows and the recomputed metrics
agree, so the claim is verifiable rather than aspirational.
"""

from __future__ import annotations

import pytest

from fda_hazard import queries as q
from fda_hazard.metrics import score
from fda_hazard.runs import record_run
from fda_hazard.splits import load_split

CUTOFF = "2025-01-01"
I, II, III = "Class I", "Class II", "Class III"


def _record(conn, system, preds, **kw):
    test = load_split(conn, "test", CUTOFF)
    metrics = score([e.classification for e in test], preds)
    return record_run(
        conn, system=system, metrics=metrics, examples=test, predictions=preds,
        cutoff=CUTOFF, **kw
    ), test, metrics


def test_record_run_writes_one_row_per_example(corpus):
    uid, test, _ = _record(corpus, "baseline_majority", [II, II, II])
    n = corpus.execute(
        "SELECT COUNT(*) FROM predictions WHERE run_uid = ?", (uid,)
    ).fetchone()[0]
    assert n == len(test) == 3


def test_stored_headline_metrics_match_the_scorer(corpus):
    uid, _, metrics = _record(corpus, "agent", [II, I, III])
    row = q.run_by_uid(corpus, uid)
    assert row["accuracy"] == pytest.approx(metrics.accuracy)
    assert row["macro_f1"] == pytest.approx(metrics.macro_f1)
    assert row["total_cost"] == pytest.approx(metrics.total_cost)
    assert row["class1_missed"] == metrics.class1_missed


def test_stored_costs_sum_to_the_run_total(corpus):
    """The Errors page sums per-row costs; that must equal the run's total."""
    uid, _, metrics = _record(corpus, "agent", [II, II, II])
    total = corpus.execute(
        "SELECT SUM(cost) FROM predictions WHERE run_uid = ?", (uid,)
    ).fetchone()[0]
    assert total == pytest.approx(metrics.total_cost)


def test_correct_flag_matches_truth_equality(corpus):
    uid, test, _ = _record(corpus, "agent", [II, I, III])
    rows = corpus.execute(
        "SELECT truth, predicted, correct FROM predictions WHERE run_uid = ?", (uid,)
    ).fetchall()
    for r in rows:
        assert bool(r["correct"]) == (r["truth"] == r["predicted"])


def test_mismatched_prediction_count_raises(corpus):
    test = load_split(corpus, "test", CUTOFF)
    metrics = score([e.classification for e in test], [II] * len(test))
    with pytest.raises(ValueError, match="predictions"):
        record_run(
            corpus, system="agent", metrics=metrics, examples=test,
            predictions=[II], cutoff=CUTOFF,
        )


def test_errors_query_returns_only_wrong_predictions(corpus):
    uid, test, _ = _record(corpus, "agent", [II, II, II])
    rows = q.errors(corpus, uid)
    assert rows, "expected some errors"
    for r in rows:
        assert r["truth"] != r["predicted"]
    n_wrong = sum(1 for e in test if e.classification != II)
    assert len(rows) == n_wrong


def test_errors_are_sorted_by_cost_descending(corpus):
    uid, _, _ = _record(corpus, "agent", [II, II, II])
    costs = [r["cost"] for r in q.errors(corpus, uid, sort="cost")]
    assert costs == sorted(costs, reverse=True)


def test_errors_truth_filter(corpus):
    uid, _, _ = _record(corpus, "agent", [II, II, II])
    rows = q.errors(corpus, uid, truth=I)
    assert rows and all(r["truth"] == I for r in rows)


def test_error_summary_totals_match_predictions_table(corpus):
    uid, _, metrics = _record(corpus, "agent", [II, II, II])
    s = q.error_summary(corpus, uid)
    assert s["total_cost"] == pytest.approx(metrics.total_cost)
    assert s["n_total"] == metrics.n
    assert sum(b["n"] for b in s["buckets"]) == s["n_errors"]


def test_scoreboard_reports_latest_run_per_system(corpus):
    _record(corpus, "agent", [II, II, II])
    uid2, _, _ = _record(corpus, "agent", [I, I, I])
    board = q.scoreboard(corpus)
    agent = next(r for r in board if r["system"] == "agent")
    assert agent["run_uid"] == uid2


def test_run_diff_against_previous_run(corpus):
    _record(corpus, "agent", [II, II, II], prompt_version="v1", model="m1")
    uid2, _, _ = _record(corpus, "agent", [II, I, III], prompt_version="v2", model="m2")
    row = q.run_by_uid(corpus, uid2)
    diff = q.run_diff(corpus, row)
    assert diff is not None
    assert diff["prompt_changed"] and diff["model_changed"]
    assert diff["deltas"]["accuracy"]["delta"] > 0
    assert diff["deltas"]["accuracy"]["better"] is True


def test_run_diff_marks_lower_cost_as_better(corpus):
    _record(corpus, "agent", [III, III, III])   # expensive
    uid2, _, _ = _record(corpus, "agent", [II, I, III])  # perfect
    diff = q.run_diff(corpus, q.run_by_uid(corpus, uid2))
    assert diff["deltas"]["mean_cost"]["delta"] < 0
    assert diff["deltas"]["mean_cost"]["better"] is True


def test_first_run_has_no_diff(corpus):
    uid, _, _ = _record(corpus, "agent", [II, II, II])
    assert q.run_diff(corpus, q.run_by_uid(corpus, uid)) is None


def test_accuracy_by_year_matches_stored_predictions(corpus):
    uid, test, _ = _record(corpus, "agent", [II, I, III])
    by_year = q.accuracy_by_year(corpus, uid)
    assert sum(y["n"] for y in by_year) == len(test)
    for y in by_year:
        assert 0.0 <= y["accuracy"] <= 1.0


def test_split_distribution_counts_match_the_split(corpus):
    dist = q.split_distribution(corpus, CUTOFF)
    assert dist["train_total"] == len(load_split(corpus, "train", CUTOFF))
    assert dist["test_total"] == len(load_split(corpus, "test", CUTOFF))


# -- no-mixing rule (Step 3) -------------------------------------------------
# A scored run must never blend predictions from two different agent
# architectures (tool_loop vs single_call). This is the test the plan asked
# for: it must FAIL (raise) if a run mixes them.


def test_record_run_rejects_mixed_architectures(corpus):
    test = load_split(corpus, "test", CUTOFF)
    preds = [II, II, II]
    details = [
        {"record_key": test[0].record_key, "architecture": "tool_loop"},
        {"record_key": test[1].record_key, "architecture": "single_call"},
        {"record_key": test[2].record_key, "architecture": "tool_loop"},
    ]
    metrics = score([e.classification for e in test], preds)
    with pytest.raises(ValueError, match="mixes architectures"):
        record_run(
            corpus, system="agent", metrics=metrics, examples=test,
            predictions=preds, cutoff=CUTOFF, details=details,
        )
    # And nothing partial got written.
    assert corpus.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 0
    assert corpus.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0


def test_record_run_rejects_details_disagreeing_with_declared_architecture(corpus):
    test = load_split(corpus, "test", CUTOFF)
    preds = [II, II, II]
    details = [{"record_key": e.record_key, "architecture": "single_call"} for e in test]
    metrics = score([e.classification for e in test], preds)
    with pytest.raises(ValueError, match="does not match the declared architecture"):
        record_run(
            corpus, system="agent", metrics=metrics, examples=test,
            predictions=preds, cutoff=CUTOFF, architecture="tool_loop", details=details,
        )


def test_record_run_accepts_a_single_consistent_architecture(corpus):
    test = load_split(corpus, "test", CUTOFF)
    preds = [II, II, II]
    details = [{"record_key": e.record_key, "architecture": "single_call"} for e in test]
    metrics = score([e.classification for e in test], preds)
    uid = record_run(
        corpus, system="agent", metrics=metrics, examples=test,
        predictions=preds, cutoff=CUTOFF, architecture="single_call", details=details,
    )
    row = corpus.execute(
        "SELECT DISTINCT architecture FROM predictions WHERE run_uid=?", (uid,)
    ).fetchall()
    assert [r["architecture"] for r in row] == ["single_call"]
    assert corpus.execute(
        "SELECT architecture FROM eval_runs WHERE run_uid=?", (uid,)
    ).fetchone()["architecture"] == "single_call"


def test_record_run_defaults_to_tool_loop_with_no_details(corpus):
    """Baselines and the existing agent path never pass an 'architecture' key
    in details -- they must keep working exactly as before, tagged tool_loop."""
    test = load_split(corpus, "test", CUTOFF)
    preds = [II, II, II]
    metrics = score([e.classification for e in test], preds)
    uid = record_run(
        corpus, system="baseline_majority", metrics=metrics, examples=test,
        predictions=preds, cutoff=CUTOFF,
    )
    assert corpus.execute(
        "SELECT architecture FROM eval_runs WHERE run_uid=?", (uid,)
    ).fetchone()["architecture"] == "tool_loop"


def test_record_run_rejects_unknown_architecture_name(corpus):
    test = load_split(corpus, "test", CUTOFF)
    preds = [II, II, II]
    metrics = score([e.classification for e in test], preds)
    with pytest.raises(ValueError, match="unknown architecture"):
        record_run(
            corpus, system="agent", metrics=metrics, examples=test,
            predictions=preds, cutoff=CUTOFF, architecture="ensemble_vote",
        )
