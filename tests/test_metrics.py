"""Tests for the scoring code. These numbers are hand-computed, not
regenerated from the implementation -- a test that asserts whatever the code
currently returns proves nothing."""

from __future__ import annotations

import pytest

from fda_hazard.metrics import (
    COST_MATRIX,
    confusion_matrix,
    cost_of,
    score,
    wilson_interval,
)

I, II, III = "Class I", "Class II", "Class III"


def test_perfect_predictions():
    truths = [I, II, III, II]
    m = score(truths, list(truths))
    assert m.accuracy == 1.0
    assert m.macro_f1 == 1.0
    assert m.total_cost == 0.0
    assert m.class1_missed == 0
    assert m.class1_recall == 1.0


def test_all_wrong_in_one_direction():
    # Every Class I called Class II: the expensive error.
    truths = [I, I, I]
    preds = [II, II, II]
    m = score(truths, preds)
    assert m.accuracy == 0.0
    assert m.class1_recall == 0.0
    assert m.class1_missed == 3
    assert m.class1_missed_as == {II: 3, III: 0}
    assert m.total_cost == 15.0  # 3 * 5


def test_accuracy_and_confusion_hand_computed():
    truths = [I, I, II, II, II, III]
    preds = [I, II, II, II, I, III]
    m = score(truths, preds)
    # 4 of 6 correct: I->I, II->II, II->II, III->III
    assert m.accuracy == pytest.approx(4 / 6)
    assert m.confusion[I] == {I: 1, II: 1, III: 0}
    assert m.confusion[II] == {I: 1, II: 2, III: 0}
    assert m.confusion[III] == {I: 0, II: 0, III: 1}


def test_per_class_precision_recall_hand_computed():
    truths = [I, I, II, II, II, III]
    preds = [I, II, II, II, I, III]
    m = score(truths, preds)

    # Class I: predicted twice (1 true, 1 from a Class II), support 2.
    c1 = m.per_class[I]
    assert (c1.true_positives, c1.false_positives, c1.false_negatives) == (1, 1, 1)
    assert c1.precision == pytest.approx(0.5)
    assert c1.recall == pytest.approx(0.5)
    assert c1.f1 == pytest.approx(0.5)

    # Class II: predicted 3 times, 2 correct; support 3.
    c2 = m.per_class[II]
    assert (c2.true_positives, c2.false_positives, c2.false_negatives) == (2, 1, 1)
    assert c2.precision == pytest.approx(2 / 3)
    assert c2.recall == pytest.approx(2 / 3)

    c3 = m.per_class[III]
    assert c3.precision == 1.0 and c3.recall == 1.0


def test_macro_f1_is_unweighted_mean():
    truths = [I, I, II, II, II, III]
    preds = [I, II, II, II, I, III]
    m = score(truths, preds)
    expected = (0.5 + (2 / 3) + 1.0) / 3
    assert m.macro_f1 == pytest.approx(expected)


def test_macro_f1_punishes_majority_only_predictions():
    # 8 Class II, 1 Class I, 1 Class III. Always guessing Class II gets 80%
    # accuracy but only catches one of three classes.
    truths = [II] * 8 + [I, III]
    preds = [II] * 10
    m = score(truths, preds)
    assert m.accuracy == pytest.approx(0.8)
    # Class II f1 = 2*0.8*1/(1.8); others are 0.
    assert m.macro_f1 == pytest.approx((2 * 0.8 / 1.8) / 3)
    assert m.class1_recall == 0.0


def test_cost_asymmetry_is_five_to_one():
    """The headline requirement: predicting II when truth is I costs 5x the
    reverse."""
    assert cost_of(I, II) == 5.0
    assert cost_of(II, I) == 1.0
    assert cost_of(I, II) == 5 * cost_of(II, I)


def test_cost_matrix_diagonal_is_free():
    for c in (I, II, III):
        assert cost_of(c, c) == 0.0


def test_under_calling_always_costs_more_than_over_calling():
    severity = {I: 0, II: 1, III: 2}
    for truth in (I, II, III):
        for pred in (I, II, III):
            if severity[pred] > severity[truth]:  # under-called the hazard
                mirror = COST_MATRIX[pred][truth]
                assert cost_of(truth, pred) > mirror, (truth, pred)


def test_total_and_mean_cost():
    truths = [I, II, III]
    preds = [II, I, III]  # 5 + 1 + 0
    m = score(truths, preds)
    assert m.total_cost == 6.0
    assert m.mean_cost == pytest.approx(2.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        score([I, II], [I])


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        score([], [])


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unknown"):
        score([I], ["Class IV"])
    with pytest.raises(ValueError, match="unknown class"):
        cost_of(I, "Class IV")


def test_confusion_matrix_totals_equal_n():
    truths = [I, I, II, II, II, III, III]
    preds = [I, II, II, I, III, III, I]
    matrix = confusion_matrix(truths, preds)
    assert sum(sum(row.values()) for row in matrix.values()) == len(truths)


def test_absent_class_scores_zero_not_nan():
    # No Class III anywhere: its precision/recall/f1 must be 0, not NaN.
    m = score([I, II], [I, II])
    c3 = m.per_class[III]
    assert (c3.precision, c3.recall, c3.f1) == (0.0, 0.0, 0.0)
    assert m.macro_f1 == pytest.approx(2 / 3)


def test_wilson_interval_brackets_point_estimate():
    lo, hi = wilson_interval(30, 67)
    assert lo < 30 / 67 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_wilson_interval_is_wider_for_small_samples():
    lo_small, hi_small = wilson_interval(5, 10)
    lo_big, hi_big = wilson_interval(500, 1000)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_wilson_interval_handles_zero_total():
    assert wilson_interval(0, 0) == (0.0, 0.0)
