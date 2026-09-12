"""Scoring. Pure Python, no model and no ML library involved.

Every number the dashboard displays is produced here and stored in SQLite.
The model is never asked to compute, estimate, or summarise a metric -- it only
ever emits a class label, which this module compares against FDA's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

from .db import LABEL_CLASSES

# Cost matrix: COST[truth][predicted].
#
# The asymmetry is the point. Missing a Class I -- a recall with reasonable
# probability of serious injury or death -- and calling it Class II is the
# expensive error, so it costs 5x the reverse mistake of over-calling a Class II
# as Class I. Under-calling severity is always penalised more than over-calling,
# because the failure mode of over-calling is wasted effort while the failure
# mode of under-calling is an unretrieved dangerous drug.
COST_MATRIX: dict[str, dict[str, float]] = {
    "Class I": {"Class I": 0.0, "Class II": 5.0, "Class III": 10.0},
    "Class II": {"Class I": 1.0, "Class II": 0.0, "Class III": 5.0},
    "Class III": {"Class I": 1.0, "Class II": 1.0, "Class III": 0.0},
}


def cost_of(truth: str, predicted: str) -> float:
    """Cost of predicting `predicted` when FDA said `truth`."""
    try:
        return COST_MATRIX[truth][predicted]
    except KeyError as exc:
        raise ValueError(f"unknown class in ({truth!r}, {predicted!r})") from exc


@dataclass
class ClassReport:
    label: str
    support: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass
class Metrics:
    n: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    total_cost: float
    mean_cost: float
    labels: list[str]
    # confusion[truth][predicted]
    confusion: dict[str, dict[str, int]]
    per_class: dict[str, ClassReport]
    class1_recall: float = 0.0
    class1_precision: float = 0.0
    class1_f1: float = 0.0
    class1_support: int = 0
    class1_missed: int = 0
    class1_missed_as: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_class"] = {k: asdict(v) for k, v in self.per_class.items()}
        return d


def _safe_div(num: float, den: float) -> float:
    """Division that returns 0.0 rather than raising on an empty denominator.

    A class with no support and no predictions scores 0, which is the
    conventional choice and keeps macro-F1 comparable across runs.
    """
    return num / den if den else 0.0


def confusion_matrix(
    truths: Sequence[str], preds: Sequence[str], labels: Sequence[str] = LABEL_CLASSES
) -> dict[str, dict[str, int]]:
    """Full confusion matrix as confusion[truth][predicted]."""
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(truths, preds, strict=True):
        if t not in matrix:
            raise ValueError(f"unknown truth label {t!r}")
        if p not in matrix[t]:
            raise ValueError(f"unknown predicted label {p!r}")
        matrix[t][p] += 1
    return matrix


def score(
    truths: Sequence[str], preds: Sequence[str], labels: Sequence[str] = LABEL_CLASSES
) -> Metrics:
    """Score a set of predictions against ground truth.

    `truths` and `preds` must be equal length and aligned; a length mismatch is
    an error rather than something to silently zip away.
    """
    if len(truths) != len(preds):
        raise ValueError(f"length mismatch: {len(truths)} truths, {len(preds)} preds")
    if not truths:
        raise ValueError("cannot score an empty prediction set")

    labels = list(labels)
    matrix = confusion_matrix(truths, preds, labels)
    n = len(truths)

    correct = sum(matrix[c][c] for c in labels)
    accuracy = correct / n

    per_class: dict[str, ClassReport] = {}
    for c in labels:
        tp = matrix[c][c]
        fn = sum(matrix[c][p] for p in labels if p != c)
        fp = sum(matrix[t][c] for t in labels if t != c)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[c] = ClassReport(
            label=c,
            support=tp + fn,
            predicted_count=tp + fp,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    macro_f1 = sum(r.f1 for r in per_class.values()) / len(labels)
    macro_p = sum(r.precision for r in per_class.values()) / len(labels)
    macro_r = sum(r.recall for r in per_class.values()) / len(labels)

    total_cost = sum(cost_of(t, p) for t, p in zip(truths, preds, strict=True))

    c1 = per_class.get("Class I")
    metrics = Metrics(
        n=n,
        accuracy=accuracy,
        macro_f1=macro_f1,
        macro_precision=macro_p,
        macro_recall=macro_r,
        total_cost=total_cost,
        mean_cost=total_cost / n,
        labels=labels,
        confusion=matrix,
        per_class=per_class,
    )
    if c1 is not None:
        metrics.class1_recall = c1.recall
        metrics.class1_precision = c1.precision
        metrics.class1_f1 = c1.f1
        metrics.class1_support = c1.support
        metrics.class1_missed = c1.false_negatives
        metrics.class1_missed_as = {
            p: matrix["Class I"][p] for p in labels if p != "Class I"
        }
    return metrics


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used for Class I recall, where the test split holds few enough positives
    that a bare point estimate would overstate what we actually know.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def format_report(m: Metrics, title: str = "") -> str:
    """Plain-text scoring report. Tabular numbers, no emoji."""
    lines: list[str] = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(f"n={m.n:,}  accuracy={m.accuracy:.4f}  macro_f1={m.macro_f1:.4f}")
    lines.append(f"total_cost={m.total_cost:,.0f}  mean_cost={m.mean_cost:.4f}")
    lo, hi = wilson_interval(
        m.per_class["Class I"].true_positives, m.class1_support
    ) if "Class I" in m.per_class else (0.0, 0.0)
    lines.append(
        f"CLASS I recall={m.class1_recall:.4f} [95% CI {lo:.3f}-{hi:.3f}]  "
        f"precision={m.class1_precision:.4f}  missed={m.class1_missed}/{m.class1_support}"
    )
    if m.class1_missed_as:
        as_str = ", ".join(f"{k}: {v}" for k, v in m.class1_missed_as.items())
        lines.append(f"  Class I misses landed on -> {as_str}")
    lines.append("")
    lines.append(f"{'class':<11}{'support':>9}{'prec':>8}{'recall':>8}{'f1':>8}")
    lines.append("-" * 44)
    for c in m.labels:
        r = m.per_class[c]
        lines.append(
            f"{c:<11}{r.support:>9,}{r.precision:>8.4f}{r.recall:>8.4f}{r.f1:>8.4f}"
        )
    lines.append("")
    lines.append("confusion matrix (rows = FDA truth, cols = predicted)")
    header = " " * 12 + "".join(f"{c:>11}" for c in m.labels)
    lines.append(header)
    for t in m.labels:
        row = "".join(f"{m.confusion[t][p]:>11,}" for p in m.labels)
        lines.append(f"{t:<12}{row}")
    return "\n".join(lines)
