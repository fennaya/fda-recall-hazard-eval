"""The two baselines the agent has to beat to justify its cost.

Both are fit on train only and scored on the held-out test split. Neither uses
the scoring code to fit anything -- scoring stays in metrics.py, which no model
touches.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .db import connect
from .metrics import Metrics, format_report, score
from .splits import SPLIT_CUTOFF, Example, load_split, verify_split

# Fixed so the baseline is reproducible run to run.
RANDOM_SEED = 20250101


def majority_baseline(
    train: list[Example], test: list[Example]
) -> tuple[Metrics, list[str]]:
    """Always predict the most common class in train.

    Scores ~87% on this test split purely because Class II dominates, which is
    exactly why accuracy alone is a useless headline number here.
    """
    majority = Counter(e.classification for e in train).most_common(1)[0][0]
    preds = [majority] * len(test)
    return score([e.classification for e in test], preds), preds


def tfidf_baseline(
    train: list[Example],
    test: list[Example],
    reason_only: bool = True,
    balanced: bool = False,
) -> tuple[Metrics, list[str]]:
    """TF-IDF over reason_for_recall, then multinomial logistic regression.

    Both weightings are scored because they disagree about which is better, and
    which one you call "the baseline" changes the bar the agent has to clear:
    unweighted chases accuracy and drifts toward the majority class, while
    balanced trades a lot of accuracy for Class I recall. Reporting only one
    would be picking a convenient opponent.
    """
    def text(e: Example) -> str:
        return e.reason_for_recall if reason_only else e.text()

    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vec.fit_transform([text(e) for e in train])
    x_test = vec.transform([text(e) for e in test])

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced" if balanced else None,
        random_state=RANDOM_SEED,
    )
    clf.fit(x_train, [e.classification for e in train])
    preds = list(clf.predict(x_test))
    return score([e.classification for e in test], preds), preds


def run_baselines(
    conn: sqlite3.Connection, cutoff: str = SPLIT_CUTOFF
) -> dict[str, tuple[Metrics, list[str]]]:
    verify_split(conn, cutoff)  # refuse to score a leaking split
    train = load_split(conn, "train", cutoff)
    test = load_split(conn, "test", cutoff)
    return {
        "baseline_majority": majority_baseline(train, test),
        "baseline_tfidf": tfidf_baseline(train, test, balanced=False),
        "baseline_tfidf_balanced": tfidf_baseline(train, test, balanced=True),
    }


def main(argv: list[str] | None = None) -> int:
    from .runs import record_run

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", default=SPLIT_CUTOFF)
    ap.add_argument("--no-save", action="store_true", help="print without writing rows")
    args = ap.parse_args(argv)

    conn = connect()
    test = load_split(conn, "test", args.cutoff)
    results = run_baselines(conn, args.cutoff)

    for name, (metrics, preds) in results.items():
        print(format_report(metrics, name))
        print()
        if not args.no_save:
            uid = record_run(
                conn,
                system=name,
                metrics=metrics,
                examples=test,
                predictions=preds,
                cutoff=args.cutoff,
                config={
                    "random_seed": RANDOM_SEED,
                    "balanced": name.endswith("_balanced"),
                },
            )
            print(f"  saved as run {uid}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
