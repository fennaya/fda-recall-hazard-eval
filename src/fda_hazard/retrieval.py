"""Precedent retrieval over the TRAIN split only.

The test-split ban is enforced structurally, not by convention: the index is
built from a list of train Examples and never holds a reference to the
database, so there is no code path by which a test record could be returned.
The constructor additionally refuses any example dated on or after the cutoff,
so a caller who passes the wrong split gets an exception rather than a quietly
inflated score.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .splits import SPLIT_CUTOFF, Example, LeakageError, load_split


@dataclass(frozen=True)
class Precedent:
    record_key: str
    classification: str
    product_description: str
    reason_for_recall: str
    recall_initiation_date: str | None
    similarity: float

    def to_dict(self) -> dict:
        return {
            "record_key": self.record_key,
            "classification": self.classification,
            "product_description": self.product_description,
            "reason_for_recall": self.reason_for_recall,
            "recall_initiation_date": self.recall_initiation_date,
            "similarity": round(self.similarity, 4),
        }


class PrecedentIndex:
    """TF-IDF nearest-neighbour index over historical, already-classified recalls."""

    def __init__(self, train: list[Example], cutoff: str = SPLIT_CUTOFF) -> None:
        if not train:
            raise ValueError("cannot build a precedent index from zero examples")

        # Structural guarantee #1: nothing at or after the cutoff may enter.
        contaminated = [
            e.record_key
            for e in train
            if (e.recall_initiation_date or "") >= cutoff
        ]
        if contaminated:
            raise LeakageError(
                f"{len(contaminated)} example(s) at/after the cutoff {cutoff} were "
                f"passed to the precedent index: {contaminated[:5]}. "
                "The index must be built from the train split only."
            )

        self.cutoff = cutoff
        self.examples = list(train)
        self._keys = {e.record_key for e in self.examples}
        # min_df/max_df prune noise on the real 16k-document corpus but wipe out
        # the whole vocabulary on a small one, which would make every search
        # return nothing -- silently, and looking exactly like "no similar
        # precedent exists". Relax the pruning when the corpus is too small for
        # it to mean anything.
        small = len(self.examples) < 50
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1 if small else 2,
            max_df=1.0 if small else 0.9,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._matrix = self._vectorizer.fit_transform(
            [e.text() for e in self.examples]
        )
        if not self._vectorizer.vocabulary_:
            raise ValueError(
                f"precedent index built an empty vocabulary from "
                f"{len(self.examples)} examples; every search would return nothing"
            )

    def __len__(self) -> int:
        return len(self.examples)

    def search(
        self, query_text: str, k: int = 5, exclude_keys: frozenset[str] = frozenset()
    ) -> list[Precedent]:
        """Return the k most similar train recalls, most similar first."""
        if k <= 0:
            return []
        vec = self._vectorizer.transform([query_text])
        sims = linear_kernel(vec, self._matrix).ravel()

        order = sims.argsort()[::-1]
        out: list[Precedent] = []
        for idx in order:
            if len(out) >= k:
                break
            ex = self.examples[idx]
            if ex.record_key in exclude_keys:
                continue
            if sims[idx] <= 0.0:
                break
            out.append(
                Precedent(
                    record_key=ex.record_key,
                    classification=ex.classification,
                    product_description=ex.product_description,
                    reason_for_recall=ex.reason_for_recall,
                    recall_initiation_date=ex.recall_initiation_date,
                    similarity=float(sims[idx]),
                )
            )
        return out

    def contains(self, record_key: str) -> bool:
        return record_key in self._keys


def build_index(
    conn: sqlite3.Connection, cutoff: str = SPLIT_CUTOFF
) -> PrecedentIndex:
    """Build the index from the train split. The only supported entry point."""
    return PrecedentIndex(load_split(conn, "train", cutoff), cutoff=cutoff)
