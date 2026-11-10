"""Scores for a set of predictions, the same for every model.

Macro-F1 is the headline: four chapters are most of the data, and a model that learned
only those would look good on accuracy while failing every other chapter. Accuracy,
per-class scores and the most common confusions are kept next to it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# A class with no predictions scores 0, not a warning. (Passed through a dict typed Any:
# scikit-learn's annotations only admit the string form of this argument.)
ZERO: dict[str, Any] = {"zero_division": 0}


@dataclass
class Scores:
    examples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusions: list[tuple[str, str, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": self.per_class,
            "confusions": [list(c) for c in self.confusions],
        }


def score(truth: list[str], predicted: list[str], labels: list[str]) -> Scores:
    """Scores over `labels`, the classes the dataset defines, so a class the model never
    predicts still counts against macro-F1 instead of disappearing from it.

    A class with no examples in `truth` is left out of the averages: its F1 would be 0
    for every model, whatever it predicts, which says nothing about the models and only
    lowers all of them alike. Wrong predictions of it still cost recall elsewhere."""
    metrics: Any = precision_recall_fscore_support(truth, predicted, labels=labels, **ZERO)
    precision, recall, f1, support = metrics
    per_class = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    }
    mistakes = Counter((t, p) for t, p in zip(truth, predicted, strict=True) if t != p)
    present = [label for label in labels if label in set(truth)]
    return Scores(
        examples=len(truth),
        accuracy=float(accuracy_score(truth, predicted)),
        macro_f1=float(f1_score(truth, predicted, labels=present, average="macro", **ZERO)),
        weighted_f1=float(f1_score(truth, predicted, labels=present, average="weighted", **ZERO)),
        per_class=per_class,
        confusions=[(t, p, n) for (t, p), n in mistakes.most_common(10)],
    )


@dataclass
class Comparison:
    """Model a against model b on the same test items."""

    macro_f1: tuple[float, float, float]  # difference a - b, and its 95% interval
    accuracy: tuple[float, float, float]
    only_a: int  # items only a gets right
    only_b: int
    mcnemar_p: float


def compare(
    truth: list[str], a: list[str], b: list[str], rounds: int = 2000, seed: int = 0
) -> Comparison:
    """Whether a beats b by more than the test set's sampling noise.

    A paired bootstrap resamples test items, scores both models on the same resample and
    keeps the differences, so the interval reflects how much the gap would move on
    another test set like this one. McNemar's exact test asks the same of accuracy
    using only the items where exactly one model is right.
    """
    rng = np.random.default_rng(seed)
    y, pa, pb = np.array(truth), np.array(a), np.array(b)
    macro, accuracy = [], []
    for _ in range(rounds):
        i = rng.integers(0, len(y), len(y))
        labels = sorted(set(y[i]))
        macro.append(
            f1_score(y[i], pa[i], labels=labels, average="macro", **ZERO)
            - f1_score(y[i], pb[i], labels=labels, average="macro", **ZERO)
        )
        accuracy.append(float((pa[i] == y[i]).mean() - (pb[i] == y[i]).mean()))

    def interval(values: list[float]) -> tuple[float, float, float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return float(np.mean(values)), float(low), float(high)

    only_a = int(((pa == y) & (pb != y)).sum())
    only_b = int(((pb == y) & (pa != y)).sum())
    return Comparison(interval(macro), interval(accuracy), only_a, only_b, mcnemar(only_a, only_b))


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value, with Python integers: 2**n overflows numpy's."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)
