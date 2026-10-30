"""The two models that every other model has to beat.

- Majority: always the most common training label. The floor.
- TF-IDF over character n-grams with logistic regression. Item descriptions are
  catalogue-like ("Seringa descartável 5ml, bico luer lock"), full of units,
  abbreviations and misspellings, which character n-grams handle without a tokenizer.
  It trains on a laptop CPU in seconds, so anything heavier has to justify itself
  against it.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ncm.build import normalize
from ncm.evaluate import score

# Regularisation strengths tried on the validation split; the best by macro-F1 is kept.
C_GRID = (1.0, 4.0, 16.0)


@dataclass
class Fitted:
    name: str
    model: Pipeline | str  # a pipeline, or the single label the majority model answers
    params: dict[str, float]
    train_seconds: float

    def predict(self, texts: list[str]) -> list[str]:
        if isinstance(self.model, str):
            return [self.model] * len(texts)
        return [str(label) for label in self.model.predict(texts)]


def majority(train: pd.DataFrame) -> Fitted:
    label = Counter(train["label"]).most_common(1)[0][0]
    return Fitted("majority", str(label), {}, 0.0)


def tfidf_pipeline(c: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize,
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            # Balanced class weights, so the rare chapters are not drowned by the four
            # that make up most of the data; macro-F1 is what this optimises for.
            ("classifier", LogisticRegression(C=c, max_iter=2_000, class_weight="balanced")),
        ]
    )


def tfidf(train: pd.DataFrame, validation: pd.DataFrame, labels: list[str]) -> Fitted:
    best: Fitted | None = None
    best_f1 = -1.0
    for c in C_GRID:
        start = time.perf_counter()
        pipeline = tfidf_pipeline(c).fit(train["text"].tolist(), train["label"].tolist())
        fitted = Fitted("tfidf-logreg", pipeline, {"C": c}, time.perf_counter() - start)
        f1 = score(
            validation["label"].tolist(), fitted.predict(validation["text"].tolist()), labels
        )
        if f1.macro_f1 > best_f1:
            best, best_f1 = fitted, f1.macro_f1
    assert best is not None
    return best
