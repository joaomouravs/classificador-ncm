from __future__ import annotations

import json
import random

import pandas as pd
import pytest

from ncm.baselines import majority, tfidf
from ncm.evaluate import compare, mcnemar, score
from ncm.results import build_report, record

LABELS = ["30", "40", "90", "outros"]


def test_a_class_never_predicted_still_counts_against_macro_f1():
    truth = ["90", "90", "90", "30"]
    predicted = ["90", "90", "90", "90"]

    scores = score(truth, predicted, LABELS)

    assert scores.accuracy == 0.75
    # 90 has F1 6/7 and 30, never predicted, has 0. 40 and outros have no examples here,
    # so they are left out of the average rather than counted as 0 for every model.
    assert scores.macro_f1 == pytest.approx((6 / 7) / 2)
    assert scores.per_class["30"]["recall"] == 0
    assert scores.per_class["40"]["support"] == 0
    assert scores.confusions == [("30", "90", 1)]


WORDS = {
    "30": ["dipirona", "amoxicilina", "paracetamol", "ibuprofeno", "omeprazol"],
    "40": ["luva latex", "pneu aro", "camara de ar", "borracha", "mangueira"],
    "90": ["seringa", "termometro", "estetoscopio", "cateter", "oximetro"],
    "outros": ["caneta", "papel a4", "grampeador", "pasta", "envelope"],
}


def dataset(seed: int, size: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(size):
        label = rng.choice(LABELS)
        rows.append(
            {"text": f"{rng.choice(WORDS[label])} {rng.randint(1, 500)} unidade", "label": label}
        )
    return pd.DataFrame(rows)


@pytest.fixture
def splits():
    return {"train": dataset(1, 400), "validation": dataset(2, 80), "test": dataset(3, 80)}


def test_tfidf_beats_majority_on_separable_text(splits):
    floor = majority(splits["train"])
    model = tfidf(splits["train"], splits["validation"], LABELS)

    texts, truth = splits["test"]["text"].tolist(), splits["test"]["label"].tolist()
    assert score(truth, floor.predict(texts), LABELS).macro_f1 < 0.2
    assert score(truth, model.predict(texts), LABELS).macro_f1 > 0.9
    assert model.params["C"] in (1.0, 4.0, 16.0)


def test_results_and_report(splits, tmp_path):
    model = tfidf(splits["train"], splits["validation"], LABELS)
    record("tfidf-logreg", model.predict, splits, LABELS, {"params": model.params}, tmp_path)
    floor = majority(splits["train"])
    record("majority", floor.predict, splits, LABELS, {}, tmp_path)

    saved = json.loads((tmp_path / "tfidf-logreg.json").read_text(encoding="utf-8"))
    assert saved["test"]["examples"] == 80
    assert saved["ms_per_1000_items"] > 0

    report = build_report(tmp_path)
    rows = [
        line
        for line in report.splitlines()
        if line.startswith("| tfidf") or line.startswith("| majority")
    ]
    assert rows[0].startswith("| tfidf-logreg")  # best first
    assert "## majority" in report


def test_mcnemar_exact():
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(5, 5) == 1.0
    # 30 against 51 overflowed numpy's integers; the exact value is about 0.026.
    assert mcnemar(51, 30) == pytest.approx(0.0257, abs=1e-4)
    assert mcnemar(10, 0) == pytest.approx(2 / 2**10)


def test_compare_separates_a_real_gap_from_noise():
    truth = ["30", "90"] * 200
    good = list(truth)
    noisy = [("90" if t == "30" else "30") if i % 5 == 0 else t for i, t in enumerate(truth)]

    clear = compare(truth, good, noisy, rounds=300)
    assert clear.only_a == 80 and clear.only_b == 0
    assert clear.accuracy[1] > 0  # the whole interval is above zero
    assert clear.mcnemar_p < 0.001

    same = compare(truth, good, list(good), rounds=300)
    assert same.macro_f1 == (0.0, 0.0, 0.0)
    assert same.mcnemar_p == 1.0


def test_report_compares_the_two_best(splits, tmp_path):
    model = tfidf(splits["train"], splits["validation"], LABELS)
    record("tfidf-logreg", model.predict, splits, LABELS, {"params": model.params}, tmp_path)
    record("majority", majority(splits["train"]).predict, splits, LABELS, {}, tmp_path)

    report = build_report(tmp_path)
    assert "## tfidf-logreg against majority" in report
    assert "McNemar's exact test p = 0.000" in report
