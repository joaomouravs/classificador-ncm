from __future__ import annotations

from ncm.chapters import CHAPTERS, OTHER_TITLE, title
from ncm.evaluate import score
from ncm.hub import dataset_card, model_card, space_page, space_readme, space_summary

STATS = {
    "items": 1000,
    "material_items": 800,
    "labelled_items": 300,
    "distinct_descriptions": 250,
    "repeated_descriptions": 40,
    "conflicting_descriptions": 0,
    "labelled_with_catalog": 299,
    "catalog_items": 200,
    "splits": {"train": 200, "validation": 20, "test": 30},
    "test_days": ["2026-10-23"],
    "classes": ["30", "85", "outros"],
    "distribution": {
        "train": {"30": 120, "85": 60, "outros": 20},
        "validation": {"30": 12, "85": 8},
        "test": {"30": 15, "85": 10, "outros": 5},
    },
}


def test_every_chapter_but_the_reserved_one_has_a_title():
    assert set(CHAPTERS) == {f"{n:02d}" for n in range(1, 98)} - {"77"}
    assert title("85").startswith("Máquinas")
    assert title("outros") == OTHER_TITLE


def test_card_numbers_come_from_the_stats():
    card = dataset_card(STATS, ("2026-10-19", "2026-10-23"), "someone/pncp-ncm-chapters")
    assert card.startswith("---\nlanguage:\n- pt\n")
    assert "- n<1K" in card
    assert "  - split: test\n    path: data/test.parquet" in card
    assert 'load_dataset("someone/pncp-ncm-chapters")' in card
    assert "| `30` | Produtos farmacêuticos | 120 | 12 | 15 |" in card
    # A class absent from a split shows zero instead of breaking the table.
    assert "| `outros` | Outros capítulos (poucos exemplos no treino) | 20 | 0 | 5 |" in card
    assert "from 2026-10-19 to 2026-10-23" in card
    assert "(38% of the goods)" in card
    assert "299 of the\n  300 labelled items carry a CATMAT code" in card


def _result(name, predictions, truth, classes):
    return {
        "model": name,
        "params": {
            "base": "neuralmind/bert-base-portuguese-cased",
            "epochs": 3,
            "lr": 3e-5,
            "batch": 32,
            "max_length": 128,
        },
        "train_seconds": 218.3,
        "hardware": "Tesla T4",
        "test": score(truth, predictions, classes).as_dict(),
        "test_predictions": predictions,
    }


def test_model_card_compares_with_the_runner_up_from_the_results():
    classes = ["30", "85", "outros"]
    truth = ["30", "85"] * 100
    model = _result("bertimbau", list(truth), truth, classes)
    weaker = _result("tfidf-logreg", ["30"] * 200, truth, classes)
    gemini = _result("gemini-zero-shot", ["85"] * 10, truth[:10], classes)  # a sample only

    card = model_card(
        model, [weaker, gemini], "u/pncp-ncm-chapters", "u/ncm-chapter-bertimbau", truth
    )

    assert "base_model: neuralmind/bert-base-portuguese-cased" in card
    assert 'pipeline("text-classification", model="u/ncm-chapter-bertimbau"' in card
    assert "Against tfidf-logreg, a paired bootstrap" in card
    assert "a real edge" in card
    assert "| `30` | Produtos farmacêuticos | 1.00 | 100 |" in card
    assert not any(line.startswith("| `outros`") for line in card.splitlines())  # no examples
    assert "3.6 minutes on one Tesla T4" in card


QUANTIZATION = {
    "chosen": "int8-per-channel",
    "fp32": {"megabytes": 415.8, "macro_f1": 0.9136, "accuracy": 0.9491},
    "int8-per-channel": {
        "megabytes": 105.0,
        "macro_f1": 0.9125,
        "accuracy": 0.9477,
        "agrees_with_pytorch": 0.9933,
    },
}


def _summary():
    classes = ["30", "85", "outros"]
    truth = ["30", "85"] * 100
    results = [
        _result("tfidf-logreg", ["30"] * 200, truth, classes),
        _result("bertimbau", list(truth), truth, classes),
        _result("gemini-zero-shot", ["85"] * 10, truth[:10], classes),  # a sample only
    ]
    return space_summary(results, truth, "u/model", "u/dataset", QUANTIZATION)


def test_space_summary_ranks_full_test_runs_and_skips_samples():
    summary = _summary()

    assert [m["model"] for m in summary["models"]] == ["bertimbau", "tfidf-logreg"]
    assert summary["test_items"] == 200
    assert summary["versus"].startswith("bertimbau against tfidf-logreg, a paired bootstrap")
    assert summary["quantized"].startswith("The page runs an int8 export: 105 MB against 416 MB")
    assert "on 99.3% of the test items" in summary["quantized"]


def test_space_page_embeds_the_numbers_and_cannot_be_broken_by_a_title(tmp_path):
    template = tmp_path / "index.html"
    template.write_text(
        '<script type="application/json" id="summary">__SUMMARY__</script>'
        '<script type="application/json" id="chapters">__CHAPTERS__</script>',
        encoding="utf-8",
    )
    summary = _summary() | {"versus": "a </script> in the text"}

    page = space_page(summary, template)

    assert "__SUMMARY__" not in page and "__CHAPTERS__" not in page
    assert page.count("</script>") == 2  # the one in the text was escaped
    assert '"model": "u/model"' in page
    assert "Produtos farmacêuticos" in page


def test_the_real_page_has_both_placeholders():
    page = space_page(_summary())
    assert "__SUMMARY__" not in page and "__CHAPTERS__" not in page
    assert "@huggingface/transformers@4.3.1" in page
    assert "max_length: MAX_LENGTH" in page


def test_space_readme_is_static_and_links_model_and_dataset():
    readme = space_readme("u/ncm-chapter-bertimbau", "u/pncp-ncm-chapters")
    assert readme.startswith("---\ntitle: NCM Chapter Classifier\n")
    assert "sdk: static\n" in readme
    assert "models:\n- u/ncm-chapter-bertimbau\n" in readme
    assert "datasets:\n- u/pncp-ncm-chapters\n" in readme


def test_model_card_reports_the_int8_export():
    classes = ["30", "85", "outros"]
    truth = ["30", "85"] * 100
    model = _result("bertimbau", list(truth), truth, classes)
    card = model_card(model, [], "u/d", "u/m", truth, QUANTIZATION)
    assert "## In the browser" in card
    assert "`onnx/model_quantized.onnx` is an int8 export" in card
    assert "transformers.js): 105 MB against 416 MB in fp32" in card
