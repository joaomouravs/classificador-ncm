"""Publishing to the Hugging Face Hub: the dataset card and the upload.

The card is generated from data/dataset/stats.json, so its numbers are always the ones
of the files it ships with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ncm.chapters import title
from ncm.evaluate import Comparison, compare

DATASET_NAME = "pncp-ncm-chapters"
SPLITS = ("train", "validation", "test")


def _size_category(n: int) -> str:
    for limit, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < limit:
            return name
    return "100K<n<1M"


def dataset_card(stats: dict[str, Any], collected: tuple[str, str], repo_id: str) -> str:
    """The README.md of the dataset repository."""
    total = sum(stats["splits"].values())
    distribution = stats["distribution"]
    first, last = collected
    header = "\n".join(
        [
            "---",
            "language:",
            "- pt",
            "license: other",
            "license_name: brazilian-open-government-data",
            "license_link: https://www.gov.br/pncp/pt-br",
            "task_categories:",
            "- text-classification",
            "tags:",
            "- ncm",
            "- public-procurement",
            "- brazil",
            "size_categories:",
            f"- {_size_category(total)}",
            "configs:",
            "- config_name: default",
            "  data_files:",
            *[f"  - split: {s}\n    path: data/{s}.parquet" for s in SPLITS],
            "---",
        ]
    )
    split_rows = "\n".join(f"| {s} | {stats['splits'][s]:,} |" for s in SPLITS)
    class_rows = "\n".join(
        f"| `{c}` | {title(c)} | "
        + " | ".join(f"{distribution[s].get(c, 0):,}" for s in SPLITS)
        + " |"
        for c in stats["classes"]
    )
    test_days = ", ".join(stats["test_days"])
    return f"""{header}

# NCM chapters of Brazilian public procurement items

Item descriptions from Brazilian public tenders (*pregões eletrônicos*), each labelled
with the chapter of its NCM code (Nomenclatura Comum do Mercosul, the first two digits:
30 is pharmaceuticals, 85 electrical equipment, 90 optical and medical instruments).

The descriptions come from the PNCP (Portal Nacional de Contratações Públicas), as the
buying agencies published them. The NCM codes come with the federal goods catalogue
(CATMAT) entry each item was bought under. Built for
[classificador-ncm](https://github.com/joaomouravs/classificador-ncm).

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
```

## Fields

| Field | Description |
| --- | --- |
| `text` | The item description, as the first tender that used it wrote it. |
| `label` | An NCM chapter (`"30"`), or `"outros"` if it has under 50 training examples. |
| `chapter` | The NCM chapter itself, also for rows labelled `outros`. |
| `catalog_code` | The CATMAT code of the catalogue item; group rows by it in cross-validation. |
| `occurrences` | How many collected items share this description. |
| `first_seen` | The publication day of the first tender that used it. |

## How it was built

- **Source.** Every *pregão eletrônico* published nationwide from {first} to {last},
  and every item of each: {stats["items"]:,} items, {stats["material_items"]:,} of them
  goods, and only {stats["labelled_items"]:,}
  ({stats["labelled_items"] / stats["material_items"]:.0%} of the goods) with an NCM code.
- **The label belongs to the catalogue.** {stats["labelled_with_catalog"]:,} of the
  {stats["labelled_items"]:,} labelled items carry a CATMAT code, and a catalogue code
  has the same NCM code in every tender that uses it. The NCM is not the buyer's choice:
  it comes with the catalogue entry, which is why agencies never disagree on it
  ({stats["conflicting_descriptions"]:,} conflicting descriptions).
- **One row per description.** Descriptions are compared after lowercasing, removing
  accents and collapsing spaces; {stats["distinct_descriptions"]:,} remain, from
  {stats["catalog_items"]:,} catalogue items, and {stats["repeated_descriptions"]:,} of
  them appear in more than one item.
- **Splits by catalogue item.** Agencies reformat a catalogue item's text ("material:
  aço inoxidável" against "Material:Aço Inoxidável"), so all the descriptions of one
  catalogue item stay in the same split; split by text alone, a third of the test set
  would have been catalogue items already in training. Test is every catalogue item
  first seen on the last day collected ({test_days}), so it looks like the tenders that
  come next. Validation is a stable 10% of the rest, chosen by a hash of the code.

| Split | Rows |
| --- | --: |
{split_rows}

## Classes

| Label | Chapter | Train | Validation | Test |
| --- | --- | --: | --: | --: |
{class_rows}

## Limitations

- **Catalogue text, not free text.** Every labelled description is a catalogue entry,
  more or less reformatted. Most goods are bought without a catalogue code, described in
  the agency's own words, and those carry no NCM at all; a model trained here will do
  worse on them than the test set says.
- **The catalogue's mapping is taken as truth.** A CATMAT entry assigned to the wrong
  NCM code would be wrong in every row that uses it.
- **A short window.** {first} to {last}: seasonal purchases outside it are missing.

## License

PNCP data is Brazilian open government data, published under the access to information
law (Lei nº 12.527/2011). This derived dataset is shared on the same terms.
"""


def publish_dataset(dataset_dir: Path, repo_id: str, token: str, card: str) -> str:
    """Create (or update) the public dataset repository and upload splits and card."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    url = api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    (dataset_dir / "README.md").write_text(card, encoding="utf-8", newline="\n")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
        allow_patterns=[f"{s}.parquet" for s in SPLITS],
        path_in_repo="data",
        commit_message="Upload splits",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=dataset_dir / "README.md",
        path_in_repo="README.md",
        commit_message="Dataset card",
    )
    return str(url)


MODEL_NAME = "ncm-chapter-bertimbau"


def _versus(name: str, c: Comparison, subject: str = "") -> str:
    """One sentence on whether the gap to the runner-up is more than test-set noise."""

    def verdict(metric: str, values: tuple[float, float, float]) -> str:
        mean, low, high = (100 * v for v in values)
        side = "a real edge" if low > 0 else ("a real gap" if high < 0 else "within noise")
        return f"{metric} {mean:+.1f} points (95% interval {low:+.1f} to {high:+.1f}, {side})"

    return (
        f"{f'{subject} against' if subject else 'Against'} {name}, "
        "a paired bootstrap over the test items gives "
        f"{verdict('macro-F1', c.macro_f1)} and {verdict('accuracy', c.accuracy)}."
    )


def model_card(
    result: dict[str, Any],
    others: list[dict[str, Any]],
    dataset_repo: str,
    repo_id: str,
    truth: list[str] | None = None,
    quantization: dict[str, Any] | None = None,
) -> str:
    """The README.md of the model repository, from results/bertimbau.json and the other
    models' results, so the card shows what the model is compared against. With the test
    labels, it also says whether the gap to the best other model is more than noise;
    with the quantization report, what the int8 ONNX export costs."""
    params = result["params"]
    test = result["test"]
    browser = ""
    if quantization:
        browser = (
            "\n## In the browser\n\n"
            "`onnx/model_quantized.onnx` is an int8 export (dynamic quantization, one scale "
            "per output channel, chosen on validation over one per tensor) for "
            "[transformers.js](https://huggingface.co/docs/transformers.js): "
            + quantization_facts(quantization)
            + "\n"
        )
    runner_up = max(
        (d for d in others if truth and len(d.get("test_predictions", [])) == len(truth)),
        key=lambda d: d["test"]["macro_f1"],
        default=None,
    )
    versus = ""
    if truth and runner_up:
        gap = compare(truth, result["test_predictions"], runner_up["test_predictions"])
        versus = _versus(runner_up["model"], gap)
    header = "\n".join(
        [
            "---",
            "language:",
            "- pt",
            "license: mit",
            "library_name: transformers",
            "pipeline_tag: text-classification",
            f"base_model: {params['base']}",
            "datasets:",
            f"- {dataset_repo}",
            "metrics:",
            "- f1",
            "- accuracy",
            "tags:",
            "- ncm",
            "- public-procurement",
            "- brazil",
            "---",
        ]
    )
    rows = "\n".join(
        f"| {d['model']} | {d['test']['macro_f1']:.3f} | {d['test']['accuracy']:.3f} |"
        for d in sorted([result, *others], key=lambda d: -d["test"]["macro_f1"])
    )
    classes = "\n".join(
        f"| `{label}` | {title(label)} | {c['f1']:.2f} | {c['support']} |"
        for label, c in test["per_class"].items()
        if c["support"]
    )
    return f"""{header}

# NCM chapter classifier (BERTimbau)

Reads the description of a good bought by a Brazilian public body and predicts its NCM
chapter (Nomenclatura Comum do Mercosul, the first two digits of the code): one of 14
chapters, or `outros` for the rarer ones. Fine-tuned from
[{params["base"]}](https://huggingface.co/{params["base"]}) on
[{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}). Code, evaluation and
the comparison with other models:
[classificador-ncm](https://github.com/joaomouravs/classificador-ncm).

```python
from transformers import pipeline

classify = pipeline("text-classification", model="{repo_id}", top_k=3)
classify("Seringa descartável 10 ml, bico luer lock, estéril")
```

## Results

On the test set: {test["examples"]:,} descriptions of catalogue items first seen on the
last day collected, none of them, in any wording, in training.

| Model | Macro-F1 | Accuracy |
| --- | --: | --: |
{rows}

{versus} Details in the
[report](https://github.com/joaomouravs/classificador-ncm/blob/main/results/REPORT.md).

| Label | Chapter | Test F1 | Test examples |
| --- | --- | --: | --: |
{classes}

## Training

{params["epochs"]:g} epochs, learning rate {params["lr"]:g}, batch {params["batch"]}, sequences
cut at {params["max_length"]} tokens, class-weighted loss, best epoch chosen by validation
macro-F1. {result["train_seconds"] / 60:.1f} minutes on one {result["hardware"]}.
{browser}
## Limitations

- **Catalogue text only.** Every training description is a CATMAT catalogue entry. Most
  goods are bought without one, described in the agency's own words, and the model will
  do worse on them than the test set says.
- **Chapter level only**, and only the 14 chapters with enough examples; everything else
  is `outros`.
- **The catalogue's NCM mapping is taken as truth**, including where it follows its own
  convention (printer ink under printers, chapter 84).
"""


def publish_model(model_dir: Path, repo_id: str, token: str, card: str) -> str:
    """Create (or update) the public model repository and upload weights and card."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    url = api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        allow_patterns=["config.json", "model.safetensors", "tokenizer*.json"],
        commit_message="Upload fine-tuned model",
    )
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Model card",
    )
    return str(url)


def publish_onnx(onnx_dir: Path, repo_id: str, token: str) -> None:
    """Add the int8 ONNX export to the model repository, where transformers.js looks for
    it (onnx/model_quantized.onnx), so the browser demo can load the model."""
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        repo_id=repo_id,
        path_or_fileobj=onnx_dir / "onnx" / "model_quantized.onnx",
        path_in_repo="onnx/model_quantized.onnx",
        commit_message="ONNX export, int8 per-channel, for transformers.js",
    )


def update_model_card(repo_id: str, token: str, card: str) -> None:
    from huggingface_hub import HfApi

    HfApi(token=token).upload_file(
        repo_id=repo_id,
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        commit_message="Model card",
    )


SPACE_NAME = "ncm-chapter-classifier"
# A free Space can only be static (Gradio and Docker Spaces on the free CPU now need a
# paid plan), so the demo is one HTML page that runs the model in the browser.
SPACE_PAGE = Path(__file__).resolve().parents[2] / "space" / "index.html"


def quantization_facts(report: dict[str, Any]) -> str:
    """What running the model in the browser costs, from results/quantization/."""
    chosen, fp32 = report[report["chosen"]], report["fp32"]
    return (
        f"{chosen['megabytes']:.0f} MB against {fp32['megabytes']:.0f} MB in fp32, and "
        f"it agrees with the full-precision model on "
        f"{100 * chosen['agrees_with_pytorch']:.1f}% of the test items: macro-F1 "
        f"{chosen['macro_f1']:.3f} and accuracy {chosen['accuracy']:.3f}."
    )


def space_summary(
    results: list[dict[str, Any]],
    truth: list[str],
    model_repo: str,
    dataset_repo: str,
    quantization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The numbers the Space shows, from results/: every model on the full test set,
    the comparison of the best two, and what quantizing cost."""
    full = sorted(
        (d for d in results if len(d.get("test_predictions", [])) == len(truth)),
        key=lambda d: -d["test"]["macro_f1"],
    )
    versus = ""
    if len(full) >= 2:
        gap = compare(truth, full[0]["test_predictions"], full[1]["test_predictions"])
        versus = _versus(full[1]["model"], gap, subject=full[0]["model"])
    return {
        "test_items": len(truth),
        "classes": list(full[0]["test"]["per_class"]) if full else [],
        "model": model_repo,
        "dataset": dataset_repo,
        "models": [
            {
                "model": d["model"],
                "macro_f1": d["test"]["macro_f1"],
                "accuracy": d["test"]["accuracy"],
            }
            for d in full
        ],
        "versus": versus,
        "quantized": f"The page runs an int8 export: {quantization_facts(quantization)}"
        if quantization
        else "",
    }


def space_page(summary: dict[str, Any], template: Path = SPACE_PAGE) -> str:
    """The page with the summary and the chapter titles written in. "</" is escaped so
    no title can close the script element that holds it."""
    from ncm.chapters import CHAPTERS

    def embed(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")

    page = template.read_text(encoding="utf-8")
    return page.replace("__SUMMARY__", embed(summary)).replace("__CHAPTERS__", embed(CHAPTERS))


def space_readme(model_repo: str, dataset_repo: str) -> str:
    return f"""---
title: NCM Chapter Classifier
emoji: 📦
colorFrom: green
colorTo: blue
sdk: static
pinned: false
license: mit
models:
- {model_repo}
datasets:
- {dataset_repo}
short_description: NCM chapter of a public procurement item, in the browser
---

Type the description of a good bought by a Brazilian public body and see the NCM chapter
a fine-tuned BERTimbau predicts, running in the browser with transformers.js. Code and
evaluation: [classificador-ncm](https://github.com/joaomouravs/classificador-ncm).
"""


def publish_space(repo_id: str, token: str, files: dict[str, str]) -> str:
    """Create (or update) the public static Space with the generated files, in one
    commit so it rebuilds once."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    url = api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    operations = [CommitOperationAdd(p, content.encode("utf-8")) for p, content in files.items()]
    api.create_commit(
        repo_id=repo_id, repo_type="space", operations=operations, commit_message="Deploy"
    )
    return str(url)
