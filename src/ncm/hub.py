"""Publishing to the Hugging Face Hub: the dataset card and the upload.

The card is generated from data/dataset/stats.json, so its numbers are always the ones
of the files it ships with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ncm.chapters import title

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

The descriptions and codes were written by the buying agencies themselves and published
on the PNCP (Portal Nacional de Contratações Públicas). Built for
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
| `occurrences` | How many collected items share this description. |
| `first_seen` | The publication day of the first tender that used it. |

## How it was built

- **Source.** Every *pregão eletrônico* published nationwide from {first} to {last},
  and every item of each: {stats["items"]:,} items, {stats["material_items"]:,} of them
  goods, {stats["labelled_items"]:,} with an NCM code.
- **One row per description.** Agencies reuse the government catalogue's text, so the
  same description appears in many tenders. Descriptions are compared after lowercasing,
  removing accents and collapsing spaces; {stats["distinct_descriptions"]:,} remain, and
  {stats["repeated_descriptions"]:,} of them appear more than once.
- **Label.** The chapter most tenders gave the description. In
  {stats["conflicting_descriptions"]:,} repeated descriptions the agencies disagreed on
  the chapter, a lower bound on the label noise.
- **Splits.** Test is every description first seen on the last day collected
  ({test_days}): no test text ever appears in training, and the test set looks like the
  tenders that come next. Validation is a stable 10% of the rest, chosen by a hash of
  the description.

| Split | Rows |
| --- | --: |
{split_rows}

## Classes

| Label | Chapter | Train | Validation | Test |
| --- | --- | --: | --: | --: |
{class_rows}

## Limitations

- **The labels are the agencies' own.** Buyers pick NCM codes by hand, and some are
  wrong; a model's accuracy on this test set is bounded by that noise.
- **Only a third of items carry a code.** Agencies that fill the field may describe
  items differently from those that do not.
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
