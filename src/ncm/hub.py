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
