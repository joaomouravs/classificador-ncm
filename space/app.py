"""Hugging Face Space: type an item description, see the NCM chapters two models predict.

The fine-tuned BERTimbau comes from the Hub; the TF-IDF model is a file in this Space.
Both were trained and evaluated in https://github.com/joaomouravs/classificador-ncm, and
the `ncm` package installed from there provides what the TF-IDF model needs to load
(its text normalisation) and the chapter titles.
"""

from __future__ import annotations

import json
import os

import gradio as gr
import joblib
from transformers import pipeline

from ncm.chapters import title

MODEL = os.environ.get("NCM_MODEL", "joaovux/ncm-chapter-bertimbau")
TFIDF = os.environ.get("NCM_TFIDF", "models/tfidf-logreg.joblib")
TOP = 5

bert = pipeline("text-classification", model=MODEL, top_k=TOP, device="cpu")
tfidf = joblib.load(TFIDF)


def named(label: str) -> str:
    return f"{label} · {title(label)}"


def predict(description: str) -> tuple[dict[str, float], dict[str, float]]:
    text = " ".join(description.split())
    if not text:
        return {}, {}
    bert_scores = {named(r["label"]): float(r["score"]) for r in bert(text)[0]}
    probabilities = tfidf.predict_proba([text])[0]
    ranked = sorted(zip(tfidf.classes_, probabilities, strict=True), key=lambda p: -p[1])
    tfidf_scores = {named(str(label)): float(p) for label, p in ranked[:TOP]}
    return bert_scores, tfidf_scores


EXAMPLES = [
    "Seringa descartável 10 ml, bico luer lock, estéril",
    "Cartucho toner impressora HP preto, compatível",
    "Abaixador de língua, material: plástico, comprimento: 14 cm",
    "Dipirona sódica 500 mg, comprimido",
    "Cadeira giratória com rodízios e braços reguláveis",
    "Notebook processador 8 núcleos, memória 16 GB, SSD 512 GB, tela 15,6 polegadas",
]

# Written by `ncm publish-space` from the repository's results/, so the numbers on the
# page are the evaluated ones.
with open(os.environ.get("NCM_SUMMARY", "summary.json"), encoding="utf-8") as f:
    SUMMARY = json.load(f)

ROWS = "\n".join(
    f"| {m['model']} | {m['macro_f1']:.3f} | {m['accuracy']:.3f} |" for m in SUMMARY["models"]
)
INTRO = f"""
# NCM chapter classifier

Type the description of a good bought by a Brazilian public body and see which **NCM
chapter** (the first two digits of the Mercosur product code) two models predict: one
of {len(SUMMARY["classes"]) - 1} chapters, or `outros` for the rarer ones.

On a held-out test set of {SUMMARY["test_items"]:,} catalogue descriptions:

| Model | Macro-F1 | Accuracy |
| --- | --: | --: |
{ROWS}

{SUMMARY["versus"]}
[Code and evaluation](https://github.com/joaomouravs/classificador-ncm) ·
[model](https://huggingface.co/{MODEL}) ·
[dataset](https://huggingface.co/datasets/{SUMMARY["dataset"]})

Both models learned from catalogue entries (CATMAT); descriptions in an agency's own
words, without a catalogue code, will be harder for them.
"""

with gr.Blocks(title="NCM chapter classifier") as demo:
    gr.Markdown(INTRO)
    description = gr.Textbox(label="Item description", lines=3, placeholder=EXAMPLES[0])
    button = gr.Button("Classify", variant="primary")
    with gr.Row():
        bert_out = gr.Label(label="BERTimbau, fine-tuned", num_top_classes=TOP)
        tfidf_out = gr.Label(label="TF-IDF + logistic regression", num_top_classes=TOP)
    gr.Examples(EXAMPLES, inputs=description)
    button.click(predict, inputs=description, outputs=[bert_out, tfidf_out])
    description.submit(predict, inputs=description, outputs=[bert_out, tfidf_out])

if __name__ == "__main__":
    demo.launch()
