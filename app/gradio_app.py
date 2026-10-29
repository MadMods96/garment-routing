"""Gradio demo. CPU inference works once a checkpoint path is set.

Example photographs are loaded only from ``app/examples`` when that folder
contains a ``manifest.json`` produced from the test split. This file does not
ship garment images.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from PIL import Image

EXAMPLES = ROOT / "app" / "examples" / "manifest.json"
_CACHED: tuple | None = None


def _predict(front: Image.Image | None, back: Image.Image | None, passport_text: str) -> tuple[str, str, str]:
    if front is None or back is None:
        return "Upload a front and a back photograph.", "", ""
    checkpoint = os.environ.get("GARMENT_CHECKPOINT", "")
    if not checkpoint or not Path(checkpoint).is_file():
        return (
            "No checkpoint is configured. Set GARMENT_CHECKPOINT to a trained weights file. "
            "The demo will not invent a route.",
            "",
            "",
        )
    document = None
    if passport_text.strip():
        try:
            document = json.loads(passport_text)
        except json.JSONDecodeError:
            return "The passport box is not valid JSON.", "", ""
    from garmentroute.infer import assess, load_model

    global _CACHED
    if _CACHED is None or _CACHED[0] != checkpoint:
        _CACHED = (checkpoint, *load_model(Path(checkpoint)))
    _path, model, config, device = _CACHED
    result = assess(
        model,
        config,
        front,
        back,
        document,
        Path(os.environ.get("GARMENT_ROUTER", ROOT / "configs" / "router.yaml")),
        device,
        temperature=float(os.environ.get("GARMENT_TEMPERATURE", "1")),
        abstain_threshold=float(os.environ["GARMENT_ABSTAIN"]) if os.environ.get("GARMENT_ABSTAIN") else None,
    )
    if result["route"] == "HUMAN_REVIEW":
        headline = "Sent to human review"
    else:
        headline = f"Route: {result['route']}"
    reasons = "\n".join(f"- {reason}" for reason in result["reasons"])
    details = json.dumps(
        {
            "condition": result["condition"],
            "pilling": result["pilling"],
            "stains": result["stains"],
            "holes": result["holes"],
            "usage_class": result["usage_class"],
            "confidence": result["confidence"],
            "usage_probabilities": result["usage_probabilities"],
            "alternatives": result["alternatives"],
        },
        indent=2,
    )
    bar = f"Usage confidence {result['usage_probabilities'][result['usage_class']]:.2f}"
    return f"{headline}\n{bar}", reasons, details


def _examples() -> list[list[str]] | None:
    if not EXAMPLES.is_file():
        return None
    manifest = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    rows = []
    for item in manifest:
        front = ROOT / item["front"]
        back = ROOT / item["back"]
        if front.is_file() and back.is_file():
            rows.append([str(front), str(back), item.get("passport", "")])
    return rows or None


def build() -> gr.Blocks:
    with gr.Blocks(title="Second-life garment routing") as demo:
        gr.Markdown(
            "Upload the front and the back of a used garment. "
            "The route uses the photograph for condition and damage, and the optional "
            "passport for fibre. Router weights are illustrative placeholders, not LCA results. "
            "Dataset images, when you add examples, are CC-BY 4.0 "
            "(Nauman / RISE / Wargön Innovation / Myrorna)."
        )
        with gr.Row():
            front = gr.Image(label="Front", type="pil")
            back = gr.Image(label="Back", type="pil")
        passport = gr.Textbox(label="Optional mock passport JSON", lines=6)
        button = gr.Button("Assess")
        headline = gr.Textbox(label="Decision")
        reasons = gr.Textbox(label="Reasons", lines=6)
        details = gr.Code(label="Predictions", language="json")
        button.click(_predict, [front, back, passport], [headline, reasons, details])
        examples = _examples()
        if examples:
            gr.Examples(examples=examples, inputs=[front, back, passport])
    return demo


if __name__ == "__main__":
    build().launch()
