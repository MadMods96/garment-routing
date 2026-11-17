"""Public Gradio demo for the front-only garment routing model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from garmentroute.infer import assess, load_model
from garmentroute.router import HUMAN_REVIEW

MODEL_ID = "MadnessOverloaded/garment-routing-front"
ATTRIBUTION = (
    "Photographs and labels: Clothing Dataset for Second-Hand Fashion, "
    "Zenodo DOI 10.5281/zenodo.13788681, CC-BY 4.0 "
    "(Wargön Innovation, Myrorna, RISE / CISUTAC)."
)
_OTHER_ROUTES = {"Repair", "Remake", "Energy recovery"}


def _resolve_weights() -> Path:
    local = ROOT / "model.fp16.pt"
    if local.is_file():
        return local
    parent = ROOT.parent / "model.fp16.pt"
    if parent.is_file():
        return parent
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MODEL_ID, "model.fp16.pt"))


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{round(float(value) * 100)}%"


def _collapse(route: str) -> str:
    return "Other" if route in _OTHER_ROUTES else route


def _open_image(value: Image.Image | str | Path | None) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    return Image.open(value).convert("RGB")


def _front_name(value: Image.Image | str | Path | None) -> str | None:
    if isinstance(value, (str, Path)):
        return Path(value).name
    return None


WEIGHTS = _resolve_weights()
MODEL, CONFIG, DEVICE, META = load_model(WEIGHTS)
ROUTER = ROOT / "router.yaml"
TEMPERATURE = float(META.get("temperature") or 1.0)
THRESHOLD = META.get("abstain_threshold")
EXAMPLES_DIR = ROOT / "examples"
MANIFEST = (
    json.loads((EXAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))
    if (EXAMPLES_DIR / "manifest.json").is_file()
    else {"examples": []}
)
BY_FRONT = {row["front"]: row for row in MANIFEST.get("examples", [])}


def _disagree_line(model_usage: str, rule_route: str, final_route: str, confidence: float, reasons: list[str] | None = None) -> str:
    reason_text = " ".join(reasons or [])
    if final_route == HUMAN_REVIEW and THRESHOLD is not None and confidence < float(THRESHOLD):
        return (
            f"Calibrated confidence {_pct(confidence)} is below the abstain threshold "
            f"{_pct(THRESHOLD)}, so the final route is human review."
        )
    if final_route == HUMAN_REVIEW and "hard-rule" in reason_text:
        return "Hard-rule safety override contradicts the model — sent to human review."
    if final_route == model_usage and rule_route and _collapse(rule_route) != model_usage:
        return (
            f"Final route follows the model ({model_usage}). "
            f"Soft rules would have preferred {rule_route} (explanation only)."
        )
    if model_usage == "Other" and rule_route in _OTHER_ROUTES:
        return (
            f"The 4-way head collapses Repair/Remake/Energy recovery into Other; "
            f"soft rules preferred {rule_route}."
        )
    return ""


def predict(front, back):
    if front is None:
        raise gr.Error("Upload a front photograph.")
    front_img = _open_image(front)
    back_img = _open_image(back)
    started = time.perf_counter()
    result = assess(
        MODEL,
        CONFIG,
        front_img,
        back_img,
        passport=None,
        router_config_path=ROUTER,
        device=DEVICE,
        temperature=TEMPERATURE,
        abstain_threshold=THRESHOLD,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    final_route = result["route"]
    model_usage = result["usage_class"]
    rule_route = result["rule_route"]
    confidence = float(result["usage_confidence"])
    expert = BY_FRONT.get(_front_name(front) or "")
    if expert:
        expert_md = (
            f"**Expert sorter (dataset):** usage **{expert['usage_class']}**, "
            f"condition **{expert['condition']}/5**"
        )
    else:
        expert_md = "*No expert label for this upload (examples show the sorter’s labels).*"

    warning = result.get("ood_warning")
    final_md = ""
    if warning:
        final_md += f"### ⚠️ OOD warning\n\n**{warning}**\n\n"
    final_md += f"## Final route: `{final_route}`"
    disagree = _disagree_line(
        model_usage, rule_route, final_route, confidence, list(result.get("reasons") or [])
    )
    details = (
        f"Predicted condition: {result['condition']}/5\n"
        f"Stains: {result['stains']}\n"
        f"Holes: {result['holes']}\n"
        f"Pilling: {result['pilling']}/5\n"
        f"Calibrated usage confidence: {_pct(confidence)}\n"
        f"Abstain threshold (fit on val): {_pct(THRESHOLD)}\n"
        f"CPU inference: {elapsed_ms:.0f} ms\n"
        f"\n{ATTRIBUTION}"
    )
    reasons = "\n".join(f"- {reason}" for reason in result["reasons"]) or "- none"
    return (
        final_md,
        model_usage,
        rule_route,
        disagree,
        expert_md,
        details,
        reasons,
    )


def _enable_submit(front):
    return gr.update(interactive=front is not None)


EXAMPLE_ROWS = []
EXAMPLE_LABELS = []
for row in MANIFEST.get("examples", []):
    front_path = EXAMPLES_DIR / row["front"]
    back_path = EXAMPLES_DIR / row["back"]
    if front_path.is_file() and back_path.is_file():
        EXAMPLE_ROWS.append([str(front_path), str(back_path)])
        EXAMPLE_LABELS.append(
            f"{row['usage_class']} · condition {row['condition']}/5 · {row['item_id']}"
        )


with gr.Blocks(title="Second-life garment routing") as demo:
    gr.Markdown(
        "# Second-life garment routing\n\n"
        "Front-only ConvNeXt-Tiny on Swedish sorting-facility photographs. "
        "Back is optional. Low calibrated confidence is sent to human review.\n\n"
        f"{ATTRIBUTION}"
    )
    with gr.Row():
        front_in = gr.Image(type="filepath", label="Front photograph")
        back_in = gr.Image(type="filepath", label="Back photograph (optional)")
    expert_out = gr.Markdown("*Select an example or upload a front photograph.*")
    submit = gr.Button("Assess", variant="primary", interactive=False)
    final_out = gr.Markdown("## Final route: —")
    with gr.Row():
        model_out = gr.Textbox(label="Model usage class", interactive=False)
        router_out = gr.Textbox(label="Rule router route", interactive=False)
    disagree_out = gr.Markdown("")
    details_out = gr.Textbox(label="Details", lines=10, interactive=False)
    reasons_out = gr.Textbox(label="Reasons", lines=6, interactive=False)

    outputs = [final_out, model_out, router_out, disagree_out, expert_out, details_out, reasons_out]
    front_in.change(_enable_submit, inputs=front_in, outputs=submit)
    submit.click(predict, inputs=[front_in, back_in], outputs=outputs)

    if EXAMPLE_ROWS:
        gr.Examples(
            examples=EXAMPLE_ROWS,
            inputs=[front_in, back_in],
            outputs=outputs,
            fn=predict,
            cache_examples=False,
            run_on_click=True,
            example_labels=EXAMPLE_LABELS,
            label="Examples (held-out test garments)",
        )

if __name__ == "__main__":
    demo.queue().launch(share=True)
