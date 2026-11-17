"""Write the Hugging Face model card from reports/results.json and the demo pack."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.reporting import _fmt, _fmt_ci, _get


def main() -> None:
    results = json.loads((ROOT / "reports" / "results.json").read_text(encoding="utf-8"))
    front = _get(results, "front", "splits", "test", "after_temperature")
    cross = _get(results, "front", "splits", "cross_site", "after_temperature")
    fusion_test = _get(results, "fusion", "splits", "test", "after_temperature")
    fusion_cross = _get(results, "fusion", "splits", "cross_site", "after_temperature")
    gold = results.get("gold") or {}
    boot = results.get("bootstrap") or {}
    chosen = results.get("chosen_vision") or {}
    pack = ROOT / "demo" / "hf_model" / "model.fp16.pt"
    image_size = 384
    temperature = _get(results, "front", "temperature")
    abstain = _get(results, "front", "abstain_threshold", "threshold")
    if pack.exists():
        import torch

        ck = torch.load(pack, map_location="cpu", weights_only=False)
        cfg = ck.get("config") or {}
        image_size = int(cfg.get("image_size") or image_size)
        temperature = ck.get("temperature", temperature)
        abstain = ck.get("abstain_threshold", abstain)

    lines = [
        "---",
        "license: mit",
        "library_name: pytorch",
        "tags:",
        "- image-classification",
        "- fashion",
        "- circular-economy",
        "- garment-routing",
        "pipeline_tag: image-classification",
        "---",
        "",
        "# Second-life garment routing (front-only)",
        "",
        "Used textiles are sorted by hand into a next life: reuse, export, repair, remake, "
        "or recycling. This ConvNeXt-Tiny model looks at a **front** photograph, estimates "
        "condition and damage, and feeds a rule router that can abstain to human review. "
        "Built as a TU/e EngD portfolio prototype for digital textile ecosystem identification "
        "(Madhusudan Pareek). It is research / decision-support, not a certified sorter.",
        "",
        f"Pack: front-only fusion, image size {image_size}, float16 weights, "
        f"validation-fitted temperature {_fmt(temperature)}, "
        f"abstain threshold {_fmt(abstain)}.",
        "",
        "Code: [MadMods96/garment-routing](https://github.com/MadMods96/garment-routing).",
        "",
        "## Data",
        "",
        "Clothing Dataset for Second-Hand Fashion v3, Zenodo "
        "[10.5281/zenodo.13788681](https://doi.org/10.5281/zenodo.13788681), **CC-BY 4.0**. "
        "Expert labels from Wargön Innovation (stations 1–2) and Myrorna (station 3). "
        "Training uses stations 1–2 only. Station 3 and the gold set never enter training "
        "or temperature / threshold fitting.",
        "",
        "## Best vs other (real eval)",
        "",
        "Chosen model: **front-only** "
        f"({chosen.get('criterion', 'test usage macro-F1, 4 training classes')}).",
        "",
        "| Model | Test usage macro-F1 (4-class) | Station 3 usage macro-F1 |",
        "| --- | --- | --- |",
        "| **front-only (best / this repo)** | "
        + _fmt_ci(boot.get("test_usage_macro_f1"))
        + f" | {_fmt(_get(cross, 'usage', 'macro_f1'))} |",
        "| two-view concat | "
        + f"{_fmt(_get(fusion_test, 'usage', 'macro_f1'))} "
        + f"| {_fmt(_get(fusion_cross, 'usage', 'macro_f1'))} |",
        "| majority baseline | "
        + f"{_fmt(_get(results, 'majority', 'test', 'usage_class', 'macro_f1'))} "
        + f"| {_fmt(_get(results, 'majority', 'cross_site', 'usage_class', 'macro_f1'))} |",
        "",
        "Attribute LightGBM ceiling on test (photos unused, 6 expert classes): "
        f"{_fmt(_get(results, 'attribute_baselines', 'models', 'lightgbm', 'splits', 'test', 'macro_f1'))}.",
        "",
        "## Front-only detail",
        "",
        "| Split | Usage macro-F1 (4-class) | Balanced accuracy | Condition MAE | Condition within ±1 | ECE after T |",
        "| --- | --- | --- | --- | --- | --- |",
        "| test | "
        + _fmt_ci(boot.get("test_usage_macro_f1"))
        + f" | {_fmt(_get(front, 'usage', 'balanced_accuracy'))} "
        + f"| {_fmt(_get(front, 'condition', 'mae'))} "
        + f"| {_fmt(_get(front, 'condition', 'within_1_accuracy'))} "
        + f"| {_fmt(_get(front, 'calibration', 'ece'))} |",
        "| station 3 | "
        + f"{_fmt(_get(cross, 'usage', 'macro_f1'))} "
        + f"| {_fmt(_get(cross, 'usage', 'balanced_accuracy'))} "
        + f"| {_fmt(_get(cross, 'condition', 'mae'))} "
        + f"| {_fmt(_get(cross, 'condition', 'within_1_accuracy'))} "
        + f"| {_fmt(_get(cross, 'calibration', 'ece'))} |",
        "",
        "### Gold-set Cohen's kappa (garment-level 95% bootstrap CIs)",
        "",
        f"- Expert versus expert, usage: {_fmt_ci(boot.get('gold_expert_vs_expert_usage'))}",
        f"- Expert versus expert, condition (quadratic): {_fmt_ci(boot.get('gold_expert_vs_expert_condition_quadratic'))}",
        f"- Model versus Wargön (Wargön photo), usage: {_fmt_ci(boot.get('model_vs_wargon_usage'))}",
        f"- Model versus Myrorna (same Wargön photo), usage: {_fmt_ci(boot.get('model_vs_myrorna_usage'))}",
        "",
        gold.get("wargon_photo_note")
        or "Myrorna's gold photographs are missing from the archive, so its annotations are scored using the Wargön photograph of the same garment.",
        "",
        "## Intended use",
        "",
        "Research / portfolio demo for second-hand sorting decision support. "
        "A person remains responsible for the garment. Not for unattended sorting, "
        "EPR evidence, fibre ID from photos, or brand authentication.",
        "",
        "## Limitations",
        "",
        "- Router scores in `configs/router.yaml` are illustrative placeholders, not LCA or partner capacity.",
        "- Repair / Remake / Energy recovery collapse to the training class Other.",
        "- Expert disagreement is large; read model kappa next to expert-versus-expert kappa.",
        "- Station 3 checks a second Swedish site, not another country.",
        "- Absolute macro-F1 is modest; the claim is careful evaluation under label noise, not SOTA accuracy.",
        "- CPU demo loads float16 weights and runs float32 inference.",
        "",
        "## Files",
        "",
        "- `model.fp16.pt` — state dict (float16), config, temperature, abstain threshold",
        "- `router.yaml` — illustrative routing weights",
        "",
        "## Citation",
        "",
        "Dataset: Farrukh Nauman, RISE, Wargön Innovation AB, Myrorna AB, Vinnova / EU CISUTAC. CC-BY 4.0.",
        "",
    ]
    out = ROOT / "demo" / "hf_model" / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
