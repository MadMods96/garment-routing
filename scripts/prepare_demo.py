"""Bootstrap CIs for the front model, export the fp16 demo pack, sync the README."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.evaluate import predict_checkpoint
from garmentroute.infer import export_demo_pack
from garmentroute.labels import USAGE_CLASSES
from garmentroute.metrics import (
    WARGON_PHOTO_NOTE,
    bootstrap_macro_f1,
    bootstrap_model_vs_site_kappas,
    bootstrap_pairwise_kappa,
    model_vs_site_from_wargon_photos,
)
from garmentroute.reporting import sync
from garmentroute.uncertainty import apply_temperature
from garmentroute.util import write_json
from garmentroute.views import filter_to_subset, find_subset_members


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _usage_rows(frame: pd.DataFrame, predictions: list[dict], temperature: float) -> tuple[list[str], list[str]]:
    by_id = {row["item_id"]: row for row in predictions}
    truth: list[str] = []
    pred: list[str] = []
    for record in frame.to_dict("records"):
        label = record.get("usage_class")
        row = by_id.get(str(record["item_id"]))
        if not isinstance(label, str) or row is None:
            continue
        scaled = apply_temperature(np.asarray(row["usage_logits"], dtype=float)[None, :], temperature)[0]
        predicted = USAGE_CLASSES[int(np.argmax(_softmax(scaled)))]
        truth.append(label)
        pred.append(predicted)
    return truth, pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "convnext_tiny_front" / "best.pt")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "resized")
    parser.add_argument("--out-pack", type=Path, default=ROOT / "demo" / "model.fp16.pt")
    parser.add_argument("--examples-dir", type=Path, default=ROOT / "demo" / "space" / "examples")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    front = results.get("front") or {}
    temperature = float(front.get("temperature") or 1.0)
    threshold = (front.get("abstain_threshold") or {}).get("threshold")

    frame = pd.read_parquet(args.index)
    members = find_subset_members(ROOT / "data", ROOT)
    if members is not None:
        frame = filter_to_subset(frame, members)

    print("predict test", flush=True)
    test = frame[frame["split"] == "test"]
    test_pred = predict_checkpoint(args.checkpoint, test, args.image_root, batch_size=32, mc_passes=0) or []
    truth, pred = _usage_rows(test, test_pred, temperature)
    test_ci = bootstrap_macro_f1(truth, pred, USAGE_CLASSES, n_boot=args.n_boot, seed=args.seed)
    print("test usage macro-F1 CI", test_ci, flush=True)

    print("predict gold", flush=True)
    gold = frame[frame["split"] == "gold"]
    gold_pred = predict_checkpoint(args.checkpoint, gold, args.image_root, batch_size=32, mc_passes=0) or []
    prediction_by_item = {}
    for row in gold_pred:
        scaled = apply_temperature(np.asarray(row["usage_logits"], dtype=float)[None, :], temperature)[0]
        prediction_by_item[str(row["item_id"])] = {
            "usage": USAGE_CLASSES[int(np.argmax(_softmax(scaled)))],
            "condition": row["condition"],
        }
    site = model_vs_site_from_wargon_photos(gold, prediction_by_item)
    site_ci = bootstrap_model_vs_site_kappas(
        gold, prediction_by_item, n_boot=args.n_boot, seed=args.seed
    )
    expert_usage_ci = bootstrap_pairwise_kappa(gold, "usage", n_boot=args.n_boot, seed=args.seed)
    expert_condition_ci = bootstrap_pairwise_kappa(
        gold, "condition", n_boot=args.n_boot, seed=args.seed, quadratic=True
    )

    results["gold"]["model_vs_wargon"] = site["model_vs_wargon"]
    results["gold"]["model_vs_myrorna"] = site["model_vs_myrorna"]
    results["gold"]["n_garments_with_wargon_photos"] = site["n_garments_with_wargon_photos"]
    results["gold"]["wargon_photo_note"] = WARGON_PHOTO_NOTE
    results["bootstrap"] = {
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "definition": (
            "Garment-level bootstrap percentile intervals. Test usage macro-F1 resamples "
            "test garments. Gold kappas resample gold_id clusters."
        ),
        "test_usage_macro_f1": test_ci,
        "gold_expert_vs_expert_usage": {
            "point": expert_usage_ci["point"],
            "n_garments": expert_usage_ci["n_garments"],
            "n_boot": expert_usage_ci["n_boot"],
            "mean": expert_usage_ci["mean"],
            "ci_low": expert_usage_ci["ci_low"],
            "ci_high": expert_usage_ci["ci_high"],
            "alpha": expert_usage_ci["alpha"],
        },
        "gold_expert_vs_expert_condition_quadratic": {
            "point": expert_condition_ci["point"],
            "n_garments": expert_condition_ci["n_garments"],
            "n_boot": expert_condition_ci["n_boot"],
            "mean": expert_condition_ci["mean"],
            "ci_low": expert_condition_ci["ci_low"],
            "ci_high": expert_condition_ci["ci_high"],
            "alpha": expert_condition_ci["alpha"],
        },
        "model_vs_wargon_usage": site_ci["model_vs_wargon_usage"],
        "model_vs_wargon_condition_quadratic": site_ci["model_vs_wargon_condition_quadratic"],
        "model_vs_myrorna_usage": site_ci["model_vs_myrorna_usage"],
        "model_vs_myrorna_condition_quadratic": site_ci["model_vs_myrorna_condition_quadratic"],
    }
    write_json(args.results, results)
    sync(ROOT / "README.md", args.results, ROOT / "reports" / "eda_summary.json")

    pack = export_demo_pack(args.checkpoint, args.out_pack, temperature, threshold)
    print("demo pack", pack, "bytes", pack.stat().st_size, flush=True)

    # Six test examples with both photographs.
    args.examples_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    scored_ids = {row["item_id"] for row in test_pred}
    candidates = test[test["item_id"].isin(scored_ids)].copy()
    for record in candidates.sample(n=min(6, len(candidates)), random_state=args.seed).to_dict("records"):
        front_rel = str(record["front_path"]).replace("\\", "/")
        back_rel = str(record["back_path"]).replace("\\", "/")
        front_src = args.image_root / front_rel
        back_src = args.image_root / back_rel
        if not front_src.is_file() or not back_src.is_file():
            continue
        stem = str(record["item_id"]).replace(":", "_").replace("/", "_")
        front_dst = args.examples_dir / f"{stem}_front.jpg"
        back_dst = args.examples_dir / f"{stem}_back.jpg"
        shutil.copy2(front_src, front_dst)
        shutil.copy2(back_src, back_dst)
        examples.append(
            {
                "item_id": record["item_id"],
                "front": front_dst.name,
                "back": back_dst.name,
                "usage_class": record.get("usage_class"),
                "condition": record.get("condition"),
                "attribution": (
                    "Photograph from the Clothing Dataset for Second-Hand Fashion "
                    "(Zenodo 10.5281/zenodo.13788681), CC-BY 4.0. Wargön Innovation / Myrorna / RISE."
                ),
            }
        )
        if len(examples) >= 6:
            break
    write_json(args.examples_dir / "manifest.json", {"examples": examples})
    print("examples", len(examples), flush=True)


if __name__ == "__main__":
    main()
