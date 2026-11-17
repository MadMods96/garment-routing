"""Evaluate 224 and 384 front checkpoints on the expanded 16200 subset.

Same split assignment as the index; more garments once images are on disk.
Writes holes Major recall with bootstrap CIs and router HUMAN_REVIEW share.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.evaluate import predict_checkpoint, router_from_predictions, score_predictions
from garmentroute.labels import TERNARY
from garmentroute.metrics import bootstrap_class_recall, classification_metrics
from garmentroute.model import decode_ternary
from garmentroute.util import load_yaml, write_json
from garmentroute.views import filter_to_subset, list_image_relatives


def _holes_labels(frame: pd.DataFrame, predictions: list[dict], major_threshold: float | None):
    by_id = {row["item_id"]: row for row in predictions}
    truth, pred = [], []
    for record in frame.to_dict("records"):
        row = by_id.get(record["item_id"])
        label = record.get("holes")
        if row is None or not isinstance(label, str):
            continue
        if "holes_logits" in row:
            logits = torch.tensor([row["holes_logits"]], dtype=torch.float32)
            index = int(decode_ternary(logits, major_threshold=major_threshold).item())
            predicted = TERNARY[index]
        else:
            predicted = row["holes"]
        truth.append(label)
        pred.append(predicted)
    return truth, pred


def _score_checkpoint(
    name: str,
    checkpoint: Path,
    frame: pd.DataFrame,
    image_root: Path,
    batch_size: int,
    router_config: dict,
    temperature: float | None,
    abstain_threshold: float | None,
    holes_major_threshold: float | None,
    n_boot: int,
    seed: int,
) -> dict:
    if not checkpoint.is_file():
        return {"status": "missing_checkpoint", "checkpoint": str(checkpoint)}
    available_splits = {}
    for split in ("val", "test"):
        split_frame = frame[frame["split"] == split]
        print(f"{name} predict {split} n_index={len(split_frame)}", flush=True)
        predictions = predict_checkpoint(checkpoint, split_frame, image_root, batch_size, mc_passes=0) or []
        print(f"{name} scored {split} n={len(predictions)}", flush=True)
        scored = score_predictions(split_frame, predictions, temperature=1.0)
        truth, pred = _holes_labels(split_frame, predictions, holes_major_threshold)
        holes_report = classification_metrics(truth, pred, list(TERNARY))
        major_ci = bootstrap_class_recall(truth, pred, "Major", n_boot=n_boot, seed=seed)
        block = {
            "n_scored": len(predictions),
            "holes": holes_report,
            "holes_major_recall_bootstrap": major_ci,
            "stains": scored.get("stains"),
            "condition": scored.get("condition"),
        }
        if split == "test":
            # Prefer checkpoint-stored calibration when present.
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            temp = float(temperature if temperature is not None else ckpt.get("temperature") or 1.0)
            thr = abstain_threshold
            if thr is None:
                thr = ckpt.get("abstain_threshold")
            if thr is not None:
                thr = float(thr)
            major_thr = holes_major_threshold
            if major_thr is None:
                major_thr = ckpt.get("holes_major_threshold")
            if major_thr is not None:
                major_thr = float(major_thr)
            # Re-decode holes with threshold for router inputs.
            for row in predictions:
                if "holes_logits" in row:
                    logits = torch.tensor([row["holes_logits"]], dtype=torch.float32)
                    row["holes"] = TERNARY[int(decode_ternary(logits, major_threshold=major_thr).item())]
            router = router_from_predictions(split_frame, predictions, router_config, temp, thr)
            block["router_on_predicted_attributes"] = {
                key: value
                for key, value in router.items()
                if key
                in {
                    "n",
                    "abstained",
                    "human_review_share",
                    "human_review_share_note",
                    "usage_class_agreement",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                }
            }
            share = router.get("human_review_share")
            print(
                f"{name} test HUMAN_REVIEW share={share} "
                f"abstained={router.get('abstained')}/{router.get('n')}",
                flush=True,
            )
            if share is not None and share > 0.5:
                print(
                    f"{name}: more than half of test garments go to HUMAN_REVIEW under the current router.",
                    flush=True,
                )
        available_splits[split] = block
        major = holes_report["per_class_recall"].get("Major")
        support = holes_report["per_class_support"].get("Major")
        print(
            f"{name} {split} holes Major recall={major} (n={support}) "
            f"CI=[{major_ci.get('ci_low')}, {major_ci.get('ci_high')}]",
            flush=True,
        )
    return {"checkpoint": str(checkpoint), "splits": available_splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "resized")
    parser.add_argument("--members", type=Path, default=ROOT / "data" / "subset_members_16200.txt")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--router", type=Path, default=ROOT / "configs" / "router.yaml")
    parser.add_argument("--ckpt-224", type=Path, default=ROOT / "runs" / "convnext_tiny_front" / "best.pt")
    parser.add_argument(
        "--ckpt-384",
        type=Path,
        default=ROOT / "runs" / "convnext_tiny_front_384_10k" / "best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holes-major-threshold", type=float, default=None)
    args = parser.parse_args()

    frame = pd.read_parquet(args.index)
    frame = filter_to_subset(frame, args.members)
    available = list_image_relatives(args.image_root)
    front = frame["front_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    back = frame["back_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    has_image = front.isin(available) | back.isin(available)
    frame = frame.loc[has_image].copy()
    print(
        "expanded with images on disk: "
        + " ".join(f"{split}={int((frame.split == split).sum())}" for split in ("train", "val", "test")),
        flush=True,
    )
    for split in ("val", "test"):
        major_n = int(frame.loc[frame.split.eq(split) & frame.holes.eq("Major")].shape[0])
        print(f"{split} holes Major support in index+images={major_n}", flush=True)

    router_config = load_yaml(args.router)
    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.is_file() else {}
    major_thr = args.holes_major_threshold
    if major_thr is None:
        stored = ((results.get("front") or {}).get("holes_major_threshold") or {}).get("threshold")
        if stored is not None:
            major_thr = float(stored)
    print(f"holes_major_threshold={major_thr}", flush=True)
    block = {
        "members": str(args.members),
        "n_garments_with_images": int(len(frame)),
        "holes_major_threshold": major_thr,
        "ckpt_384": str(args.ckpt_384),
        "models": {},
    }
    for name, ckpt in (("front_224", args.ckpt_224), ("front_384", args.ckpt_384)):
        block["models"][name] = _score_checkpoint(
            name,
            ckpt,
            frame,
            args.image_root,
            args.batch_size,
            router_config,
            temperature=None,
            abstain_threshold=None,
            holes_major_threshold=major_thr,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    results["expanded_dual_eval"] = block
    write_json(args.results, results)
    print(f"wrote {args.results} expanded_dual_eval", flush=True)


if __name__ == "__main__":
    main()
