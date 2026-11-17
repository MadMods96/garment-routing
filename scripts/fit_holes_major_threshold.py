"""Fit holes Major gate on val by F2 (fallback: recall with precision >= 0.3)."""

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

from garmentroute.evaluate import predict_checkpoint, score_predictions
from garmentroute.labels import TERNARY
from garmentroute.metrics import classification_metrics
from garmentroute.model import decode_ternary
from garmentroute.util import write_json
from garmentroute.views import filter_to_subset, find_subset_members


def _f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision + recall <= 0:
        return 0.0
    beta2 = beta * beta
    return float((1 + beta2) * precision * recall / (beta2 * precision + recall))


def _major_prf(truth: list[str], pred: list[str]) -> tuple[float, float, float]:
    report = classification_metrics(truth, pred, list(TERNARY))
    cm = report["confusion_matrix"]["matrix"]
    labels = report["confusion_matrix"]["labels"]
    major_i = labels.index("Major")
    tp = cm[major_i][major_i]
    support = sum(cm[major_i])
    predicted = sum(row[major_i] for row in cm)
    recall = float(tp / support) if support else 0.0
    precision = float(tp / predicted) if predicted else 0.0
    return precision, recall, _f_beta(precision, recall, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "convnext_tiny_front" / "best.pt")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "resized")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-precision", type=float, default=0.3)
    args = parser.parse_args()

    frame = pd.read_parquet(args.index)
    members = find_subset_members(ROOT / "data", ROOT)
    if members is not None:
        frame = filter_to_subset(frame, members)
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    print("predict val/test", flush=True)
    val_pred = predict_checkpoint(args.checkpoint, val, args.image_root, args.batch_size, mc_passes=0) or []
    test_pred = predict_checkpoint(args.checkpoint, test, args.image_root, args.batch_size, mc_passes=0) or []

    def pairs(split_frame, predictions):
        by_id = {row["item_id"]: row for row in predictions}
        truth, logits = [], []
        for record in split_frame.to_dict("records"):
            row = by_id.get(record["item_id"])
            label = record.get("holes")
            if row is None or not isinstance(label, str) or "holes_logits" not in row:
                continue
            truth.append(label)
            logits.append(row["holes_logits"])
        return truth, torch.tensor(logits, dtype=torch.float32)

    val_truth, val_logits = pairs(val, val_pred)
    test_truth, test_logits = pairs(test, test_pred)
    baseline_pred = [TERNARY[int(i)] for i in decode_ternary(val_logits).tolist()]
    base_p, base_r, base_f2 = _major_prf(val_truth, baseline_pred)
    print(f"val argmax Major P/R/F2={base_p:.3f}/{base_r:.3f}/{base_f2:.3f}", flush=True)

    best_t, best_f2, best_p, best_r = None, -1.0, 0.0, 0.0
    constrained = []
    for t in np.linspace(0.05, 0.95, 37):
        pred = [TERNARY[int(i)] for i in decode_ternary(val_logits, major_threshold=float(t)).tolist()]
        precision, recall, f2 = _major_prf(val_truth, pred)
        if f2 > best_f2:
            best_t, best_f2, best_p, best_r = float(t), f2, precision, recall
        if precision >= args.min_precision:
            constrained.append((f2, recall, precision, float(t)))
    if constrained:
        constrained.sort(key=lambda row: (-row[0], -row[1], -row[2]))
        best_f2, best_r, best_p, best_t = constrained[0]
        criterion = f"max_f2_subject_to_precision_ge_{args.min_precision}"
    else:
        criterion = "max_f2_no_precision_feasible"
    print(
        f"selected t={best_t:.3f} criterion={criterion} "
        f"val Major P/R/F2={best_p:.3f}/{best_r:.3f}/{best_f2:.3f}",
        flush=True,
    )

    def scored(truth, logits, threshold):
        pred = [TERNARY[int(i)] for i in decode_ternary(logits, major_threshold=threshold).tolist()]
        return classification_metrics(truth, pred, list(TERNARY))

    before_val = scored(val_truth, val_logits, None)
    after_val = scored(val_truth, val_logits, best_t)
    before_test = scored(test_truth, test_logits, None)
    after_test = scored(test_truth, test_logits, best_t)
    for name, report in (
        ("before_val", before_val),
        ("after_val", after_val),
        ("before_test", before_test),
        ("after_test", after_test),
    ):
        print(
            name,
            "Major",
            report["per_class_recall"].get("Major"),
            "n",
            report["per_class_support"].get("Major"),
            flush=True,
        )

    results = json.loads(args.results.read_text(encoding="utf-8"))
    front = results.setdefault("front", {})
    front["holes_major_threshold"] = {
        "threshold": best_t,
        "criterion": criterion,
        "val_major_precision": best_p,
        "val_major_recall": best_r,
        "val_major_f2": best_f2,
        "before": {"val": before_val, "test": before_test},
        "after": {"val": after_val, "test": after_test},
    }
    write_json(args.results, results)
    print(f"wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
