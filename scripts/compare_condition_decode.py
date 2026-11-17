"""Compare CORAL decode methods on val; pick by quadratic weighted kappa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.evaluate import predict_checkpoint
from garmentroute.model import CORAL_DECODE_METHODS, coral_decode
from garmentroute.metrics import ordinal_metrics
from garmentroute.util import write_json
from garmentroute.views import filter_to_subset, find_subset_members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "convnext_tiny_front" / "best.pt")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "resized")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    frame = pd.read_parquet(args.index)
    members = find_subset_members(ROOT / "data", ROOT)
    if members is not None:
        frame = filter_to_subset(frame, members)
    val = frame[frame["split"] == "val"]
    print("predict val", len(val), flush=True)
    predictions = predict_checkpoint(args.checkpoint, val, args.image_root, args.batch_size, mc_passes=0) or []
    by_id = {row["item_id"]: row for row in predictions}
    truth = []
    logits = []
    for record in val.to_dict("records"):
        row = by_id.get(record["item_id"])
        label = record.get("condition")
        if row is None or label is None or label != label:
            continue
        if "condition_logits" not in row:
            raise SystemExit("predictions lack condition_logits; re-run with updated evaluate.py")
        truth.append(int(label))
        logits.append(row["condition_logits"])
    logit_tensor = torch.tensor(logits, dtype=torch.float32)
    truth_arr = np.asarray(truth, dtype=int)
    comparison = {}
    best_name = None
    best_kappa = float("-inf")
    for name in CORAL_DECODE_METHODS:
        pred = coral_decode(logit_tensor, method=name).cpu().numpy().astype(int)
        kappa = float(cohen_kappa_score(truth_arr, pred, weights="quadratic"))
        mae = ordinal_metrics(truth_arr.tolist(), pred.tolist())["mae"]
        counts = {str(level): int((pred == level).sum()) for level in range(1, 6)}
        comparison[name] = {
            "quadratic_kappa": kappa,
            "mae": mae,
            "n_pred_1": counts["1"],
            "n_pred_5": counts["5"],
            "predicted_counts": counts,
        }
        print(
            f"{name}: quadratic_kappa={kappa:.4f} mae={mae:.4f} "
            f"n_pred_1={counts['1']} n_pred_5={counts['5']}",
            flush=True,
        )
        if kappa > best_kappa:
            best_kappa = kappa
            best_name = name
    print(f"selected {best_name}", flush=True)
    results = json.loads(args.results.read_text(encoding="utf-8"))
    front = results.setdefault("front", {})
    front["condition_decode"] = {
        "selected": best_name,
        "criterion": "quadratic_weighted_kappa_on_val",
        "comparison": comparison,
    }
    write_json(args.results, results)
    # Point the runtime default at the winner without editing source by hand:
    model_path = ROOT / "src" / "garmentroute" / "model.py"
    text = model_path.read_text(encoding="utf-8")
    old = 'CORAL_DECODE_DEFAULT = "expected"'
    new = f'CORAL_DECODE_DEFAULT = "{best_name}"'
    if old in text and best_name != "expected":
        model_path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"updated CORAL_DECODE_DEFAULT -> {best_name}", flush=True)


if __name__ == "__main__":
    main()
