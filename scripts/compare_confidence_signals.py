"""Compare max-prob / margin / entropy confidence on val AURC; report test coverage curve."""

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

from garmentroute.evaluate import predict_checkpoint
from garmentroute.labels import USAGE_CLASSES
from garmentroute.uncertainty import (
    apply_temperature,
    confidence_from_probs,
    selective_coverage_report,
)
from garmentroute.util import write_json
from garmentroute.views import filter_to_subset, find_subset_members

SIGNALS = ("max_prob", "margin", "entropy")
FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=-1, keepdims=True)


def _rows(frame: pd.DataFrame, predictions: list[dict], temperature: float):
    by_id = {row["item_id"]: row for row in predictions}
    truth, pred, probs = [], [], []
    for record in frame.to_dict("records"):
        row = by_id.get(record["item_id"])
        label = record.get("usage_class")
        if row is None or not isinstance(label, str):
            continue
        scaled = apply_temperature(np.asarray(row["usage_logits"], dtype=float)[None, :], temperature)[0]
        probability = _softmax(scaled[None, :])[0]
        truth.append(label)
        pred.append(USAGE_CLASSES[int(np.argmax(probability))])
        probs.append(probability)
    return truth, pred, np.asarray(probs, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "convnext_tiny_front" / "best.pt")
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "resized")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.is_file() else {}
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    temperature = float((results.get("front") or {}).get("temperature") or ckpt.get("temperature") or 1.0)

    frame = pd.read_parquet(args.index)
    members = find_subset_members(ROOT / "data", ROOT)
    if members is not None:
        frame = filter_to_subset(frame, members)

    reports = {}
    for split in ("val", "test"):
        split_frame = frame[frame["split"] == split]
        print(f"predict {split} n={len(split_frame)}", flush=True)
        predictions = predict_checkpoint(
            args.checkpoint, split_frame, args.image_root, args.batch_size, mc_passes=0
        ) or []
        truth, pred, probs = _rows(split_frame, predictions, temperature)
        split_report = {"n": len(truth), "signals": {}}
        for signal in SIGNALS:
            confidence = confidence_from_probs(probs, signal)
            report = selective_coverage_report(
                confidence, truth, pred, USAGE_CLASSES, fractions=FRACTIONS
            )
            split_report["signals"][signal] = report
            print(
                f"{split} {signal}: AURC={report['aurc']} "
                + " ".join(
                    f"cov{point['requested_coverage']:.2f} "
                    f"acc={point['accuracy']:.3f} f1={point['macro_f1']:.3f}"
                    for point in report["points"]
                ),
                flush=True,
            )
            targets = report["coverage_at_accuracy"]
            print(
                f"  coverage@acc60={targets.get('0.60')} coverage@acc70={targets.get('0.70')}",
                flush=True,
            )
        reports[split] = split_report

    val_aurc = {
        signal: reports["val"]["signals"][signal]["aurc"]
        for signal in SIGNALS
        if reports["val"]["signals"][signal]["aurc"] is not None
    }
    selected = min(val_aurc, key=val_aurc.get) if val_aurc else "max_prob"
    print(f"selected_on_val_aurc={selected} aurc={val_aurc.get(selected)}", flush=True)

    block = {
        "criterion": "lowest_aurc_on_val",
        "selected_signal": selected,
        "temperature": temperature,
        "fractions": list(FRACTIONS),
        "val": reports["val"],
        "test": {
            "n": reports["test"]["n"],
            "selected_signal": selected,
            "report": reports["test"]["signals"][selected],
            "all_signals": reports["test"]["signals"],
        },
    }
    front = results.setdefault("front", {})
    front["confidence_signal_comparison"] = block
    # Convenience copy next to the router policy block.
    test_split = front.setdefault("splits", {}).setdefault("test", {})
    test_split["selective_coverage"] = {
        "signal": selected,
        "report": block["test"]["report"],
        "comparison_aurc_val": val_aurc,
    }
    write_json(args.results, results)
    print(f"wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
