"""Print holes/stains per-class recall with support for val/test; merge into results.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.util import write_json


def _from_cm(report: dict | None) -> dict | None:
    if not report:
        return None
    recall = report.get("per_class_recall")
    support = report.get("per_class_support")
    if recall is None:
        cm = (report.get("confusion_matrix") or {})
        labels = cm.get("labels") or []
        matrix = cm.get("matrix") or []
        recall = {}
        support = {}
        for index, label in enumerate(labels):
            row_sum = int(sum(matrix[index])) if index < len(matrix) else 0
            support[str(label)] = row_sum
            diag = int(matrix[index][index]) if row_sum and index < len(matrix[index]) else 0
            recall[str(label)] = None if row_sum == 0 else float(diag / row_sum)
    return {
        "n": report.get("n"),
        "balanced_accuracy": report.get("balanced_accuracy"),
        "macro_f1": report.get("macro_f1"),
        "per_class_recall": recall,
        "per_class_support": support,
    }


def _print_block(title: str, block: dict | None) -> None:
    print(title, flush=True)
    if not block:
        print("  (missing)", flush=True)
        return
    recall = block.get("per_class_recall") or {}
    support = block.get("per_class_support") or {}
    parts = []
    for label, value in recall.items():
        n = support.get(label)
        if value is None:
            parts.append(f"{label}=n/a (n={n})")
        else:
            parts.append(f"{label}={value:.3f} (n={n})")
    print(
        f"  n={block.get('n')} bal_acc={block.get('balanced_accuracy')} "
        f"macro_f1={block.get('macro_f1')} | " + " ".join(parts),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--tag", default="before")
    args = parser.parse_args()
    results = json.loads(args.results.read_text(encoding="utf-8"))
    front = results.get("front") or {}
    diagnostics = front.setdefault("diagnostics", {})
    snapshot = {"tag": args.tag, "holes": {}, "stains": {}}
    val_holes = _from_cm(front.get("val_holes"))
    val_stains = _from_cm(front.get("val_stains"))
    snapshot["holes"]["val"] = val_holes
    snapshot["stains"]["val"] = val_stains
    _print_block(f"{args.tag} val holes", val_holes)
    _print_block(f"{args.tag} val stains", val_stains)
    for split in ("test", "cross_site", "gold"):
        after = ((front.get("splits") or {}).get(split) or {}).get("after_temperature") or {}
        holes = _from_cm(after.get("holes"))
        stains = _from_cm(after.get("stains"))
        snapshot["holes"][split] = holes
        snapshot["stains"][split] = stains
        _print_block(f"{args.tag} {split} holes", holes)
        _print_block(f"{args.tag} {split} stains", stains)
    diagnostics[args.tag] = snapshot
    write_json(args.results, results)
    print(f"wrote {args.results} front.diagnostics.{args.tag}", flush=True)


if __name__ == "__main__":
    main()
