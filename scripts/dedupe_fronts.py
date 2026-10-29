"""Drop near-duplicate fronts and keep a photograph out of two splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from garmentroute.eda import write_eda
from garmentroute.index import cross_split_near_duplicates, mark_split_safe_duplicates
from garmentroute.util import load_yaml, write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index.parquet"))
    parser.add_argument("--image-root", type=Path, default=Path("data/resized"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--log", type=Path, default=Path("reports/duplicate_pairs.jsonl"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--summary", type=Path, default=Path("reports/eda_summary.json"))
    args = parser.parse_args(argv)
    config = load_yaml(args.config) if args.config.exists() else {}
    threshold = int(config.get("phash_threshold", 4))
    frame = pd.read_parquet(args.index)
    frame, removals = mark_split_safe_duplicates(frame, args.image_root, threshold=threshold)
    violations = cross_split_near_duplicates(frame, threshold=threshold)
    if violations:
        preview = json.dumps(violations[:5])
        raise SystemExit(f"{len(violations)} near-duplicate pairs still cross a split: {preview}")
    args.index.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.index, index=False)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("w", encoding="utf-8") as handle:
        for removal in removals:
            handle.write(json.dumps(removal) + "\n")
    split_dir = args.index.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, group in frame.groupby("split"):
        (split_dir / f"{split}.txt").write_text("\n".join(group["item_id"].astype(str)) + "\n", encoding="utf-8")
    summary = write_eda(args.index, args.figures, args.summary, args.image_root)
    summary["duplicate_train_test_dropped"] = sum(row["reason"] == "train_test" for row in removals)
    summary["duplicate_cross_split_dropped"] = sum(row["reason"] == "cross_split" for row in removals)
    summary["duplicate_within_split_dropped"] = sum(row["reason"] == "within_split" for row in removals)
    write_json(args.summary, summary)
    print(
        f"removed {len(removals)} "
        f"train_test {summary['duplicate_train_test_dropped']} "
        f"cross_split {summary['duplicate_cross_split_dropped']} "
        f"within_split {summary['duplicate_within_split_dropped']} "
        f"remaining_cross_split 0 -> {args.log}"
    )


if __name__ == "__main__":
    main()
