"""Parse local JSON annotations into one parquet row per garment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from garmentroute.eda import write_eda
from garmentroute.index import assign_splits, choose_subset, iter_rows, mark_near_duplicates, subset_members
from garmentroute.util import load_yaml, write_json


def _archive_names(manifest_path: Path) -> set[str] | None:
    if not manifest_path.exists():
        return None
    members = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {member["name"] for member in members}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/zip_manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("data/index.parquet"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--summary", type=Path, default=Path("reports/eda_summary.json"))
    args = parser.parse_args(argv)
    config = load_yaml(args.config) if args.config.exists() else {}
    seed = int(config.get("seed", 42))
    rows, skipped = iter_rows(args.raw, _archive_names(args.manifest))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit(f"no annotation JSON files under {args.raw}")
    frame["duplicate_of"] = pd.NA
    frame["phash"] = pd.NA
    frame = mark_near_duplicates(frame, args.image_root, threshold=int(config.get("phash_threshold", 4)))
    frame, meta = assign_splits(
        frame,
        seed=seed,
        val_fraction=float(config.get("val_fraction", 0.15)),
        test_fraction=float(config.get("test_fraction", 0.15)),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    write_json(args.out.parent / "split_meta.json", meta)
    skipped_path = args.out.parent / "skipped.jsonl"
    with skipped_path.open("w", encoding="utf-8") as handle:
        for row in skipped:
            handle.write(json.dumps(row) + "\n")
    split_dir = args.out.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split, group in frame.groupby("split"):
        (split_dir / f"{split}.txt").write_text("\n".join(group["item_id"].astype(str)) + "\n", encoding="utf-8")
    subset = choose_subset(
        frame,
        target=int(config.get("subset_size", 6000)),
        seed=seed,
        station3_n=int(config.get("station3_subset_size", 1500)),
    )
    members = subset_members(subset)
    (args.out.parent / "subset_ids.txt").write_text("\n".join(subset["item_id"].astype(str)) + "\n", encoding="utf-8")
    (args.out.parent / "subset_members.txt").write_text("\n".join(members) + "\n", encoding="utf-8")
    write_eda(args.out, args.figures, args.summary, args.image_root)
    print(
        f"rows {len(frame)} skipped {len(skipped)} subset {len(subset)} "
        f"duplicates {int(frame['duplicate_of'].notna().sum())} -> {args.out}"
    )


if __name__ == "__main__":
    main()
