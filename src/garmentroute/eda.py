"""Label-only figures and a machine-readable EDA summary.

Sample photographs are written only when the image files are already on disk.
Class counts in the summary are computed from the index, not typed in.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from garmentroute.labels import loose_gold_number
from garmentroute.util import is_missing, write_json

COUNT_COLUMNS = ("usage", "condition", "pilling", "stains_3class", "holes", "smell", "material", "type")


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Correlation ratio of a numeric column grouped by a categorical column."""
    paired = pd.DataFrame({"category": categories, "value": values}).dropna()
    if paired.empty:
        return None
    numbers = paired["value"].to_numpy(dtype=float)
    total = float(np.sum((numbers - numbers.mean()) ** 2))
    if total == 0:
        return None
    between = 0.0
    for _name, group in paired.groupby("category"):
        between += len(group) * float(group["value"].mean() - numbers.mean()) ** 2
    return float(np.sqrt(between / total))


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_figures(frame: pd.DataFrame, figure_dir: Path, image_root: Path | None = None) -> dict[str, str]:
    """Write the EDA figures and return their paths."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    usage_counts = frame["usage"].fillna("missing").value_counts()
    figure, axis = plt.subplots(figsize=(8, 4))
    usage_counts.plot(kind="bar", ax=axis, color="#2f6f8f")
    axis.set_title("Expert usage labels")
    axis.set_xlabel("")
    axis.set_ylabel("Garments")
    path = figure_dir / "usage_counts.png"
    _save(figure, path)
    written["usage_counts"] = str(path)

    stations = [station for station in ("station1", "station2", "station3") if station in set(frame["station"])]
    usage_order = [label for label in usage_counts.index if label != "missing"] + (
        ["missing"] if "missing" in usage_counts.index else []
    )
    pivot = (
        frame.assign(usage=frame["usage"].fillna("missing"))
        .pivot_table(index="station", columns="usage", values="item_id", aggfunc="count", fill_value=0)
        .reindex(index=stations, columns=usage_order, fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(9, 4))
    pivot.plot(kind="bar", ax=axis)
    axis.set_title("Usage by sorting station")
    axis.set_xlabel("")
    axis.set_ylabel("Garments")
    axis.legend(loc="upper right", fontsize=8)
    path = figure_dir / "usage_by_station.png"
    _save(figure, path)
    written["usage_by_station"] = str(path)

    missing = pd.DataFrame({column: frame[column].isna() for column in COUNT_COLUMNS if column in frame.columns})
    rates = missing.groupby(frame["station"]).mean().reindex(stations)
    figure, axis = plt.subplots(figsize=(9, 3.5))
    image = axis.imshow(rates.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(rates.shape[1]), rates.columns, rotation=45, ha="right")
    axis.set_yticks(range(rates.shape[0]), rates.index)
    axis.set_title("Share of missing labels by station")
    figure.colorbar(image, ax=axis, fraction=0.046)
    path = figure_dir / "missingness.png"
    _save(figure, path)
    written["missingness"] = str(path)

    heat = (
        frame.dropna(subset=["condition", "usage"])
        .pivot_table(index="usage", columns="condition", values="item_id", aggfunc="count", fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="Blues")
    axis.set_xticks(range(5), [1, 2, 3, 4, 5])
    axis.set_yticks(range(len(heat.index)), heat.index)
    axis.set_xlabel("Condition (5 is best)")
    axis.set_title("Condition against expert usage")
    figure.colorbar(image, ax=axis, fraction=0.046)
    path = figure_dir / "condition_by_usage.png"
    _save(figure, path)
    written["condition_by_usage"] = str(path)

    if image_root is not None:
        example_path = _usage_examples(frame, image_root, figure_dir / "usage_examples.png")
        if example_path is not None:
            written["usage_examples"] = str(example_path)
    return written


def _usage_examples(frame: pd.DataFrame, image_root: Path, destination: Path) -> str | None:
    classes = [label for label in ("Reuse", "Export", "Recycle", "Repair", "Remake", "Energy recovery")]
    chosen = []
    for label in classes:
        subset = frame[(frame["usage"] == label) & frame["front_path"].notna()]
        for _index, row in subset.iterrows():
            path = image_root / row["front_path"]
            if path.is_file():
                chosen.append((label, path))
                break
    if not chosen:
        return None
    figure, axes = plt.subplots(1, len(chosen), figsize=(2.4 * len(chosen), 3))
    if len(chosen) == 1:
        axes = [axes]
    for axis, (label, path) in zip(axes, chosen):
        axis.imshow(plt.imread(path))
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    _save(figure, destination)
    return str(destination)


def _blank(value: Any) -> bool:
    if is_missing(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing)


def _gold_item_key(value: Any) -> str | None:
    if _blank(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        return None
    return str(int(text))


def gold_census(frame: pd.DataFrame) -> dict[str, Any]:
    """Count gold-split files, garments, and the records that enter agreement.

    One index row is one annotation record, and in this dataset that record is
    one JSON file. A garment is one item number parsed from a comment that
    contains ``Test`` plus digits. The parser is case-sensitive, so a lowercase
    ``test 12`` stays in the split when the file is under ``test100`` but does
    not receive a garment number.
    """
    empty: dict[str, Any] = {
        "n_split_rows": 0,
        "n_json_files": 0,
        "n_garments": 0,
        "n_annotations_with_item_number": 0,
        "mean_annotations_per_garment": None,
        "n_lowercase_annotations_of_same_garments": 0,
        "n_other_numbered_comments": 0,
        "n_unnumbered_comments": 0,
        "unnumbered_comment_text": [],
    }
    if "is_gold" not in frame.columns or frame.empty:
        return empty
    gold = frame[frame["is_gold"].astype(bool)].copy()
    empty["n_split_rows"] = int(len(gold))
    if "json_path" in gold.columns:
        empty["n_json_files"] = int(gold["json_path"].nunique())
    else:
        empty["n_json_files"] = int(len(gold))
    if gold.empty or "gold_id" not in gold.columns:
        return empty
    keys = gold["gold_id"].map(_gold_item_key)
    numbered = keys.notna()
    known = set(keys[numbered].tolist())
    empty["n_annotations_with_item_number"] = int(numbered.sum())
    empty["n_garments"] = int(len(known))
    if known:
        empty["mean_annotations_per_garment"] = float(numbered.sum() / len(known))
    lowercase_same = 0
    other_numbered = 0
    unnumbered_text: list[str] = []
    comments = gold["comment"] if "comment" in gold.columns else pd.Series([None] * len(gold), index=gold.index)
    for index in gold.index[~numbered]:
        comment = comments.loc[index]
        number = loose_gold_number(comment)
        if number is None:
            text = "" if _blank(comment) else str(comment).strip()
            unnumbered_text.append(text)
        elif number in known:
            lowercase_same += 1
        else:
            other_numbered += 1
    empty["n_lowercase_annotations_of_same_garments"] = lowercase_same
    empty["n_other_numbered_comments"] = other_numbered
    empty["n_unnumbered_comments"] = len(unnumbered_text)
    empty["unnumbered_comment_text"] = sorted(set(unnumbered_text))
    accounted = (
        empty["n_annotations_with_item_number"]
        + empty["n_lowercase_annotations_of_same_garments"]
        + empty["n_other_numbered_comments"]
        + empty["n_unnumbered_comments"]
    )
    if accounted != empty["n_split_rows"]:
        raise ValueError(f"gold census does not partition the split: {accounted} != {empty['n_split_rows']}")
    return empty


def summarise(frame: pd.DataFrame) -> dict:
    """Counts and the condition/usage association. All values are computed here."""
    def counts(series: pd.Series) -> dict[str, int]:
        filled = series.fillna("missing")
        return {str(key): int(value) for key, value in filled.value_counts().items()}

    summary = {
        "n_rows": int(len(frame)),
        "n_per_station": {str(key): int(value) for key, value in frame["station"].value_counts().items()},
        "n_gold": int(frame["is_gold"].sum()),
        "n_gold_with_id": int(frame["gold_id"].fillna("").astype(str).str.len().gt(0).sum()),
        "usage_raw_counts": counts(frame["usage"]),
        "usage_class_counts": counts(frame["usage_class"]),
        "split_counts": {str(key): int(value) for key, value in frame["split"].value_counts().items()}
        if "split" in frame.columns
        else {},
        "missing_rate": {
            column: float(frame[column].isna().mean()) for column in COUNT_COLUMNS if column in frame.columns
        },
        "stains_legacy_n": int(frame["stains_legacy"].sum()) if "stains_legacy" in frame.columns else 0,
        "mean_condition_by_usage": {
            str(key): float(value)
            for key, value in frame.dropna(subset=["condition", "usage"]).groupby("usage")["condition"].mean().items()
        },
        "condition_usage_correlation_ratio": correlation_ratio(frame["usage"], frame["condition"]),
        "duplicate_n": int(frame["duplicate_of"].notna().sum()) if "duplicate_of" in frame.columns else 0,
        "front_in_archive_n": None
        if "front_in_archive" not in frame.columns or bool(frame["front_in_archive"].isna().all())
        else int(frame["front_in_archive"].fillna(False).sum()),
        "back_in_archive_n": None
        if "back_in_archive" not in frame.columns or bool(frame["back_in_archive"].isna().all())
        else int(frame["back_in_archive"].fillna(False).sum()),
        "gold_census": gold_census(frame),
    }
    if "gold_id" in frame.columns:
        gold = frame[frame["is_gold"].astype(bool) & frame["gold_id"].notna()]
        per_item = gold.groupby("gold_id").size() if len(gold) else pd.Series(dtype=int)
        summary["gold_annotations_per_item"] = {
            "n_items": int(per_item.shape[0]),
            "mean": float(per_item.mean()) if len(per_item) else None,
            "min": int(per_item.min()) if len(per_item) else None,
            "max": int(per_item.max()) if len(per_item) else None,
        }
    return summary


def write_metric_figures(results: dict, figure_dir: Path) -> dict[str, str]:
    """Plot coverage and reliability when a checkpoint section contains them."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for model_name in ("fusion", "front"):
        block = results.get(model_name)
        if not isinstance(block, dict):
            continue
        for split in ("test", "cross_site"):
            after = ((block.get("splits") or {}).get(split) or {}).get("after_temperature") or {}
            curve = after.get("coverage_accuracy") or []
            if curve:
                figure, axis = plt.subplots(figsize=(6, 4))
                axis.plot([row["coverage"] for row in curve], [row["accuracy"] for row in curve], marker="o")
                axis.set_xlabel("Coverage")
                axis.set_ylabel("Accuracy on the covered garments")
                axis.set_title(f"{model_name} {split}: coverage against accuracy")
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                path = figure_dir / f"coverage_{model_name}_{split}.png"
                _save(figure, path)
                written[f"coverage_{model_name}_{split}"] = str(path)
            bins = (after.get("calibration") or {}).get("bins") or []
            points = [row for row in bins if row.get("n")]
            if points:
                figure, axis = plt.subplots(figsize=(5, 5))
                axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
                axis.plot([row["confidence"] for row in points], [row["accuracy"] for row in points], marker="o")
                axis.set_xlabel("Confidence")
                axis.set_ylabel("Accuracy")
                axis.set_title(f"{model_name} {split}: reliability after temperature scaling")
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                path = figure_dir / f"reliability_{model_name}_{split}.png"
                _save(figure, path)
                written[f"reliability_{model_name}_{split}"] = str(path)
    return written


def write_eda(index_path: Path, figure_dir: Path, summary_path: Path, image_root: Path | None = None) -> dict:
    """Read the index, write figures, and write the summary JSON."""
    frame = pd.read_parquet(index_path)
    figures = write_figures(frame, figure_dir, image_root)
    summary = summarise(frame)
    summary["figures"] = figures
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index.parquet"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--summary", type=Path, default=Path("reports/eda_summary.json"))
    parser.add_argument("--image-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)
    summary = write_eda(args.index, args.figures, args.summary, args.image_root)
    print(f"rows {summary['n_rows']} gold {summary['n_gold']}")


if __name__ == "__main__":
    main()
