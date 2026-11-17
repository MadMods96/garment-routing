"""Decide which photograph to load when a garment is missing a view."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def _normalise_relative(relative: object) -> str:
    if relative is None or (isinstance(relative, float) and pd.isna(relative)):
        return ""
    text = str(relative).strip().replace("\\", "/")
    if not text or text.lower() == "nan":
        return ""
    return text


def _is_file(image_root: Path, relative: object, available: set[str] | None = None) -> bool:
    text = _normalise_relative(relative)
    if not text:
        return False
    if available is not None:
        return text in available
    return (image_root / text).is_file()


def list_image_relatives(image_root: Path) -> set[str]:
    """Relative paths of files under ``image_root``, from one ``os.scandir`` walk."""
    root = Path(image_root)
    found: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        found.add(Path(entry.path).relative_to(root).as_posix())
        except FileNotFoundError:
            continue
    return found


def filter_to_subset(frame: pd.DataFrame, members_path: Path | None) -> pd.DataFrame:
    """Keep rows whose item id or image path is listed in ``members_path``."""
    if members_path is None or not Path(members_path).is_file():
        return frame
    names = {
        line.strip().replace("\\", "/")
        for line in Path(members_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not names:
        return frame
    front = frame["front_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    back = frame["back_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    keep = front.isin(names) | back.isin(names)
    if "item_id" in frame.columns:
        keep = keep | frame["item_id"].astype(str).isin(names)
    return frame.loc[keep].copy()


def find_subset_members(*roots: Path) -> Path | None:
    """Return an explicit members file, else the first known members name under ``roots``."""
    for root in roots:
        if root is None:
            continue
        path = Path(root)
        # Config may point at subset_members_16200.txt (or any path list).
        if path.is_file():
            return path
        for name in ("subset_members_16200.txt", "subset_members.txt", "subset_ids.txt"):
            for candidate in (path / name, path / "data" / name):
                if candidate.is_file():
                    return candidate
    return None


def resolve_views(
    frame: pd.DataFrame,
    image_root: Path,
    available: set[str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Keep garments with at least one image and record a missing front or back.

    ``view`` is ``both``, ``front``, or ``back``. A garment with neither file
    is omitted. The log lists only garments that are missing a view.
    """
    root = Path(image_root)
    if available is None:
        available = list_image_relatives(root)
    kept: list[dict] = []
    log: list[dict[str, str]] = []
    for record in frame.to_dict("records"):
        front_ok = _is_file(root, record.get("front_path"), available)
        back_ok = _is_file(root, record.get("back_path"), available)
        item_id = str(record.get("item_id", ""))
        if front_ok and back_ok:
            record["view"] = "both"
            kept.append(record)
        elif front_ok:
            record["view"] = "front"
            kept.append(record)
            log.append(
                {"item_id": item_id, "missing": "back", "fallback": "single_view", "used_view": "front"}
            )
        elif back_ok:
            record["view"] = "back"
            kept.append(record)
            log.append(
                {"item_id": item_id, "missing": "front", "fallback": "single_view", "used_view": "back"}
            )
        else:
            log.append({"item_id": item_id, "missing": "both", "fallback": "excluded", "used_view": ""})
    if "split" in frame.columns and log:
        by_item = frame.astype({"item_id": str}).drop_duplicates("item_id").set_index("item_id")["split"]
        counts: dict[str, int] = {}
        for row in log:
            if row["fallback"] != "single_view":
                continue
            split = str(by_item.get(row["item_id"], "unknown"))
            counts[split] = counts.get(split, 0) + 1
        excluded_splits: dict[str, int] = {}
        for row in log:
            if row["fallback"] != "excluded":
                continue
            split = str(by_item.get(row["item_id"], "unknown"))
            excluded_splits[split] = excluded_splits.get(split, 0) + 1
        print(f"single_view_by_split {counts} excluded_by_split {excluded_splits}", flush=True)
    return pd.DataFrame(kept), log
