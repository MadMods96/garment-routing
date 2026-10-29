"""Decide which photograph to load when a garment is missing a view."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _is_file(image_root: Path, relative: object) -> bool:
    if relative is None or (isinstance(relative, float) and pd.isna(relative)):
        return False
    text = str(relative).strip()
    if not text or text.lower() == "nan":
        return False
    return (image_root / text).is_file()


def resolve_views(frame: pd.DataFrame, image_root: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Keep garments with at least one image and record a missing front or back.

    ``view`` is ``both``, ``front``, or ``back``. A garment with neither file
    is omitted. The log lists only garments that are missing a view.
    """
    root = Path(image_root)
    kept: list[dict] = []
    log: list[dict[str, str]] = []
    for record in frame.to_dict("records"):
        front_ok = _is_file(root, record.get("front_path"))
        back_ok = _is_file(root, record.get("back_path"))
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
    return pd.DataFrame(kept), log
