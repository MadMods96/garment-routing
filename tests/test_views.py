"""Missing front or back photographs stay in the loader as a single view."""

from pathlib import Path

import pandas as pd

from garmentroute.views import resolve_views


def _frame(front: str, back: str) -> pd.DataFrame:
    return pd.DataFrame([{"item_id": "g1", "front_path": front, "back_path": back}])


def test_missing_back_falls_back_to_the_front(tmp_path: Path):
    (tmp_path / "front.jpg").write_bytes(b"x")
    kept, log = resolve_views(_frame("front.jpg", "back.jpg"), tmp_path)
    assert kept["view"].tolist() == ["front"]
    assert log == [{"item_id": "g1", "missing": "back", "fallback": "single_view", "used_view": "front"}]


def test_missing_front_falls_back_to_the_back(tmp_path: Path):
    (tmp_path / "back.jpg").write_bytes(b"x")
    kept, log = resolve_views(_frame("front.jpg", "back.jpg"), tmp_path)
    assert kept["view"].tolist() == ["back"]
    assert log[0]["missing"] == "front"
    assert log[0]["used_view"] == "back"


def test_neither_image_is_excluded_and_logged(tmp_path: Path):
    kept, log = resolve_views(_frame("front.jpg", "back.jpg"), tmp_path)
    assert kept.empty
    assert log == [{"item_id": "g1", "missing": "both", "fallback": "excluded", "used_view": ""}]


def test_both_images_are_not_logged(tmp_path: Path):
    (tmp_path / "front.jpg").write_bytes(b"x")
    (tmp_path / "back.jpg").write_bytes(b"x")
    kept, log = resolve_views(_frame("front.jpg", "back.jpg"), tmp_path)
    assert kept["view"].tolist() == ["both"]
    assert log == []
