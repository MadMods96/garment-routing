"""Missing front or back photographs stay in the loader as a single view."""

from pathlib import Path

import pandas as pd

from garmentroute.views import filter_to_subset, find_subset_members, resolve_views


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


def test_scandir_index_is_used_for_membership_not_exists(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "front.jpg").write_bytes(b"x")
    frame = pd.DataFrame(
        [
            {"item_id": "keep", "front_path": "a/front.jpg", "back_path": "a/back.jpg", "split": "train"},
            {"item_id": "drop", "front_path": "missing.jpg", "back_path": "gone.jpg", "split": "gold"},
        ]
    )
    from garmentroute.views import list_image_relatives

    available = list_image_relatives(tmp_path)
    assert available == {"a/front.jpg"}
    kept, log = resolve_views(frame, tmp_path, available)
    assert kept["item_id"].tolist() == ["keep"]
    assert kept["view"].tolist() == ["front"]
    assert [row["fallback"] for row in log] == ["single_view", "excluded"]
    assert [row["item_id"] for row in log if row["fallback"] == "excluded"] == ["drop"]


def test_filter_to_subset_keeps_listed_paths(tmp_path: Path):
    members = tmp_path / "subset_members.txt"
    members.write_text("a/front.jpg\nb/front.jpg\n", encoding="utf-8")
    frame = pd.DataFrame(
        [
            {"item_id": "1", "front_path": "a/front.jpg", "back_path": "a/back.jpg", "split": "train"},
            {"item_id": "2", "front_path": "c/front.jpg", "back_path": "c/back.jpg", "split": "train"},
        ]
    )
    kept = filter_to_subset(frame, members)
    assert kept["item_id"].tolist() == ["1"]


def test_filter_to_subset_keeps_listed_item_ids(tmp_path: Path):
    members = tmp_path / "subset_ids.txt"
    members.write_text("keep\n", encoding="utf-8")
    frame = pd.DataFrame(
        [
            {"item_id": "keep", "front_path": "a.jpg", "back_path": "b.jpg"},
            {"item_id": "drop", "front_path": "c.jpg", "back_path": "d.jpg"},
        ]
    )
    kept = filter_to_subset(frame, members)
    assert kept["item_id"].tolist() == ["keep"]


def test_find_subset_members_prefers_members_file(tmp_path: Path):
    (tmp_path / "subset_members.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "subset_ids.txt").write_text("y\n", encoding="utf-8")
    assert find_subset_members(tmp_path).name == "subset_members.txt"


def test_both_images_are_not_logged(tmp_path: Path):
    (tmp_path / "front.jpg").write_bytes(b"x")
    (tmp_path / "back.jpg").write_bytes(b"x")
    kept, log = resolve_views(_frame("front.jpg", "back.jpg"), tmp_path)
    assert kept["view"].tolist() == ["both"]
    assert log == []
