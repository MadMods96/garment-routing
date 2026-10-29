"""Index splits must not leak station3 or the gold set into training."""

import json
from pathlib import Path

import pandas as pd

from garmentroute.index import (
    assign_splits,
    choose_subset,
    cross_split_near_duplicates,
    iter_rows,
    resolve_near_duplicates,
)
from garmentroute.metrics import pairwise_kappa


def _write(root: Path, relative: str, payload: dict) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _garment(usage: str, stains: str = "None", holes: str | None = "None") -> dict:
    payload = {"usage": usage, "condition": 4, "pilling": 4, "stains": stains, "material": "100% cotton"}
    if holes is not None:
        payload["holes"] = holes
    return payload


def test_walk_normalises_and_skips_bad_files(tmp_path: Path):
    _write(tmp_path, "station1/sep2022/labels_2022_09_01_00_00_01.json", _garment("Reuse", stains="Yes", holes=None))
    _write(tmp_path, "station3/nov2023/labels_2023_11_01_00_00_01.json", _garment("Export"))
    _write(
        tmp_path,
        "test100/station1/dec2022/labels_2022_12_08_11_47_02.json",
        {**_garment("Export"), "comment": "Test 92"},
    )
    _write(
        tmp_path,
        "test100/station3/nov2023/labels_2023_11_23_14_45_51.json",
        {**_garment("Reuse"), "comment": "Test 92"},
    )
    _write(tmp_path, "json_errors/labels_2022_09_01_00_00_09.json", _garment("Reuse"))
    (tmp_path / "station1/sep2022/labels_2022_09_02_00_00_01.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "station1/sep2022/labels_2022_09_02_00_00_01.json").write_text("{", encoding="utf-8")
    rows, skipped = iter_rows(tmp_path)
    reasons = {row["reason"] for row in skipped}
    assert "inside json_errors" in reasons
    assert any(reason.startswith("json decode error") for reason in reasons)
    by_id = {row["source_item_id"]: row for row in rows}
    legacy = by_id["2022_09_01_00_00_01"]
    assert legacy["stains"] == "Minor_or_Major"
    assert legacy["stains_3class"] is None
    assert legacy["holes"] is None
    assert legacy["is_gold"] is False
    assert by_id["2023_11_01_00_00_01"]["station"] == "station3"
    gold = [row for row in rows if row["gold_id"] == "92"]
    assert len(gold) == 2
    assert all(row["is_gold"] for row in gold)
    assert {row["station"] for row in gold} == {"station1", "station3"}


def test_splits_hold_out_station3_and_gold():
    rows = []
    usages = (["Reuse"] * 20) + (["Export"] * 12) + (["Recycle"] * 8) + (["Repair"] * 6)
    for index, usage in enumerate(usages):
        rows.append(
            {
                "item_id": f"s1-{index}",
                "station": "station1" if index % 2 == 0 else "station2",
                "usage_class": "Other" if usage == "Repair" else usage,
                "is_gold": False,
                "duplicate_of": pd.NA,
            }
        )
    for index in range(7):
        rows.append(
            {
                "item_id": f"s3-{index}",
                "station": "station3",
                "usage_class": "Reuse",
                "is_gold": False,
                "duplicate_of": pd.NA,
            }
        )
    rows.append(
        {
            "item_id": "gold-1",
            "station": "station3",
            "usage_class": "Reuse",
            "is_gold": True,
            "duplicate_of": pd.NA,
        }
    )
    frame, meta = assign_splits(pd.DataFrame(rows), seed=42)
    train_stations = set(frame.loc[frame["split"] == "train", "station"])
    assert "station3" not in train_stations
    assert "station3" not in set(frame.loc[frame["split"].isin(["val", "test"]), "station"])
    assert frame.loc[frame["item_id"] == "gold-1", "split"].item() == "gold"
    assert set(frame.loc[frame["station"] == "station3", "split"]) <= {"cross_site", "gold"}
    assert meta["stratified"] is True
    counts = frame.loc[frame["station"].isin(["station1", "station2"]), "split"].value_counts()
    assert set(counts.index) == {"train", "val", "test"}


def test_subset_keeps_gold_and_oversamples_station3_non_reuse():
    rows = []
    for index in range(3):
        rows.append(
            {
                "item_id": f"g{index}",
                "station": "station1",
                "split": "gold",
                "is_gold": True,
                "usage": "Reuse",
                "usage_class": "Reuse",
                "duplicate_of": pd.NA,
            }
        )
    for index in range(30):
        rows.append(
            {
                "item_id": f"reuse{index}",
                "station": "station3",
                "split": "cross_site",
                "is_gold": False,
                "usage": "Reuse",
                "usage_class": "Reuse",
                "duplicate_of": pd.NA,
            }
        )
    for index, usage in enumerate(("Export", "Export", "Recycle", "Repair")):
        rows.append(
            {
                "item_id": f"other{index}",
                "station": "station3",
                "split": "cross_site",
                "is_gold": False,
                "usage": usage,
                "usage_class": "Other" if usage in {"Repair"} else usage,
                "duplicate_of": pd.NA,
            }
        )
    for split, count in (("train", 40), ("val", 20), ("test", 20)):
        for index in range(count):
            label = "Reuse" if index % 2 == 0 else "Export"
            rows.append(
                {
                    "item_id": f"{split}{index}",
                    "station": "station1",
                    "split": split,
                    "is_gold": False,
                    "usage": label,
                    "usage_class": label,
                    "duplicate_of": pd.NA,
                }
            )
    subset = choose_subset(pd.DataFrame(rows), target=20, seed=1, station3_n=8)
    assert set(subset.loc[subset["is_gold"], "item_id"]) == {"g0", "g1", "g2"}
    station3 = subset[subset["station"].eq("station3")]
    assert len(station3) == 8
    assert set(station3.loc[station3["usage"].ne("Reuse"), "usage"]) == {"Export", "Recycle", "Repair"}
    assert int(station3["usage"].ne("Reuse").sum()) == 4
    wargon = subset[subset["station"].eq("station1") & ~subset["is_gold"]]
    assert set(wargon["split"]) == {"train", "val", "test"}
    assert len(subset) == 20


def test_pairwise_kappa_groups_gold_ids():
    frame = pd.DataFrame(
        {
            "gold_id": ["1", "1", "2", "2", "3"],
            "usage": ["Reuse", "Reuse", "Export", "Recycle", "Reuse"],
        }
    )
    report = pairwise_kappa(frame, "usage")
    assert report["n_pairs"] == 2
    assert report["n_items_with_pairs"] == 2
    assert report["cohen_kappa"] is not None


def _member(item_id: str, split: str, score: int, digest: int, gold: bool = False) -> dict:
    return {
        "item_id": item_id,
        "split": split,
        "is_gold": gold,
        "score": (score, "", item_id),
        "digest": digest,
    }


def test_train_test_duplicate_drops_the_test_copy_even_when_it_is_more_complete():
    drops = resolve_near_duplicates(
        [
            _member("kept", "train", 1, 0),
            _member("dropped", "test", 9, 1),
        ],
        threshold=4,
    )
    assert len(drops) == 1
    assert drops[0]["removed_item_id"] == "dropped"
    assert drops[0]["kept_item_id"] == "kept"
    assert drops[0]["reason"] == "train_test"
    assert drops[0]["hamming"] == 1


def test_within_split_duplicate_keeps_the_more_complete_row():
    drops = resolve_near_duplicates(
        [
            _member("thin", "train", 1, 0),
            _member("full", "train", 4, 0),
        ],
        threshold=4,
    )
    assert drops[0]["removed_item_id"] == "thin"
    assert drops[0]["kept_item_id"] == "full"
    assert drops[0]["reason"] == "within_split"


def test_gold_row_is_kept_when_a_training_front_matches_it():
    drops = resolve_near_duplicates(
        [
            _member("gold-a", "gold", 2, 0, gold=True),
            _member("gold-b", "gold", 1, 0, gold=True),
            _member("train-copy", "train", 9, 0),
        ],
        threshold=4,
    )
    assert [row["removed_item_id"] for row in drops] == ["train-copy"]
    assert drops[0]["kept_split"] == "gold"


def test_a_chain_does_not_drop_a_front_more_than_threshold_bits_from_the_keeper():
    drops = resolve_near_duplicates(
        [
            _member("train", "train", 1, 0),
            _member("near", "test", 1, 0xF),
            _member("far", "test", 1, 0xFF),
        ],
        threshold=4,
    )
    assert [row["removed_item_id"] for row in drops] == ["near"]
    assert drops[0]["reason"] == "train_test"
    assert drops[0]["hamming"] == 4


def test_kept_rows_do_not_report_two_gold_fronts_as_a_split_crossing():
    frame = pd.DataFrame(
        [
            {"item_id": "g1", "split": "gold", "is_gold": True, "duplicate_of": pd.NA, "phash": "0000000000000000"},
            {"item_id": "g2", "split": "gold", "is_gold": True, "duplicate_of": pd.NA, "phash": "0000000000000000"},
            {"item_id": "t1", "split": "train", "is_gold": False, "duplicate_of": pd.NA, "phash": "00000000000000ff"},
            {"item_id": "e1", "split": "test", "is_gold": False, "duplicate_of": "t1", "phash": "00000000000000fe"},
        ]
    )
    assert cross_split_near_duplicates(frame, threshold=4) == []
    leaked = frame.copy()
    leaked.loc[leaked["item_id"].eq("e1"), "duplicate_of"] = pd.NA
    assert cross_split_near_duplicates(leaked, threshold=4)
