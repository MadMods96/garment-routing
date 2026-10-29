"""The results block rounds JSON floats to 3 decimals and must not invent a second score."""

from pathlib import Path

from garmentroute.reporting import render_results, splice_readme


def test_render_copies_the_json_value_and_skips_missing_models(tmp_path: Path):
    results = {
        "dataset": "clothing_v3",
        "smoke": False,
        "seed": 42,
        "split_meta": {"stratified": True},
        "majority": {
            "test": {
                "usage_class": {"accuracy": 0.5, "macro_f1": 0.19290500904770425, "mae": None, "within_1_accuracy": None}
            }
        },
        "gold": {"expert_vs_expert_usage": {"cohen_kappa": 0.4, "raw_agreement": 0.7, "n_pairs": 10}},
        "router": {"on_expert_attributes": {}},
        "front": None,
        "fusion": None,
    }
    eda = {"n_rows": 3, "n_gold": 1}
    text = render_results(results, eda)
    assert "0.500" in text
    assert "0.193" in text
    assert "0.19290500904770425" not in text
    assert "0.99" not in text
    assert "not computed" in text
    assert "Macro-F1, all 6 expert classes, averages F1" in text
    assert "Macro-F1, 4 training classes, averages F1" in text
    assert "Front-only and two-view checkpoints have not been scored." in text


def test_gold_census_separates_garments_from_files():
    import pandas as pd

    from garmentroute.eda import gold_census

    frame = pd.DataFrame(
        [
            {"is_gold": True, "gold_id": "1", "comment": "Test 1", "json_path": "test100/a.json"},
            {"is_gold": True, "gold_id": "1", "comment": "Test 1 again", "json_path": "test100/b.json"},
            {"is_gold": True, "gold_id": None, "comment": "test 1, detergent smell", "json_path": "test100/c.json"},
            {"is_gold": True, "gold_id": None, "comment": "Test for new update", "json_path": "station1/d.json"},
        ]
    )
    census = gold_census(frame)
    assert census["n_split_rows"] == 4
    assert census["n_json_files"] == 4
    assert census["n_garments"] == 1
    assert census["n_annotations_with_item_number"] == 2
    assert census["mean_annotations_per_garment"] == 2.0
    assert census["n_lowercase_annotations_of_same_garments"] == 1
    assert census["n_unnumbered_comments"] == 1
    assert census["unnumbered_comment_text"] == ["Test for new update"]
    text = render_results(
        {
            "dataset": "clothing_v3",
            "smoke": False,
            "seed": 1,
            "split_meta": {"stratified": True},
            "majority": {},
            "gold": {"expert_vs_expert_usage": {"cohen_kappa": 0.2, "raw_agreement": 0.5, "n_pairs": 1, "n_items": 1, "n_items_with_pairs": 1}},
            "router": {"on_expert_attributes": {}},
            "front": None,
            "fusion": None,
        },
        {"n_rows": 4, "gold_census": census},
    )
    assert "Garments. One garment is one item number" in text
    assert "Distinct JSON files" in text
    assert "Annotation records in the gold split" in text
    assert "`Test for new update`" in text
    assert "2.0" in text


def test_smoke_results_are_not_rendered_as_the_study():
    text = render_results(
        {"dataset": "smoke", "smoke": True, "majority": {"test": {"usage_class": {"accuracy": 1}}}},
        None,
    )
    assert "No clothing_v3" in text
    assert "accuracy" not in text


def test_splice_replaces_only_the_marked_block(tmp_path: Path):
    readme = "Before\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nAfter\n"
    updated = splice_readme(readme, "new\n")
    assert updated.startswith("Before\n")
    assert updated.endswith("After\n")
    assert "old" not in updated
    assert "new" in updated
