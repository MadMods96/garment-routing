"""Label normalisation."""

from garmentroute.labels import (
    USAGE_CLASSES,
    balanced_class_weights,
    canonical_ternary,
    canonical_usage,
    is_gold_comment,
    normalise_record,
    normalise_stains,
    loose_gold_number,
    parse_gold_id,
    parse_ordinal,
    usage_class,
)


def test_usage_aliases_and_training_class():
    assert canonical_usage("Energy Recovery") == "Energy recovery"
    assert canonical_usage("Rcycle") == "Recycle"
    assert usage_class("Repair") == "Other"
    assert usage_class("Remake") == "Other"
    assert usage_class("Energy recovery") == "Other"
    assert usage_class("Reuse") == "Reuse"
    assert canonical_usage("not a route") is None


def test_legacy_stains_are_kept_out_of_the_three_class_head():
    yes = normalise_stains("Yes")
    assert yes.stains == "Minor_or_Major"
    assert yes.stains_3class is None
    assert yes.stains_legacy is True
    assert yes.has_stain is True
    no = normalise_stains(" No ")
    assert no.stains == "No"
    assert no.stains_3class is None
    assert no.has_stain is False
    minor = normalise_stains("minor")
    assert minor.stains_3class == "Minor"
    assert minor.has_stain is True
    assert normalise_stains("None").has_stain is False
    assert normalise_stains(None).has_stain is None


def test_missing_holes_are_not_filled_with_none():
    assert canonical_ternary(None) is None
    assert canonical_ternary("") is None
    assert canonical_ternary("None") == "None"
    assert canonical_ternary("major") == "Major"
    record = normalise_record({"usage": "Reuse", "condition": "4", "stains": "No"})
    assert record["holes"] is None
    assert record["smell"] is None
    assert record["condition"] == 4
    assert record["stains_legacy"] is True


def test_present_none_hole_stays_a_class():
    record = normalise_record({"holes": "None", "smell": "Minor", "usage": "Export"})
    assert record["holes"] == "None"
    assert record["smell"] == "Minor"
    assert record["usage_class"] == "Export"


def test_gold_comment_marker():
    assert parse_gold_id("Test 92") == "92"
    assert parse_gold_id("Test92") == "92"
    assert parse_gold_id("test 92") is None
    assert is_gold_comment("Test") is True
    assert parse_gold_id("Test") is None
    assert is_gold_comment("latest test of the zipper") is False
    assert loose_gold_number("test 92, detergent smell") == "92"
    assert loose_gold_number("Test for new update") is None


def test_class_weights_cap_other_at_five():
    counts = {"Reuse": 1400, "Export": 1200, "Recycle": 193, "Other": 40}
    uncapped = 2833 / (4 * 40)
    assert uncapped > 5
    weights = balanced_class_weights(counts, USAGE_CLASSES, cap=5.0)
    assert weights[USAGE_CLASSES.index("Other")] == 5.0
    assert max(weights) <= 5.0
    assert weights[USAGE_CLASSES.index("Reuse")] < 1.0


def test_class_weights_zero_for_missing_class():
    counts = {"Reuse": 10, "Export": 10, "Recycle": 10}
    weights = balanced_class_weights(counts, USAGE_CLASSES, cap=5.0)
    assert weights[USAGE_CLASSES.index("Other")] == 0.0


def test_ordinal_rejects_out_of_range():
    assert parse_ordinal("0") is None
    assert parse_ordinal(5) == 5
    assert parse_ordinal(3.0) == 3
