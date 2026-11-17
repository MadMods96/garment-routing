"""Router rules, fibre value, abstention, and capacity."""

from pathlib import Path

import pytest

from garmentroute.passport import SCHEMA_NOTE, coerce_passport, empty_passport
from garmentroute.router import GarmentSignals, assign_batch, decide, load_router_config, score_routes

CONFIG = load_router_config(Path(__file__).resolve().parents[1] / "configs" / "router.yaml")


def test_config_declares_that_its_numbers_are_illustrative():
    assert "illustrative" in CONFIG["placeholder_notice"].lower()
    assert "not an official" in SCHEMA_NOTE.lower() or "Not an official" in SCHEMA_NOTE


def test_config_without_the_placeholder_notice_is_refused(tmp_path: Path):
    path = tmp_path / "router.yaml"
    path.write_text("route_order: [Reuse]\nroute_value:\n  Reuse: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_router_config(path)


def test_final_route_follows_model_usage_when_confident():
    decision = decide(
        GarmentSignals(
            condition=5,
            holes="None",
            stains="None",
            usage_confidence=0.9,
            usage_class="Export",
            holes_confidence=0.9,
            stains_confidence=0.9,
        ),
        empty_passport(),
        CONFIG,
    )
    assert decision.route == "Export"
    assert any("soft rule route would be" in reason for reason in decision.reasons)


def test_wool_minor_hole_follows_model_other_when_rules_prefer_repair():
    decision = decide(
        GarmentSignals(
            condition=4,
            holes="Minor",
            stains="None",
            usage_confidence=0.9,
            usage_class="Other",
            holes_confidence=0.9,
            stains_confidence=0.9,
        ),
        coerce_passport({"fibre_composition": "100% wool"}),
        CONFIG,
    )
    assert decision.route == "Other"
    assert decision.rule_route == "Repair"
    # Collapse(Repair)=Other, so soft disagreement is not escalated or announced.


def test_fibre_bonus_uses_the_passport_not_a_side_channel():
    signals = GarmentSignals(condition=4, holes="Minor", stains="None")
    without = score_routes(signals, empty_passport(), CONFIG, apply_hard_rules=False)[0]["Repair"]
    with_wool = score_routes(
        signals, coerce_passport({"fibre_composition": "100% ull"}), CONFIG, apply_hard_rules=False
    )[0]["Repair"]
    assert with_wool > without


def test_hard_rule_contradicting_model_goes_to_human_review():
    decision = decide(
        GarmentSignals(
            condition=2,
            holes="Major",
            usage_confidence=0.9,
            usage_class="Reuse",
            holes_confidence=0.9,
            stains_confidence=0.9,
        ),
        empty_passport(),
        CONFIG,
    )
    assert decision.route == "HUMAN_REVIEW"
    assert any("hard-rule safety override" in reason for reason in decision.reasons)


def test_major_holes_reuse_at_condition_3_goes_to_human_review():
    """Demo fallthrough case: Major holes + Reuse must not stay Reuse (forbid, not condition gate).

    Abstain is disabled here so the hard-rule path is what fires at usage 45%.
    """
    decision = decide(
        GarmentSignals(
            condition=3,
            stains="Minor",
            holes="Major",
            usage_confidence=0.45,
            usage_class="Reuse",
            usage_probs={"Reuse": 0.45, "Export": 0.25, "Recycle": 0.20, "Other": 0.10},
            holes_confidence=0.9,
            stains_confidence=0.9,
        ),
        empty_passport(),
        CONFIG,
        abstain_threshold=0.0,
    )
    assert decision.route == "HUMAN_REVIEW"
    assert any("hard-rule safety override" in reason for reason in decision.reasons)


def test_hard_rule_agreeing_with_model_keeps_model_route():
    decision = decide(
        GarmentSignals(
            condition=2,
            holes="Major",
            usage_confidence=0.9,
            usage_class="Recycle",
            holes_confidence=0.9,
            stains_confidence=0.9,
        ),
        empty_passport(),
        CONFIG,
    )
    assert decision.route == "Recycle"


def test_weak_hard_rule_signal_goes_to_human():
    decision = decide(
        GarmentSignals(
            condition=2,
            holes="Major",
            usage_confidence=0.9,
            usage_class="Reuse",
            holes_confidence=0.2,
            stains_confidence=0.9,
        ),
        empty_passport(),
        CONFIG,
    )
    assert decision.route == "HUMAN_REVIEW"
    assert any("below confidence threshold" in reason for reason in decision.reasons)


def test_high_value_fibre_does_not_waive_reuse_forbid():
    """Fibre may skip Recycle force, but Major holes still forbid Reuse → HUMAN_REVIEW."""
    decision = decide(
        GarmentSignals(
            condition=2,
            holes="Major",
            usage_confidence=0.9,
            usage_class="Reuse",
            holes_confidence=0.9,
            stains_confidence=0.9,
        ),
        coerce_passport({"fibre_composition": "100% wool"}),
        CONFIG,
    )
    assert decision.route == "HUMAN_REVIEW"
    assert any("hard-rule safety override" in reason for reason in decision.reasons)


def test_high_value_fibre_waives_recycle_force_for_repair():
    decision = decide(
        GarmentSignals(
            condition=2,
            holes="Major",
            usage_confidence=0.9,
            usage_class="Other",
            holes_confidence=0.9,
            stains_confidence=0.9,
        ),
        coerce_passport({"fibre_composition": "100% wool"}),
        CONFIG,
    )
    assert decision.route == "Other"


def test_low_usage_confidence_goes_to_a_human():
    decision = decide(
        GarmentSignals(condition=5, holes="None", usage_confidence=0.2, usage_class="Reuse"),
        empty_passport(),
        CONFIG,
    )
    assert decision.route == "HUMAN_REVIEW"
    assert decision.confidence == pytest.approx(0.2)


def test_capacity_still_limits_explicit_constrained_routes():
    def item(item_id: str, condition: int):
        return (
            item_id,
            GarmentSignals(
                condition=condition,
                holes="Minor",
                stains="None",
                usage_confidence=0.9,
                usage_class="Repair",
                holes_confidence=0.9,
                stains_confidence=0.9,
            ),
            coerce_passport({"fibre_composition": "100% wool", "product_id": item_id}),
        )

    config = dict(CONFIG)
    config["capacity"] = {"Repair": 1}
    decisions = assign_batch([item("c5", 5), item("c3", 3), item("c4", 4)], config)
    assert list(decision.route for decision in decisions).count("Repair") == 1
