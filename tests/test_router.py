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


def test_clean_cotton_goes_to_reuse_and_wool_with_a_minor_hole_goes_to_repair():
    cotton = decide(GarmentSignals(condition=5, holes="None", stains="None"), empty_passport(), CONFIG)
    wool = decide(
        GarmentSignals(condition=4, holes="Minor", stains="None"),
        coerce_passport({"fibre_composition": "100% wool"}),
        CONFIG,
    )
    assert cotton.route == "Reuse"
    assert wool.route == "Repair"
    assert any("wool" in reason for reason in wool.reasons)
    assert any("condition 4/5" in reason for reason in wool.reasons)


def test_fibre_bonus_uses_the_passport_not_a_side_channel():
    signals = GarmentSignals(condition=4, holes="Minor", stains="None")
    without = score_routes(signals, empty_passport(), CONFIG)[0]["Repair"]
    with_wool = score_routes(signals, coerce_passport({"fibre_composition": "100% ull"}), CONFIG)[0]["Repair"]
    assert with_wool > without


def test_major_holes_and_poor_condition_cannot_be_reuse():
    decision = decide(GarmentSignals(condition=2, holes="Major"), empty_passport(), CONFIG)
    assert decision.route != "Reuse"
    assert "Reuse" not in decision.scores
    assert any("major holes" in reason for reason in decision.reasons)


def test_low_usage_confidence_goes_to_a_human():
    decision = decide(
        GarmentSignals(condition=5, holes="None", usage_confidence=0.2),
        empty_passport(),
        CONFIG,
    )
    assert decision.route == "HUMAN_REVIEW"
    assert decision.confidence == pytest.approx(0.2)


def test_capacity_keeps_the_repair_slot_for_the_largest_gain():
    def item(item_id: str, condition: int):
        return (
            item_id,
            GarmentSignals(condition=condition, holes="Minor", stains="None"),
            coerce_passport({"fibre_composition": "100% wool", "product_id": item_id}),
        )

    config = dict(CONFIG)
    config["capacity"] = {"Repair": 1}
    decisions = assign_batch([item("c5", 5), item("c3", 3), item("c4", 4)], config)
    routes = {name: decision.route for name, decision in zip(["c5", "c3", "c4"], decisions)}
    assert list(routes.values()).count("Repair") == 1
    assert routes["c3"] == "Repair"
    assert routes["c5"] != "Repair"
    again = assign_batch([item("c5", 5), item("c3", 3), item("c4", 4)], config)
    assert [decision.route for decision in again] == [decision.route for decision in decisions]
