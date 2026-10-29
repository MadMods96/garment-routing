"""Decision layer: hard rules, then illustrative route values, then capacity.

Every weight in the YAML config is a placeholder. This module does not estimate
life-cycle impact. It applies the config it is given and says which rule fired.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from garmentroute.util import load_yaml

HUMAN_REVIEW = "HUMAN_REVIEW"


@dataclass
class GarmentSignals:
    """What the router is allowed to see for one garment."""

    condition: int | None = None
    pilling: int | None = None
    stains: str | None = None
    holes: str | None = None
    smell: str | None = None
    usage_confidence: float | None = None
    usage_probs: dict[str, float] | None = None
    damageimage: str | None = None
    damageloc: str | None = None
    damage: str | None = None
    damage2image: str | None = None
    damage2loc: str | None = None
    damage2: str | None = None
    damage3image: str | None = None
    damage3loc: str | None = None
    damage3: str | None = None


@dataclass
class Decision:
    """One routing decision."""

    route: str
    confidence: float
    reasons: list[str]
    alternatives: list[dict[str, Any]]
    scores: dict[str, float] = field(default_factory=dict)
    capacity_displaced: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "alternatives": self.alternatives,
            "scores": self.scores,
            "capacity_displaced": self.capacity_displaced,
        }


def load_router_config(path: str) -> dict[str, Any]:
    """Load router YAML and refuse a file that does not mark its numbers as illustrative."""
    config = load_yaml(path)
    note = str(config.get("placeholder_notice", ""))
    if "illustrative" not in note.lower():
        raise ValueError(
            "router config must include placeholder_notice text that calls the numbers illustrative"
        )
    if "route_value" not in config or "route_order" not in config:
        raise ValueError("router config needs route_order and route_value")
    return config


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return set(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def matching_fibres(composition: str | None, keywords: list[str]) -> list[str]:
    """Return keywords that appear as whole tokens in a fibre string."""
    tokens = _tokens(composition)
    hits = []
    for keyword in keywords:
        parts = keyword.lower().split()
        if parts and all(part in tokens for part in parts):
            hits.append(keyword)
    return hits


def _rule_fires(rule: Mapping[str, Any], signals: GarmentSignals) -> bool:
    if "holes" in rule and signals.holes != rule["holes"]:
        return False
    if "stains" in rule and signals.stains != rule["stains"]:
        return False
    if "smell" in rule and signals.smell != rule["smell"]:
        return False
    if "condition_lte" in rule:
        if signals.condition is None or signals.condition > int(rule["condition_lte"]):
            return False
    if "condition_gte" in rule:
        if signals.condition is None or signals.condition < int(rule["condition_gte"]):
            return False
    return True


def _attribute_reasons(signals: GarmentSignals) -> list[str]:
    reasons: list[str] = []
    if signals.condition is not None:
        reasons.append(f"condition {signals.condition}/5")
    if signals.pilling is not None:
        reasons.append(f"pilling {signals.pilling}/5")
    if signals.holes in {"Minor", "Major"}:
        side = _side_for(("hole", "holes"), signals)
        reasons.append(f"{signals.holes.lower()} hole" + (f" on {side}" if side else ""))
    if signals.stains == "Minor_or_Major":
        reasons.append("stain recorded on the older yes/no scale")
    elif signals.stains in {"Minor", "Major"}:
        side = _side_for(("stain", "stains"), signals)
        reasons.append(f"{signals.stains.lower()} stain" + (f" on {side}" if side else ""))
    if signals.smell in {"Minor", "Major"}:
        reasons.append(f"{signals.smell.lower()} smell")
    return reasons


def _side_for(keywords: tuple[str, ...], signals: GarmentSignals) -> str | None:
    triples = (
        (signals.damageimage, signals.damageloc, signals.damage),
        (signals.damage2image, signals.damage2loc, signals.damage2),
        (signals.damage3image, signals.damage3loc, signals.damage3),
    )
    for image, loc, text in triples:
        haystack = (text or "").lower()
        if any(keyword in haystack for keyword in keywords) and image:
            where = image.strip()
            if loc and loc.lower() not in {"none", ""}:
                where = f"{where} {loc.strip()}"
            return where
    return None


def score_routes(
    signals: GarmentSignals,
    passport: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], list[str], list[str]]:
    """Return route scores, human-readable reasons, and fibres that matched.

    Forbidden routes are omitted. Higher scores are preferred.
    """
    scores = {route: float(config["route_value"][route]) for route in config["route_order"]}
    reasons: list[str] = []
    for rule in config.get("hard_rules", []):
        if _rule_fires(rule, signals):
            for route in rule.get("forbid", []):
                scores.pop(route, None)
            if rule.get("reason"):
                reasons.append(str(rule["reason"]))
    bonus_routes = set(config.get("condition_bonus_routes", []))
    condition_table = config.get("condition_bonus", {})
    if signals.condition is not None and str(signals.condition) in condition_table:
        bonus = float(condition_table[str(signals.condition)])
        for route in list(scores):
            if route in bonus_routes:
                scores[route] += bonus
    adjustments = config.get("damage_adjustments", {})
    for attribute in ("holes", "stains", "smell"):
        level = getattr(signals, attribute)
        if level is None:
            continue
        for route, delta in adjustments.get(attribute, {}).get(level, {}).items():
            if route in scores:
                scores[route] += float(delta)
    fibres = matching_fibres(
        passport.get("fibre_composition") if isinstance(passport, Mapping) else None,
        list(config.get("high_value_fibres", [])),
    )
    if fibres:
        for route, delta in config.get("high_value_bonus", {}).items():
            if route in scores:
                scores[route] += float(delta)
        composition = passport.get("fibre_composition")
        reasons.append(f"{composition} (high value fibre: {', '.join(fibres)})")
    for route, delta in config.get("env_weight", {}).items():
        if route in scores:
            scores[route] += float(delta)
    weight = float(config.get("model_probability_weight", 0.0))
    if signals.usage_probs and weight:
        for route, probability in signals.usage_probs.items():
            if route in scores:
                scores[route] += weight * float(probability)
    reasons.extend(_attribute_reasons(signals))
    return scores, reasons, fibres


def _softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    if not scores:
        return {}
    temperature = max(float(temperature), 1e-6)
    names = list(scores)
    values = [scores[name] / temperature for name in names]
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights) or 1.0
    return {name: weight / total for name, weight in zip(names, weights)}


def _alternatives(scores: Mapping[str, float], probabilities: Mapping[str, float], chosen: str) -> list[dict[str, Any]]:
    ordered = sorted(scores, key=lambda name: scores[name], reverse=True)
    rows = []
    for name in ordered:
        if name == chosen:
            continue
        rows.append(
            {
                "route": name,
                "score": scores[name],
                "probability": probabilities.get(name),
            }
        )
        if len(rows) == 3:
            break
    return rows


def route_probabilities(
    signals: GarmentSignals,
    passport: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Softmax over the illustrative route scores for one garment.

    This is a normalisation of the config scores, not a trained class probability.
    """
    scores, _reasons, _fibres = score_routes(signals, passport, config)
    return _softmax(scores, float(config.get("score_temperature", 1.0)))


def decide(
    signals: GarmentSignals,
    passport: Mapping[str, Any],
    config: Mapping[str, Any],
    abstain_threshold: float | None = None,
) -> Decision:
    """Route one garment. Low model confidence becomes HUMAN_REVIEW before scores are used."""
    threshold = config.get("abstain_confidence") if abstain_threshold is None else abstain_threshold
    scores, reasons, _fibres = score_routes(signals, passport, config)
    probabilities = _softmax(scores, float(config.get("score_temperature", 1.0)))
    if (
        threshold is not None
        and signals.usage_confidence is not None
        and signals.usage_confidence < float(threshold)
    ):
        return Decision(
            route=HUMAN_REVIEW,
            confidence=float(signals.usage_confidence),
            reasons=["usage confidence below the abstention threshold", *reasons],
            alternatives=_alternatives(scores, probabilities, HUMAN_REVIEW),
            scores=scores,
        )
    if not scores:
        return Decision(
            route=HUMAN_REVIEW,
            confidence=0.0,
            reasons=["no route remained after hard rules", *reasons],
            alternatives=[],
            scores={},
        )
    route = max(scores, key=lambda name: scores[name])
    return Decision(
        route=route,
        confidence=float(probabilities.get(route, 0.0)),
        reasons=reasons,
        alternatives=_alternatives(scores, probabilities, route),
        scores=scores,
    )


@dataclass
class _BatchItem:
    item_id: str
    signals: GarmentSignals
    passport: Mapping[str, Any]
    decision: Decision
    held: bool = False


def assign_batch(
    items: list[tuple[str, GarmentSignals, Mapping[str, Any]]],
    config: Mapping[str, Any],
    abstain_threshold: float | None = None,
) -> list[Decision]:
    """Assign routes for a batch when some routes have a slot limit.

    Unconstrained routes are the ones missing from ``capacity``. Among garments
    that gain value from a constrained route, slots go to the largest gain
    first. This is a greedy heuristic, not a claim about an optimal facility plan.
    """
    prepared: list[_BatchItem] = []
    for item_id, signals, passport in items:
        decision = decide(signals, passport, config, abstain_threshold=abstain_threshold)
        prepared.append(
            _BatchItem(item_id=str(item_id), signals=signals, passport=passport, decision=decision)
        )
    remaining = {route: int(slots) for route, slots in config.get("capacity", {}).items()}
    for item in prepared:
        if item.decision.route == HUMAN_REVIEW or item.decision.route not in remaining:
            item.held = True
    for route in list(remaining):
        seekers: list[tuple[float, str, _BatchItem]] = []
        for item in prepared:
            if item.held or item.decision.route != route or route not in item.decision.scores:
                continue
            others = [value for name, value in item.decision.scores.items() if name != route]
            best_other = max(others) if others else float("-inf")
            seekers.append((item.decision.scores[route] - best_other, item.item_id, item))
        seekers.sort(key=lambda row: (-row[0], row[1]))
        for marginal, _item_id, item in seekers:
            if remaining[route] <= 0 or marginal <= 0:
                break
            remaining[route] -= 1
            item.held = True
    temperature = float(config.get("score_temperature", 1.0))
    for item in prepared:
        if item.held:
            continue
        scores = dict(item.decision.scores)
        feasible = {
            route: score
            for route, score in scores.items()
            if route not in remaining or remaining[route] > 0
        }
        if not feasible:
            item.decision = Decision(
                route=HUMAN_REVIEW,
                confidence=item.decision.confidence,
                reasons=["no route had remaining illustrative capacity", *item.decision.reasons],
                alternatives=item.decision.alternatives,
                scores=scores,
                capacity_displaced=True,
            )
            continue
        route = max(feasible, key=lambda name: feasible[name])
        if route in remaining:
            remaining[route] -= 1
        displaced = route != item.decision.route
        probabilities = _softmax(feasible, temperature)
        reasons = list(item.decision.reasons)
        if displaced:
            reasons.append("assigned under the illustrative capacity limit")
        item.decision = Decision(
            route=route,
            confidence=float(probabilities.get(route, item.decision.confidence)),
            reasons=reasons,
            alternatives=_alternatives(scores, probabilities, route),
            scores=scores,
            capacity_displaced=displaced,
        )
    return [item.decision for item in prepared]
