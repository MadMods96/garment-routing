"""Decision layer: model-first routing with hard-rule safety overrides.

Every weight in the YAML config is a placeholder. This module does not estimate
life-cycle impact. It applies the config it is given and says which rule fired.

Priority in ``decide``:
1. Low usage confidence → HUMAN_REVIEW.
2. Hard-rule safety: Major holes/stains forbid Reuse/Export (confident damage
   head). High-value fibre may skip Recycle ``force`` only — never the forbid.
   A contradicting model usage class → HUMAN_REVIEW.
3. Otherwise final route = model usage class. Soft rule-vs-model disagreement
   is recorded as an explanation, not an escalation.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from garmentroute.util import load_yaml

HUMAN_REVIEW = "HUMAN_REVIEW"
_OTHER_ROUTES = {"Repair", "Remake", "Energy recovery"}


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
    usage_class: str | None = None
    holes_confidence: float | None = None
    stains_confidence: float | None = None
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
    rule_route: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "alternatives": self.alternatives,
            "scores": self.scores,
            "capacity_displaced": self.capacity_displaced,
            "rule_route": self.rule_route,
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


def collapse_usage_route(route: str) -> str:
    """Map six-way routes onto the four-way usage head (Repair/Remake/Energy → Other)."""
    return "Other" if route in _OTHER_ROUTES else route


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


def _trigger_confidence(rule: Mapping[str, Any], signals: GarmentSignals) -> float | None:
    """Confidence of the damage head that would make this hard rule fire."""
    if "holes" in rule:
        return signals.holes_confidence
    if "stains" in rule:
        return signals.stains_confidence
    return None


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
    *,
    apply_hard_rules: bool = True,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Return route scores, human-readable reasons, and fibres that matched.

    Forbidden routes are omitted. A ``force`` list keeps only those routes.
    High-value fibre skips ``force`` so Repair/Reuse bonuses can still win.
    """
    scores = {route: float(config["route_value"][route]) for route in config["route_order"]}
    reasons: list[str] = []
    fibres = matching_fibres(
        passport.get("fibre_composition") if isinstance(passport, Mapping) else None,
        list(config.get("high_value_fibres", [])),
    )
    if apply_hard_rules:
        for rule in config.get("hard_rules", []):
            if not _rule_fires(rule, signals):
                continue
            skip_force = bool(rule.get("skip_force_if_high_value_fibre")) and bool(fibres)
            if rule.get("force") and not skip_force:
                allowed = set(rule["force"])
                scores = {route: scores[route] for route in scores if route in allowed}
            for route in rule.get("forbid", []):
                scores.pop(route, None)
            if rule.get("reason"):
                reasons.append(str(rule["reason"]))
            if skip_force and rule.get("force"):
                reasons.append("high-value fibre: skipped Recycle force")
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


def _model_usage_class(signals: GarmentSignals) -> str | None:
    if signals.usage_class:
        return signals.usage_class
    if signals.usage_probs:
        return max(signals.usage_probs, key=signals.usage_probs.get)
    return None


def _hard_rule_contradicts_model(
    rule: Mapping[str, Any],
    model_usage: str,
    *,
    skip_force: bool = False,
) -> bool:
    """True when a safety force/forbid conflicts with the model's usage class.

    ``skip_force`` (high-value fibre) drops only the Recycle force check; forbid
    Reuse/Export still applies so Major damage never falls through to resale.
    """
    forbidden = set(rule.get("forbid") or [])
    if model_usage in forbidden:
        return True
    if skip_force:
        return False
    forced = set(rule.get("force") or [])
    if forced and model_usage not in forced:
        return True
    return False


def route_probabilities(
    signals: GarmentSignals,
    passport: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Softmax over the illustrative route scores for one garment.

    This is a normalisation of the config scores, not a trained class probability.
    """
    scores, _reasons, _fibres = score_routes(signals, passport, config, apply_hard_rules=False)
    return _softmax(scores, float(config.get("score_temperature", 1.0)))


def decide(
    signals: GarmentSignals,
    passport: Mapping[str, Any],
    config: Mapping[str, Any],
    abstain_threshold: float | None = None,
) -> Decision:
    """Route one garment: model usage first, hard rules only as safety overrides."""
    usage_threshold = config.get("abstain_confidence") if abstain_threshold is None else abstain_threshold
    hard_threshold = config.get("hard_rule_min_confidence", usage_threshold)
    soft_scores, soft_reasons, fibres = score_routes(
        signals, passport, config, apply_hard_rules=False
    )
    soft_probabilities = _softmax(soft_scores, float(config.get("score_temperature", 1.0)))
    rule_route = max(soft_scores, key=soft_scores.get) if soft_scores else None
    model_usage = _model_usage_class(signals)
    usage_confidence = float(signals.usage_confidence) if signals.usage_confidence is not None else 0.0

    if (
        usage_threshold is not None
        and signals.usage_confidence is not None
        and signals.usage_confidence < float(usage_threshold)
    ):
        return Decision(
            route=HUMAN_REVIEW,
            confidence=usage_confidence,
            reasons=["usage confidence below the abstention threshold", *soft_reasons],
            alternatives=_alternatives(soft_scores, soft_probabilities, HUMAN_REVIEW),
            scores=soft_scores,
            rule_route=rule_route,
        )

    if not model_usage:
        return Decision(
            route=HUMAN_REVIEW,
            confidence=usage_confidence,
            reasons=["no model usage class available", *soft_reasons],
            alternatives=_alternatives(soft_scores, soft_probabilities, HUMAN_REVIEW),
            scores=soft_scores,
            rule_route=rule_route,
        )

    for rule in config.get("hard_rules", []):
        if not _rule_fires(rule, signals):
            continue
        skip_force = bool(rule.get("skip_force_if_high_value_fibre")) and bool(fibres)
        trigger_conf = _trigger_confidence(rule, signals)
        if hard_threshold is not None and (trigger_conf is None or trigger_conf < float(hard_threshold)):
            return Decision(
                route=HUMAN_REVIEW,
                confidence=float(trigger_conf) if trigger_conf is not None else 0.0,
                reasons=[
                    "hard-rule safety signal is below confidence threshold "
                    f"({rule.get('name', 'rule')})",
                    *soft_reasons,
                ],
                alternatives=_alternatives(soft_scores, soft_probabilities, HUMAN_REVIEW),
                scores=soft_scores,
                rule_route=rule_route,
            )
        if _hard_rule_contradicts_model(rule, model_usage, skip_force=skip_force):
            return Decision(
                route=HUMAN_REVIEW,
                confidence=usage_confidence,
                reasons=[
                    "hard-rule safety override contradicts model "
                    f"({rule.get('name', 'rule')}: model {model_usage})",
                    *soft_reasons,
                ],
                alternatives=_alternatives(soft_scores, soft_probabilities, HUMAN_REVIEW),
                scores=soft_scores,
                rule_route=rule_route,
            )

    reasons = list(soft_reasons)
    if rule_route and collapse_usage_route(rule_route) != model_usage:
        reasons.insert(
            0,
            f"soft rule route would be {rule_route}; final follows model {model_usage}",
        )
    return Decision(
        route=model_usage,
        confidence=usage_confidence,
        reasons=reasons,
        alternatives=_alternatives(soft_scores, soft_probabilities, model_usage),
        scores=soft_scores,
        rule_route=rule_route,
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
                rule_route=item.decision.rule_route,
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
            rule_route=item.decision.rule_route,
        )
    return [item.decision for item in prepared]
