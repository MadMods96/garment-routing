"""HR reason breakdown, expert hard-rule check, selective CIs, 224 vs 384 ablation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garmentroute.evaluate import predict_checkpoint, score_predictions
from garmentroute.labels import TERNARY, USAGE_CLASSES, usage_class
from garmentroute.metrics import (
    bootstrap_ci,
    bootstrap_class_recall,
    bootstrap_macro_f1,
    classification_metrics,
    ordinal_metrics,
)
from garmentroute.model import decode_ternary
from garmentroute.passport import coerce_passport
from garmentroute.router import (
    HUMAN_REVIEW,
    GarmentSignals,
    _hard_rule_contradicts_model,
    _rule_fires,
    _trigger_confidence,
    decide,
    matching_fibres,
)
from garmentroute.uncertainty import apply_temperature
from garmentroute.util import load_yaml, write_json
from garmentroute.views import filter_to_subset, list_image_relatives


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _passport(record: dict) -> dict:
    return coerce_passport(
        {
            "fibre_composition": record.get("material"),
            "product_id": record.get("item_id"),
        }
    )


def _would_hard_rule(
    signals: GarmentSignals,
    passport: dict,
    config: dict,
) -> bool:
    fibres = matching_fibres(
        passport.get("fibre_composition") if isinstance(passport, dict) else None,
        list(config.get("high_value_fibres", [])),
    )
    hard_threshold = config.get("hard_rule_min_confidence")
    model_usage = signals.usage_class
    if not model_usage:
        return False
    for rule in config.get("hard_rules", []):
        if not _rule_fires(rule, signals):
            continue
        skip_force = bool(rule.get("skip_force_if_high_value_fibre")) and bool(fibres)
        trigger_conf = _trigger_confidence(rule, signals)
        if hard_threshold is not None and (trigger_conf is None or trigger_conf < float(hard_threshold)):
            return True  # weak damage signal still escalates to HR
        if _hard_rule_contradicts_model(rule, model_usage, skip_force=skip_force):
            return True
    return False


def _bootstrap_accuracy(
    y_true: list[str],
    y_pred: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    truth = np.asarray(y_true, dtype=object)
    pred = np.asarray(y_pred, dtype=object)
    n = len(truth)
    point = float(accuracy_score(truth, pred)) if n else None
    if n == 0:
        return {"point": None, "n_garments": 0, **bootstrap_ci([])}
    rng = np.random.default_rng(seed)
    draws = []
    idx = np.arange(n)
    for _ in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        draws.append(float(accuracy_score(truth[sample], pred[sample])))
    return {"point": point, "n_garments": n, **bootstrap_ci(draws)}


def _decode_predictions(predictions: list[dict], holes_major_threshold: float | None) -> None:
    for row in predictions:
        if "holes_logits" in row:
            logits = torch.tensor([row["holes_logits"]], dtype=torch.float32)
            row["holes"] = TERNARY[int(decode_ternary(logits, major_threshold=holes_major_threshold).item())]


def _score_model(
    name: str,
    checkpoint: Path,
    test: pd.DataFrame,
    image_root: Path,
    batch_size: int,
    temperature: float,
    holes_major_threshold: float | None,
    n_boot: int,
    seed: int,
) -> dict:
    print(f"{name} predict test n={len(test)}", flush=True)
    predictions = predict_checkpoint(checkpoint, test, image_root, batch_size, mc_passes=0) or []
    _decode_predictions(predictions, holes_major_threshold)
    scored = score_predictions(test, predictions, temperature=temperature)
    by_id = {row["item_id"]: row for row in predictions}

    usage_truth, usage_pred = [], []
    holes_truth, holes_pred = [], []
    stains_truth, stains_pred = [], []
    cond_truth, cond_pred = [], []
    for record in test.to_dict("records"):
        row = by_id.get(record["item_id"])
        if row is None:
            continue
        expert_u = usage_class(str(record.get("usage"))) if isinstance(record.get("usage"), str) else None
        if expert_u is not None:
            scaled = apply_temperature(np.asarray(row["usage_logits"], dtype=float)[None, :], temperature)[0]
            probs = _softmax(scaled[None, :])[0]
            usage_truth.append(expert_u)
            usage_pred.append(USAGE_CLASSES[int(np.argmax(probs))])
        if isinstance(record.get("holes"), str):
            holes_truth.append(record["holes"])
            holes_pred.append(row["holes"])
        if isinstance(record.get("stains"), str) or isinstance(record.get("stains_3class"), str):
            label = record.get("stains") if isinstance(record.get("stains"), str) else record.get("stains_3class")
            if isinstance(label, str):
                stains_truth.append(label)
                stains_pred.append(row["stains"])
        if record.get("condition") is not None and not (isinstance(record.get("condition"), float) and np.isnan(record.get("condition"))):
            try:
                cond_truth.append(int(record["condition"]))
                cond_pred.append(int(row["condition"]))
            except (TypeError, ValueError):
                pass

    usage_f1 = bootstrap_macro_f1(usage_truth, usage_pred, USAGE_CLASSES, n_boot=n_boot, seed=seed)
    holes_major = bootstrap_class_recall(holes_truth, holes_pred, "Major", n_boot=n_boot, seed=seed)
    stains_report = classification_metrics(stains_truth, stains_pred, list(TERNARY))
    condition = ordinal_metrics(cond_truth, cond_pred)
    if cond_truth:
        condition["cohen_kappa_quadratic"] = float(
            cohen_kappa_score(cond_truth, cond_pred, weights="quadratic")
        )
    else:
        condition["cohen_kappa_quadratic"] = None

    print(
        f"{name} usage macro-F1={usage_f1['point']:.4f} "
        f"CI=[{usage_f1['ci_low']:.4f},{usage_f1['ci_high']:.4f}] n={usage_f1['n_garments']}",
        flush=True,
    )
    print(
        f"{name} holes Major recall={holes_major['point']} "
        f"CI=[{holes_major.get('ci_low')},{holes_major.get('ci_high')}] n={holes_major['n_support']}",
        flush=True,
    )
    print(
        f"{name} stains macro-F1={stains_report.get('macro_f1')} "
        f"condition κ_q={condition.get('cohen_kappa_quadratic')}",
        flush=True,
    )
    return {
        "checkpoint": str(checkpoint),
        "n_predictions": len(predictions),
        "usage_macro_f1_bootstrap": usage_f1,
        "holes_major_recall_bootstrap": holes_major,
        "stains": {
            "n": stains_report.get("n"),
            "macro_f1": stains_report.get("macro_f1"),
            "balanced_accuracy": stains_report.get("balanced_accuracy"),
            "per_class_recall": stains_report.get("per_class_recall"),
            "per_class_support": stains_report.get("per_class_support"),
        },
        "condition": {
            "n": condition.get("n"),
            "mae": condition.get("mae"),
            "within_1_accuracy": condition.get("within_1_accuracy"),
            "cohen_kappa_quadratic": condition.get("cohen_kappa_quadratic"),
        },
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.parquet")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "resized")
    parser.add_argument("--members", type=Path, default=ROOT / "data" / "subset_members_16200.txt")
    parser.add_argument("--router", type=Path, default=ROOT / "configs" / "router.yaml")
    parser.add_argument("--results", type=Path, default=ROOT / "reports" / "results.json")
    parser.add_argument("--ckpt-224", type=Path, default=ROOT / "runs" / "convnext_tiny_front" / "best.pt")
    parser.add_argument(
        "--ckpt-384", type=Path, default=ROOT / "runs" / "convnext_tiny_front_384_10k" / "best.pt"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.is_file() else {}
    front = results.get("front") or {}
    thr_block = front.get("abstain_threshold") or {}
    abstain = float(thr_block.get("threshold") if isinstance(thr_block, dict) else thr_block or 0.493864)
    temperature_384 = float(front.get("temperature") or 1.0)
    major_thr = (front.get("holes_major_threshold") or {}).get("threshold")
    if major_thr is not None:
        major_thr = float(major_thr)
    # 224 used its own fitted temperature from earlier front block if present on ckpt.
    ckpt224 = torch.load(args.ckpt_224, map_location="cpu", weights_only=False)
    temperature_224 = float(ckpt224.get("temperature") or front.get("temperature") or 1.0)

    frame = pd.read_parquet(args.index)
    frame = filter_to_subset(frame, args.members)
    available = list_image_relatives(args.image_root)
    front_path = frame["front_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    back_path = frame["back_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    frame = frame.loc[front_path.isin(available) | back_path.isin(available)].copy()
    test = frame[frame["split"] == "test"].copy()
    print(f"expanded test with images n={len(test)}", flush=True)

    router_config = load_yaml(args.router)
    ablation = {}
    ablation["front_224"] = _score_model(
        "front_224",
        args.ckpt_224,
        test,
        args.image_root,
        args.batch_size,
        temperature_224,
        None,
        args.n_boot,
        args.seed,
    )
    ablation["front_384_10k"] = _score_model(
        "front_384_10k",
        args.ckpt_384,
        test,
        args.image_root,
        args.batch_size,
        temperature_384,
        major_thr,
        args.n_boot,
        args.seed,
    )

    predictions = ablation["front_384_10k"].pop("predictions")
    ablation["front_224"].pop("predictions", None)
    by_id = {row["item_id"]: row for row in predictions}

    low_only = hard_only = both = neither_hr = 0
    hr_total = 0
    decided_truth: list[str] = []
    decided_pred: list[str] = []
    hard_rule_hr_ids: list[str] = []
    rows_out = []

    for record in test.to_dict("records"):
        row = by_id.get(record["item_id"])
        expert = record.get("usage")
        expert_c = usage_class(str(expert)) if isinstance(expert, str) else None
        if row is None or expert_c is None:
            continue
        scaled = apply_temperature(
            np.asarray(row["usage_logits"], dtype=float)[None, :], temperature_384
        )[0]
        probs = _softmax(scaled[None, :])[0]
        usage_index = int(np.argmax(probs))
        model_class = USAGE_CLASSES[usage_index]
        confidence = float(probs[usage_index])
        holes_probs = _softmax(np.asarray(row.get("holes_logits") or [0, 0, 0], dtype=float)[None, :])[0]
        stains_probs = _softmax(np.asarray(row.get("stains_logits") or [0, 0, 0], dtype=float)[None, :])[0]
        holes_label = row["holes"]
        stains_label = row["stains"]
        holes_conf = float(holes_probs[TERNARY.index(holes_label)]) if holes_label in TERNARY else None
        stains_conf = float(stains_probs[TERNARY.index(stains_label)]) if stains_label in TERNARY else None
        signals = GarmentSignals(
            condition=int(row["condition"]) if row.get("condition") is not None else None,
            pilling=int(row["pilling"]) if row.get("pilling") is not None else None,
            stains=stains_label,
            holes=holes_label,
            usage_confidence=confidence,
            usage_probs={USAGE_CLASSES[i]: float(probs[i]) for i in range(len(USAGE_CLASSES))},
            usage_class=model_class,
            holes_confidence=holes_conf,
            stains_confidence=stains_conf,
        )
        passport = _passport(record)
        decision = decide(signals, passport, router_config, abstain_threshold=abstain)
        low = confidence < abstain
        hard = _would_hard_rule(signals, passport, router_config)
        if decision.route == HUMAN_REVIEW:
            hr_total += 1
            if low and hard:
                both += 1
                bucket = "both"
            elif low:
                low_only += 1
                bucket = "low_confidence_only"
            elif hard:
                hard_only += 1
                bucket = "hard_rule_only"
            else:
                neither_hr += 1
                bucket = "other_hr"
            if hard:
                hard_rule_hr_ids.append(str(record["item_id"]))
        else:
            bucket = "decided"
            decided_truth.append(expert_c)
            decided_pred.append(usage_class(decision.route) or decision.route)
        rows_out.append(
            {
                "item_id": record["item_id"],
                "bucket": bucket,
                "final_route": decision.route,
                "model_usage": model_class,
                "confidence": confidence,
                "pred_holes": holes_label,
                "pred_stains": stains_label,
                "pred_condition": row.get("condition"),
                "expert_usage": expert,
                "expert_usage_class": expert_c,
                "expert_holes": record.get("holes"),
                "expert_stains": record.get("stains") or record.get("stains_3class"),
            }
        )

    n = low_only + hard_only + both + neither_hr + len(decided_truth)
    # n should match scored garments; recount from rows_out
    n = len(rows_out)
    hr_n = low_only + hard_only + both + neither_hr
    breakdown = {
        "n": n,
        "human_review_n": hr_n,
        "human_review_share": float(hr_n / n) if n else None,
        "low_confidence_only": {"n": low_only, "share_of_test": low_only / n if n else None, "share_of_hr": low_only / hr_n if hr_n else None},
        "hard_rule_only": {"n": hard_only, "share_of_test": hard_only / n if n else None, "share_of_hr": hard_only / hr_n if hr_n else None},
        "both": {"n": both, "share_of_test": both / n if n else None, "share_of_hr": both / hr_n if hr_n else None},
        "other_hr": {"n": neither_hr, "share_of_test": neither_hr / n if n else None},
        "abstain_threshold": abstain,
        "temperature": temperature_384,
    }
    print("HR breakdown", json.dumps(breakdown, indent=2), flush=True)

    # Expert labels for Major holes / Major stains on test
    holes_major = test["holes"].fillna("").astype(str).eq("Major")
    if "stains_3class" in test.columns:
        stains_series = test["stains_3class"]
    else:
        stains_series = test["stains"]
    stains_major = stains_series.fillna("").astype(str).eq("Major")
    expert_holes_major = test.loc[holes_major]
    expert_stains_major = test.loc[stains_major]
    expert_either = test.loc[holes_major | stains_major]

    def usage_counts(sub: pd.DataFrame) -> dict:
        raw = Counter(str(v) for v in sub["usage"].dropna().tolist())
        collapsed = Counter()
        for key, value in raw.items():
            collapsed[usage_class(key) or key] += value
        return {
            "n": int(len(sub)),
            "usage_raw": dict(raw),
            "usage_class": dict(collapsed),
            "reuse_export_share": float(
                sum(collapsed.get(k, 0) for k in ("Reuse", "Export")) / len(sub)
            )
            if len(sub)
            else None,
        }

    expert_check = {
        "expert_major_holes": usage_counts(expert_holes_major),
        "expert_major_stains": usage_counts(expert_stains_major),
        "expert_major_holes_or_stains": usage_counts(expert_either),
    }
    # garments hard-rule sent to review: expert Reuse/Export share
    hard_hr_frame = test[test["item_id"].astype(str).isin(hard_rule_hr_ids)]
    expert_check["hard_rule_sent_to_hr"] = {
        **usage_counts(hard_hr_frame),
        "n_hard_rule_hr": len(hard_rule_hr_ids),
    }
    print("expert hard-rule check", json.dumps(expert_check, indent=2), flush=True)

    selective_acc = _bootstrap_accuracy(decided_truth, decided_pred, args.n_boot, args.seed)
    selective_f1 = bootstrap_macro_f1(
        decided_truth, decided_pred, USAGE_CLASSES, n_boot=args.n_boot, seed=args.seed
    )
    selective = {
        "n_decided": len(decided_truth),
        "coverage": float(len(decided_truth) / n) if n else None,
        "accuracy_bootstrap": selective_acc,
        "macro_f1_bootstrap": selective_f1,
    }
    print(
        f"selective n={selective['n_decided']} coverage={selective['coverage']} "
        f"acc={selective_acc['point']:.4f} CI=[{selective_acc['ci_low']:.4f},{selective_acc['ci_high']:.4f}] "
        f"macroF1={selective_f1['point']:.4f} CI=[{selective_f1['ci_low']:.4f},{selective_f1['ci_high']:.4f}]",
        flush=True,
    )

    recommendation = None
    stains_reuse_export = (expert_check["expert_major_stains"].get("reuse_export_share") or 0.0)
    holes_reuse_export = (expert_check["expert_major_holes"].get("reuse_export_share") or 0.0)
    if stains_reuse_export >= 0.35 and stains_reuse_export > holes_reuse_export + 0.05:
        recommendation = (
            "Experts often still route Major-stain garments to Reuse/Export; "
            "restrict hard rule to Major holes only, or Major stains with condition ≤ 2."
        )
    print("recommendation", recommendation, flush=True)

    block = {
        "members": str(args.members),
        "human_review_breakdown": breakdown,
        "expert_hard_rule_check": expert_check,
        "hard_rule_recommendation": recommendation,
        "selective_decided": selective,
        "ablation_expanded_test": ablation,
    }
    results["hr_breakdown_and_ablation"] = block
    write_json(args.results, results)
    print(f"wrote {args.results}", flush=True)

    # Compact ablation table to stdout
    print("\nABLATION (expanded test)", flush=True)
    for name in ("front_224", "front_384_10k"):
        row = ablation[name]
        uf = row["usage_macro_f1_bootstrap"]
        hm = row["holes_major_recall_bootstrap"]
        print(
            f"{name}: usage_macroF1={uf['point']:.3f} [{uf['ci_low']:.3f},{uf['ci_high']:.3f}] "
            f"holesMajor={hm['point']} n={hm['n_support']} [{hm.get('ci_low')},{hm.get('ci_high')}] "
            f"stains_macroF1={row['stains'].get('macro_f1')} "
            f"cond_kappa_q={row['condition'].get('cohen_kappa_quadratic')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
