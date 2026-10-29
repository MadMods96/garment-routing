"""Evaluation report. Numbers written here are the only numbers the README may quote.

Label-only sections (majority baseline, gold-set kappa, router-on-expert-attributes,
attribute-only logistic regression and LightGBM) run from the parquet. Vision
metrics are added only when a checkpoint and the images are both present.
Station3 and the gold set are never used to fit a model, a temperature, an
abstention threshold, or feature importance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from garmentroute.attribute_baseline import attribute_baseline_block
from garmentroute.label_shift import score_shift_variants
from garmentroute.labels import TERNARY, USAGE_CLASSES, USAGE_LABELS, usage_class
from garmentroute.metrics import classification_metrics, ordinal_metrics, pairwise_kappa, site_label_agreement
from garmentroute.passport import passport_from_annotation
from garmentroute.router import (
    HUMAN_REVIEW,
    GarmentSignals,
    assign_batch,
    decide,
    load_router_config,
    route_probabilities,
)
from garmentroute.uncertainty import (
    apply_temperature,
    coverage_accuracy,
    expected_calibration_error,
    fit_temperature,
    select_threshold,
)
from garmentroute.util import load_yaml, write_json

ORDINAL_LEVELS = [1, 2, 3, 4, 5]


def _mode(series: pd.Series) -> Any:
    observed = series.dropna()
    if observed.empty:
        return None
    return observed.mode().iloc[0]


def _pair(train: pd.DataFrame, evaluation: pd.DataFrame, column: str) -> tuple[list[Any], list[Any], Any] | None:
    majority = _mode(train[column])
    if majority is None:
        return None
    truth = evaluation[column].dropna()
    if truth.empty:
        return None
    return truth.tolist(), [majority] * len(truth), majority


def majority_task(train: pd.DataFrame, evaluation: pd.DataFrame, column: str, labels: list[Any], ordinal: bool) -> dict | None:
    """Predict the training majority label. Fit only on ``train``."""
    paired = _pair(train, evaluation, column)
    if paired is None:
        return None
    truth, pred, majority = paired
    if ordinal:
        truth_int = [int(value) for value in truth]
        pred_int = [int(value) for value in pred]
        report = classification_metrics(truth_int, pred_int, labels)
        report.update({key: value for key, value in ordinal_metrics(truth_int, pred_int).items() if key != "n"})
    else:
        report = classification_metrics(truth, pred, labels)
    report["majority_label"] = int(majority) if ordinal else str(majority)
    return report


def majority_block(frame: pd.DataFrame) -> dict[str, Any]:
    train = frame[frame["split"] == "train"]
    tasks = {
        "usage_class": (list(USAGE_CLASSES), False),
        "usage": (list(USAGE_LABELS), False),
        "stains_3class": (list(TERNARY), False),
        "holes": (list(TERNARY), False),
        "condition": (ORDINAL_LEVELS, True),
        "pilling": (ORDINAL_LEVELS, True),
    }
    block: dict[str, Any] = {}
    for split in ("test", "cross_site"):
        evaluation = frame[frame["split"] == split]
        block[split] = {}
        for column, (labels, ordinal) in tasks.items():
            block[split][column] = majority_task(train, evaluation, column, labels, ordinal)
    has_stain = {}
    for split in ("test", "cross_site"):
        evaluation = frame[frame["split"] == split]
        paired = _pair(train, evaluation, "has_stain")
        if paired is None:
            has_stain[split] = None
            continue
        truth, pred, majority = paired
        has_stain[split] = classification_metrics(
            [bool(value) for value in truth],
            [bool(value) for value in pred],
            [False, True],
        )
        has_stain[split]["majority_label"] = bool(majority)
    block["has_stain"] = has_stain
    block["fit_on"] = "train"
    return block


def gold_block(frame: pd.DataFrame) -> dict[str, Any]:
    gold = frame[frame["split"] == "gold"]
    return {
        "n": int(len(gold)),
        "expert_vs_expert_usage": pairwise_kappa(gold, "usage"),
        "expert_vs_expert_condition": pairwise_kappa(gold, "condition", quadratic=True),
        "site_agreement": site_label_agreement(gold),
        "model_vs_expert": None,
    }


def _present(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    return value


def _signals_from_record(record: dict[str, Any], confidence: float | None = None, probs: dict | None = None) -> GarmentSignals:
    condition = _present(record.get("condition"))
    pilling = _present(record.get("pilling"))
    return GarmentSignals(
        condition=int(condition) if condition is not None else None,
        pilling=int(pilling) if pilling is not None else None,
        stains=_present(record.get("stains")),
        holes=_present(record.get("holes")),
        smell=_present(record.get("smell")),
        usage_confidence=confidence,
        usage_probs=probs,
        damageimage=_present(record.get("damageimage")),
        damageloc=_present(record.get("damageloc")),
        damage=_present(record.get("damage")),
        damage2image=_present(record.get("damage2image")),
        damage2loc=_present(record.get("damage2loc")),
        damage2=_present(record.get("damage2")),
        damage3image=_present(record.get("damage3image")),
        damage3loc=_present(record.get("damage3loc")),
        damage3=_present(record.get("damage3")),
    )


def _passport_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return passport_from_annotation(
        product_id=str(record["item_id"]),
        garment_type=_present(record.get("type")),
        fibre_composition=_present(record.get("material")),
        brand=_present(record.get("brand")),
    )


def _router_rows(frame: pd.DataFrame, config: dict[str, Any], capacity: bool) -> dict[str, Any]:
    records = [record for record in frame.to_dict("records") if _present(record.get("usage")) is not None]
    if capacity:
        batch = [
            (str(record["item_id"]), _signals_from_record(record), _passport_from_record(record)) for record in records
        ]
        decisions = assign_batch(batch, config, abstain_threshold=None)
    else:
        decisions = [
            decide(_signals_from_record(record), _passport_from_record(record), config, abstain_threshold=None)
            for record in records
        ]
    truth = [record["usage"] for record in records]
    pred = [decision.route for decision in decisions]
    report = classification_metrics(truth, pred, list(USAGE_LABELS))
    mapped_truth = []
    mapped_pred = []
    for label, route in zip(truth, pred):
        if route == HUMAN_REVIEW:
            continue
        mapped_truth.append(usage_class(label))
        mapped_pred.append(usage_class(route))
    report["usage_class_agreement"] = classification_metrics(mapped_truth, mapped_pred, list(USAGE_CLASSES))
    report["abstained"] = int(sum(route == HUMAN_REVIEW for route in pred))
    report["route_counts"] = {route: int(pred.count(route)) for route in sorted(set(pred))}
    pairs: dict[tuple[str, str], int] = {}
    examples: dict[tuple[str, str], list[dict[str, str]]] = {}
    for record, route in zip(records, pred):
        if route == record["usage"]:
            continue
        key = (str(record["usage"]), str(route))
        pairs[key] = pairs.get(key, 0) + 1
        bucket = examples.setdefault(key, [])
        if len(bucket) < 3:
            bucket.append({"item_id": str(record["item_id"]), "front_path": str(record.get("front_path"))})
    ranked = sorted(pairs.items(), key=lambda item: -item[1])[:8]
    report["common_disagreements"] = [
        {
            "expert": key[0],
            "router": key[1],
            "n": count,
            "note": f"expert {key[0]}, router {key[1]}",
            "examples": examples[key],
        }
        for key, count in ranked
    ]
    report["inputs"] = "expert_damage_labels_and_dataset_material_as_mock_passport"
    report["capacity"] = capacity
    return report


def router_block(frame: pd.DataFrame, config_path: Path) -> dict[str, Any]:
    config = load_router_config(config_path)
    block: dict[str, Any] = {
        "config": str(config_path),
        "placeholder_notice": config.get("placeholder_notice"),
        "on_expert_attributes": {},
        "on_predicted_attributes": None,
    }
    for split in ("test", "cross_site"):
        evaluation = frame[frame["split"] == split]
        block["on_expert_attributes"][split] = _router_rows(evaluation, config, capacity=False)
        block["on_expert_attributes"][f"{split}_capacity"] = _router_rows(evaluation, config, capacity=True)
    return block


def _station3_report(frame: pd.DataFrame, probability_models: dict[str, Any], router_config: Path) -> dict[str, Any]:
    """Majority and router on the same station 3 rows as the probability models.

    The prior sample has already been removed by ``station3_context``.
    """
    sample_ids = set(probability_models["prior_sample_item_ids"])
    score = frame.loc[frame["split"].eq("cross_site") & frame["usage"].notna() & ~frame["item_id"].isin(sample_ids)]
    truth = score["usage"].astype(str).tolist()
    classes = list(USAGE_LABELS)
    source_prior = probability_models["source_prior"]
    sample_counts = probability_models["target_prior_counts"]
    alpha = float(probability_models["laplace_alpha"])
    counts = probability_models["source_prior_counts"]
    majority_label = max(classes, key=lambda label: (int(counts[label]), -classes.index(label)))
    majority_matrix = np.zeros((len(score), len(classes)), dtype=float)
    majority_matrix[:, classes.index(majority_label)] = 1.0
    config = load_router_config(router_config)
    router_matrix = np.zeros((len(score), len(classes)), dtype=float)
    for row_index, record in enumerate(score.to_dict("records")):
        signals = _signals_from_record(record)
        passport = _passport_from_record(record)
        distribution = route_probabilities(signals, passport, config)
        for class_index, label in enumerate(classes):
            router_matrix[row_index, class_index] = float(distribution.get(label, 0.0))
    models = {
        "majority": score_shift_variants(truth, majority_matrix, classes, source_prior, sample_counts, alpha),
        "rule_router": score_shift_variants(truth, router_matrix, classes, source_prior, sample_counts, alpha),
    }
    for name in ("logistic_regression", "lightgbm"):
        models[name] = probability_models[name]
    kept = {
        key: probability_models[key]
        for key in (
            "prior_sample_size",
            "prior_sample_seed",
            "prior_sample_item_ids",
            "n_scored",
            "n_cross_site",
            "source_prior",
            "source_prior_counts",
            "target_prior",
            "target_prior_counts",
            "laplace_alpha",
            "target_prior_laplace",
            "method",
            "em_reference",
        )
    }
    kept["primary_metrics"] = ["balanced_accuracy", "macro_f1"]
    kept["secondary_metrics"] = ["accuracy"]
    kept["probability_source"] = {
        "majority": "one-hot on the training majority class",
        "rule_router": "softmax of the illustrative route scores",
        "logistic_regression": "predict_proba",
        "lightgbm": "predict_proba",
    }
    kept["note"] = (
        "Each corrected prediction is the argmax of p(y|x) times a station 3 class prior divided by the training class prior. "
        "The labelled sample is a simple random sample of station 3 garments and is excluded from the scored rows. "
        "Its raw class shares are reported even when a class count is zero. "
        "Laplace smoothing adds one count to every class before dividing, so a class missing from the sample stays possible. "
        "The EM prior (Saerens, Latinne, and Decaestecker, 2002) is estimated from each model's p(y|x) on the scored rows and does not use station 3 labels. "
        "If that p(y|x) already puts more mass on a rare class than the training prior, the EM estimate can concentrate on that class. "
        "The router values being reweighted are a softmax of the illustrative route scores."
    )
    kept["models"] = models
    if len(score) != int(probability_models["n_scored"]):
        kept["n_scored_mismatch"] = int(len(score))
    return kept


def data_summary(frame: pd.DataFrame) -> dict[str, Any]:
    summary = {"n_rows": int(len(frame))}
    if "split" in frame.columns:
        summary["split_counts"] = {str(key): int(value) for key, value in frame["split"].value_counts().items()}
    summary["usage_raw_counts"] = {
        str(key): int(value) for key, value in frame["usage"].fillna("missing").value_counts().items()
    }
    return summary


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _decode_predictions(outputs: dict[str, Any], item_ids: list[str]) -> list[dict[str, Any]]:
    from garmentroute.model import coral_decode

    usage_logits = outputs["usage"].detach().float().cpu().numpy()
    condition = coral_decode(outputs["condition"]).detach().cpu().tolist()
    pilling = coral_decode(outputs["pilling"]).detach().cpu().tolist()
    stains = outputs["stains"].argmax(dim=-1).detach().cpu().tolist()
    holes = outputs["holes"].argmax(dim=-1).detach().cpu().tolist()
    rows = []
    for index, item_id in enumerate(item_ids):
        rows.append(
            {
                "item_id": item_id,
                "usage_logits": usage_logits[index].tolist(),
                "condition": int(condition[index]),
                "pilling": int(pilling[index]),
                "stains": TERNARY[int(stains[index])],
                "holes": TERNARY[int(holes[index])],
            }
        )
    return rows


def predict_checkpoint(
    checkpoint_path: Path,
    frame: pd.DataFrame,
    image_root: Path,
    batch_size: int,
    mc_passes: int,
    view_log: Path | None = None,
) -> list[dict[str, Any]] | None:
    """Run a checkpoint. Returns None when no garment has a front or back image."""
    import torch
    from torch.utils.data import DataLoader

    from garmentroute.dataset import TwoViewDataset
    from garmentroute.model import GarmentNet

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GarmentNet(
        backbone=config.get("backbone", "convnext_tiny"),
        pretrained=False,
        fusion=config.get("fusion", "concat"),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    dataset = TwoViewDataset(
        frame,
        image_root,
        int(config.get("image_size", 224)),
        train=False,
        view_log=view_log,
    )
    if len(dataset) == 0:
        return None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            front = batch["front"].to(device)
            back = batch["back"].to(device)
            outputs = model(front, back)
            decoded = _decode_predictions(outputs, list(batch["item_id"]))
            if mc_passes > 0:
                model.mc_dropout_mode()
                stack = []
                for _ in range(mc_passes):
                    mc_outputs = model(front, back)
                    stack.append(_softmax(mc_outputs["usage"].detach().float().cpu().numpy()))
                model.eval()
                mean = np.mean(stack, axis=0)
                entropy = -np.sum(mean * np.log(np.clip(mean, 1e-8, 1)), axis=1)
                for index, row in enumerate(decoded):
                    row["usage_entropy"] = float(entropy[index])
            rows.extend(decoded)
    return rows


def _head_report(truth: list[Any], pred: list[Any], labels: list[Any], ordinal: bool) -> dict[str, Any]:
    if ordinal:
        report = classification_metrics([int(value) for value in truth], [int(value) for value in pred], labels)
        report.update(
            {
                key: value
                for key, value in ordinal_metrics([int(value) for value in truth], [int(value) for value in pred]).items()
                if key != "n"
            }
        )
        return report
    return classification_metrics(truth, pred, labels)


def score_predictions(frame: pd.DataFrame, predictions: list[dict[str, Any]], temperature: float) -> dict[str, Any]:
    """Score stored logits. ``temperature`` is applied to usage logits only."""
    by_id = {row["item_id"]: row for row in predictions}
    aligned = frame[frame["item_id"].isin(by_id)].copy()
    report: dict[str, Any] = {"n_scored": int(len(aligned)), "temperature": temperature}
    heads = {
        "condition": ("condition", ORDINAL_LEVELS, True),
        "pilling": ("pilling", ORDINAL_LEVELS, True),
        "stains": ("stains_3class", list(TERNARY), False),
        "holes": ("holes", list(TERNARY), False),
    }
    for name, (column, labels, ordinal) in heads.items():
        prediction_key = "stains" if name == "stains" else name
        truth = []
        pred = []
        for record in aligned.to_dict("records"):
            value = _present(record.get(column))
            if value is None:
                continue
            truth.append(value)
            pred.append(by_id[record["item_id"]][prediction_key])
        report[prediction_key] = _head_report(truth, pred, labels, ordinal) if truth else None
    usage_truth = []
    usage_pred = []
    confidences = []
    correct = []
    logits = []
    labels = []
    mistakes = []
    for record in aligned.to_dict("records"):
        label = _present(record.get("usage_class"))
        if label is None:
            continue
        row = by_id[record["item_id"]]
        scaled = apply_temperature(np.asarray(row["usage_logits"], dtype=float)[None, :], temperature)[0]
        probabilities = _softmax(scaled[None, :])[0]
        index = int(np.argmax(probabilities))
        predicted = USAGE_CLASSES[index]
        confidence = float(probabilities[index])
        usage_truth.append(label)
        usage_pred.append(predicted)
        confidences.append(confidence)
        correct.append(predicted == label)
        logits.append(row["usage_logits"])
        labels.append(USAGE_CLASSES.index(label))
        if predicted != label:
            mistakes.append(
                {
                    "item_id": record["item_id"],
                    "front_path": record.get("front_path"),
                    "back_path": record.get("back_path"),
                    "predicted": predicted,
                    "expert": label,
                    "confidence": confidence,
                    "note": (
                        f"predicted {predicted} ({confidence:.2f}), expert {label}; "
                        f"condition pred {row['condition']} vs label {record.get('condition')}"
                    ),
                }
            )
    report["usage"] = classification_metrics(usage_truth, usage_pred, list(USAGE_CLASSES)) if usage_truth else None
    if usage_truth and len(set(usage_truth)) > 1 and len(set(usage_pred)) > 1:
        from sklearn.metrics import cohen_kappa_score

        report["usage_cohen_kappa"] = float(cohen_kappa_score(usage_truth, usage_pred))
    else:
        report["usage_cohen_kappa"] = None
    report["calibration"] = expected_calibration_error(confidences, correct)
    report["coverage_accuracy"] = coverage_accuracy(confidences, correct)
    mistakes.sort(key=lambda row: -row["confidence"])
    report["worst_mistakes"] = mistakes[:12]
    report["_val_logits"] = logits
    report["_val_labels"] = labels
    report["_confidences"] = confidences
    report["_correct"] = correct
    return report


def router_from_predictions(
    frame: pd.DataFrame,
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
    temperature: float,
    abstain_threshold: float | None,
) -> dict[str, Any]:
    """Route from predicted damage attributes. Fibre still comes from the passport."""
    by_id = {row["item_id"]: row for row in predictions}
    truth: list[str] = []
    pred: list[str] = []
    for record in frame.to_dict("records"):
        row = by_id.get(record["item_id"])
        expert = _present(record.get("usage"))
        if row is None or expert is None:
            continue
        scaled = apply_temperature(np.asarray(row["usage_logits"], dtype=float)[None, :], temperature)[0]
        probabilities = _softmax(scaled[None, :])[0]
        overridden = dict(record)
        overridden["condition"] = row["condition"]
        overridden["pilling"] = row["pilling"]
        overridden["stains"] = row["stains"]
        overridden["holes"] = row["holes"]
        signals = _signals_from_record(
            overridden,
            confidence=float(probabilities.max()),
            probs={USAGE_CLASSES[index]: float(probabilities[index]) for index in range(len(USAGE_CLASSES))},
        )
        decision = decide(signals, _passport_from_record(record), config, abstain_threshold=abstain_threshold)
        truth.append(str(expert))
        pred.append(decision.route)
    report = classification_metrics(truth, pred, list(USAGE_LABELS) + [HUMAN_REVIEW])
    mapped_truth = []
    mapped_pred = []
    for label, route in zip(truth, pred):
        if route == HUMAN_REVIEW:
            continue
        mapped = usage_class(route)
        if mapped is None:
            continue
        mapped_truth.append(usage_class(label))
        mapped_pred.append(mapped)
    report["usage_class_agreement"] = classification_metrics(mapped_truth, mapped_pred, list(USAGE_CLASSES))
    report["abstained"] = int(sum(route == HUMAN_REVIEW for route in pred))
    report["inputs"] = "predicted_damage_attributes_and_dataset_material_as_mock_passport"
    return report


def model_block(
    frame: pd.DataFrame,
    checkpoint: Path,
    image_root: Path,
    batch_size: int,
    mc_passes: int,
    min_coverage: float,
    router_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit temperature and the abstention threshold on val, then score test and cross-site."""
    val = frame[frame["split"] == "val"]
    val_predictions = predict_checkpoint(checkpoint, val, image_root, batch_size, mc_passes=0)
    if not val_predictions:
        return {"status": "checkpoint present but val images are not on disk", "checkpoint": str(checkpoint)}
    val_scored = score_predictions(val, val_predictions, temperature=1.0)
    if not val_scored["_val_labels"]:
        return {"status": "val split has no scored usage labels", "checkpoint": str(checkpoint)}
    temperature = fit_temperature(np.asarray(val_scored["_val_logits"], dtype=float), val_scored["_val_labels"])
    val_calibrated = score_predictions(val, val_predictions, temperature=temperature)
    threshold = select_threshold(val_calibrated["_confidences"], val_calibrated["_correct"], min_coverage)
    block: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "temperature": temperature,
        "ece_val_before": val_scored["calibration"],
        "ece_val_after": val_calibrated["calibration"],
        "abstain_threshold_fit_on": "val",
        "abstain_threshold": threshold,
        "splits": {},
    }
    for split in ("test", "cross_site", "gold"):
        predictions = predict_checkpoint(
            checkpoint, frame[frame["split"] == split], image_root, batch_size, mc_passes
        )
        if not predictions:
            block["splits"][split] = None
            continue
        before = score_predictions(frame[frame["split"] == split], predictions, temperature=1.0)
        after = score_predictions(frame[frame["split"] == split], predictions, temperature=temperature)
        public_before = {key: value for key, value in before.items() if not key.startswith("_")}
        public_after = {key: value for key, value in after.items() if not key.startswith("_")}
        confidences = after["_confidences"]
        correct = after["_correct"]
        cutoff = threshold.get("threshold")
        if cutoff is None:
            selective = None
        else:
            mask = np.asarray(confidences) >= cutoff
            selective = {
                "threshold": cutoff,
                "coverage": float(mask.mean()) if len(mask) else None,
                "accuracy": float(np.mean(np.asarray(correct)[mask])) if mask.any() else None,
                "n": int(mask.sum()),
            }
        entropies = [row["usage_entropy"] for row in predictions if "usage_entropy" in row]
        router_view = None
        if router_config is not None and split in {"test", "cross_site"}:
            router_view = router_from_predictions(
                frame[frame["split"] == split],
                predictions,
                router_config,
                temperature,
                threshold.get("threshold"),
            )
        block["splits"][split] = {
            "before_temperature": public_before,
            "after_temperature": public_after,
            "selective_at_val_threshold": selective,
            "mc_dropout_passes": mc_passes,
            "mean_usage_entropy": float(np.mean(entropies)) if entropies else None,
            "router_on_predicted_attributes": router_view,
        }
    return block


def evaluate(
    index_path: Path,
    router_config: Path,
    output_path: Path,
    checkpoint: Path | None = None,
    image_root: Path | None = None,
    attribute_config: Path | None = None,
    batch_size: int = 32,
    mc_passes: int = 0,
    min_coverage: float = 0.5,
    smoke: bool = False,
) -> dict[str, Any]:
    """Write ``results.json``. Vision sections stay null until a real checkpoint is scored."""
    frame = pd.read_parquet(index_path)
    meta_path = index_path.parent / "split_meta.json"
    meta = load_yaml(meta_path) if meta_path.suffix in {".yaml", ".yml"} and meta_path.exists() else None
    if meta_path.exists() and meta is None:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "smoke" if smoke else "clothing_v3",
        "doi": None if smoke else "10.5281/zenodo.13788681",
        "smoke": smoke,
        "seed": None if meta is None else meta.get("seed"),
        "split_meta": meta,
        "data": data_summary(frame),
        "majority": majority_block(frame),
        "gold": gold_block(frame),
        "router": router_block(frame, router_config),
        "attribute_baselines": attribute_baseline_block(
            frame,
            42 if meta is None or meta.get("seed") is None else int(meta["seed"]),
            load_yaml(Path("configs/attribute_baseline.yaml") if attribute_config is None else attribute_config),
        ),
        "front": None,
        "fusion": None,
        "calibration_notes": "Temperature and the abstention threshold are fit on val only.",
    }
    probability_models = payload["attribute_baselines"].pop("station3_probability_models", None)
    payload["station3"] = (
        None
        if not probability_models
        else _station3_report(frame, probability_models, router_config)
    )
    if checkpoint is not None and image_root is not None:
        scored = model_block(
            frame,
            checkpoint,
            image_root,
            batch_size,
            mc_passes,
            min_coverage,
            router_config=load_router_config(router_config),
        )
        fusion = load_checkpoint_fusion(checkpoint)
        key = "front" if fusion == "front" else "fusion"
        payload[key] = scored
        gold_split = (scored.get("splits") or {}).get("gold") or {}
        after = gold_split.get("after_temperature") or {}
        if after:
            payload["gold"]["model_vs_expert"] = {
                "usage_cohen_kappa": after.get("usage_cohen_kappa"),
                "usage": after.get("usage"),
                "condition": after.get("condition"),
                "note": (
                    "Model-versus-expert kappa compares each gold annotation with the "
                    "model on that annotation's photographs. Expert-versus-expert kappa "
                    "is pairwise between annotators of the same gold id."
                ),
            }
        predicted = {}
        for split in ("test", "cross_site"):
            split_payload = (scored.get("splits") or {}).get(split) or {}
            predicted[split] = split_payload.get("router_on_predicted_attributes")
        if any(value is not None for value in predicted.values()):
            payload["router"]["on_predicted_attributes"] = predicted
    from garmentroute.eda import write_metric_figures

    payload["metric_figures"] = write_metric_figures(payload, output_path.parent / "figures")
    write_json(output_path, _public(payload))
    return payload


def load_checkpoint_fusion(checkpoint: Path) -> str:
    import torch

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return str(loaded.get("config", {}).get("fusion", "concat"))


def _public(value: Any) -> Any:
    """Drop private scoring buffers before the report is written."""
    if isinstance(value, dict):
        return {str(key): _public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, tuple):
        return [_public(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/index.parquet"))
    parser.add_argument("--router-config", type=Path, default=Path("configs/router.yaml"))
    parser.add_argument("--out", type=Path, default=Path("reports/results.json"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=Path("data/resized"))
    parser.add_argument("--attribute-config", type=Path, default=Path("configs/attribute_baseline.yaml"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--mc-passes", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    evaluate(
        args.index,
        args.router_config,
        args.out,
        checkpoint=args.checkpoint,
        image_root=args.image_root,
        attribute_config=args.attribute_config,
        batch_size=args.batch_size,
        mc_passes=args.mc_passes,
        min_coverage=args.min_coverage,
        smoke=args.smoke,
    )
    print(args.out)


if __name__ == "__main__":
    main()
