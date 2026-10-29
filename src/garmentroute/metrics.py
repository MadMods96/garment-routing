"""Metrics used by training and by the evaluation report.

Every figure written into ``reports/results.json`` should come from these
functions or from an equally explicit calculation in ``evaluate``. Nothing in
the README is typed in by hand.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from garmentroute.labels import USAGE_LABELS


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro-F1, and a confusion matrix with a fixed label order.

    Balanced accuracy is the mean recall over classes that occur in ``y_true``.
    A class with no scored rows is left out of that mean.
    """
    truth = list(y_true)
    pred = list(y_pred)
    label_list = list(labels)
    if not truth:
        return {
            "n": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "confusion_matrix": {"labels": label_list, "matrix": []},
        }
    matrix = confusion_matrix(truth, pred, labels=label_list)
    support = matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(
            np.diag(matrix).astype(float),
            support,
            out=np.full(len(support), np.nan, dtype=float),
            where=support > 0,
        )
    present = recall[~np.isnan(recall)]
    return {
        "n": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(present.mean()) if present.size else None,
        "macro_f1": float(f1_score(truth, pred, labels=label_list, average="macro", zero_division=0)),
        "confusion_matrix": {"labels": [str(label) for label in label_list], "matrix": matrix.tolist()},
    }


def ordinal_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    """Exact accuracy, mean absolute error, and accuracy within one level."""
    truth = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    if truth.size == 0:
        return {"n": 0, "accuracy": None, "mae": None, "within_1_accuracy": None}
    error = np.abs(truth - pred)
    return {
        "n": int(truth.size),
        "accuracy": float(np.mean(error == 0)),
        "mae": float(np.mean(error)),
        "within_1_accuracy": float(np.mean(error <= 1)),
    }


def pairwise_kappa(frame: Any, column: str, quadratic: bool = False) -> dict[str, Any]:
    """Cohen's kappa from every unordered pair of annotations that share ``gold_id``."""
    grouped_left: list[Any] = []
    grouped_right: list[Any] = []
    items = 0
    items_with_pairs = 0
    if "gold_id" not in frame.columns or column not in frame.columns:
        return {"n_items": 0, "n_items_with_pairs": 0, "n_pairs": 0, "raw_agreement": None, "cohen_kappa": None}
    work = frame[frame["gold_id"].notna() & frame["gold_id"].astype(str).str.len().gt(0)]
    for _gold_id, group in work.groupby("gold_id", dropna=True):
        values = []
        for value in group[column].tolist():
            if value is None:
                continue
            if isinstance(value, float) and value != value:
                continue
            values.append(value)
        items += 1
        if len(values) < 2:
            continue
        items_with_pairs += 1
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                grouped_left.append(values[left_index])
                grouped_right.append(values[right_index])
    result: dict[str, Any] = {
        "n_items": int(items),
        "n_items_with_pairs": int(items_with_pairs),
        "n_pairs": int(len(grouped_left)),
        "raw_agreement": None,
        "cohen_kappa": None,
    }
    if not grouped_left:
        return result
    result["raw_agreement"] = float(np.mean([left == right for left, right in zip(grouped_left, grouped_right)]))
    result["cohen_kappa"] = float(cohen_kappa_score(grouped_left, grouped_right))
    if quadratic:
        result["cohen_kappa_quadratic"] = float(
            cohen_kappa_score(grouped_left, grouped_right, weights="quadratic")
        )
    return result


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and value != value:
        return True
    return bool(pd.isna(value)) if not isinstance(value, (str, bytes)) else False


def _annotation_site(station: Any) -> str | None:
    text = str(station)
    if text in {"station1", "station2"}:
        return "wargon"
    if text == "station3":
        return "myrorna"
    return None


def _finite_kappa(left: Sequence[Any], right: Sequence[Any], weights: str | None = None) -> float | None:
    if not left:
        return None
    score = cohen_kappa_score(list(left), list(right), weights=weights)
    if score != score:
        return None
    return float(score)


def _pair_summary(left: Sequence[Any], right: Sequence[Any], quadratic: bool = False) -> dict[str, Any]:
    """Agreement of aligned annotation pairs. ``left`` and ``right`` are the two sites or two raters."""
    pairs = [(left_value, right_value) for left_value, right_value in zip(left, right)]
    summary: dict[str, Any] = {
        "n_pairs": int(len(pairs)),
        "raw_agreement": None,
        "cohen_kappa": None,
    }
    if not pairs:
        if quadratic:
            summary["cohen_kappa_quadratic"] = None
        return summary
    summary["raw_agreement"] = float(np.mean([left_value == right_value for left_value, right_value in pairs]))
    summary["cohen_kappa"] = _finite_kappa(left, right)
    if quadratic:
        summary["cohen_kappa_quadratic"] = _finite_kappa(left, right, weights="quadratic")
    return summary


def site_label_agreement(frame: Any) -> dict[str, Any]:
    """Compare Wargön and Myrorna labels on gold garments annotated at both sites.

    Station 1 and station 2 are Wargön. Station 3 is Myrorna. A cross-site pair
    is one Wargön annotation joined to one Myrorna annotation of the same
    ``gold_id``. A within-Wargön pair is an unordered pair of Wargön annotations
    on one of those same garments. Garments labelled at only one site are left out.
    """
    empty_usage = _pair_summary([], [])
    empty_condition = _pair_summary([], [], quadratic=True)
    report: dict[str, Any] = {
        "n_garments": 0,
        "definition": (
            "Wargön is station 1 and station 2. Myrorna is station 3. "
            "Cross-site pairs join every Wargön annotation to every Myrorna annotation of the same gold id. "
            "Within-Wargön pairs are unordered pairs of Wargön annotations on those garments. "
            "Same view means the condition scores are equal. Same route means the usage labels are equal."
        ),
        "cross_site": {"usage": empty_usage, "condition": empty_condition},
        "within_wargon": {"usage": empty_usage, "condition": empty_condition},
        "view_versus_route": {
            "n_pairs": 0,
            "n_garments": 0,
            "same_view_same_route": 0,
            "same_view_different_route": 0,
            "different_view_same_route": 0,
            "different_view_different_route": 0,
        },
    }
    required = {"gold_id", "station", "usage", "condition"}
    if not required.issubset(getattr(frame, "columns", [])):
        return report
    work = frame[frame["gold_id"].notna() & frame["gold_id"].astype(str).str.len().gt(0)]
    if "item_id" in work.columns:
        work = work.sort_values(["gold_id", "item_id"])
    cross_usage_left: list[str] = []
    cross_usage_right: list[str] = []
    cross_condition_left: list[int] = []
    cross_condition_right: list[int] = []
    within_usage_left: list[str] = []
    within_usage_right: list[str] = []
    within_condition_left: list[int] = []
    within_condition_right: list[int] = []
    view_counts = {key: 0 for key in report["view_versus_route"] if key not in {"n_pairs", "n_garments"}}
    view_garments = 0
    for _gold_id, group in work.groupby("gold_id", dropna=True, sort=False):
        wargon = []
        myrorna = []
        for record in group.to_dict("records"):
            site = _annotation_site(record.get("station"))
            if site == "wargon":
                wargon.append(record)
            elif site == "myrorna":
                myrorna.append(record)
        if not wargon or not myrorna:
            continue
        report["n_garments"] += 1
        for left, right in combinations(wargon, 2):
            left_usage = left.get("usage")
            right_usage = right.get("usage")
            if not _missing(left_usage) and not _missing(right_usage):
                within_usage_left.append(str(left_usage))
                within_usage_right.append(str(right_usage))
            left_condition = left.get("condition")
            right_condition = right.get("condition")
            if not _missing(left_condition) and not _missing(right_condition):
                within_condition_left.append(int(left_condition))
                within_condition_right.append(int(right_condition))
        garment_view = False
        for left in wargon:
            for right in myrorna:
                left_usage = left.get("usage")
                right_usage = right.get("usage")
                if not _missing(left_usage) and not _missing(right_usage):
                    cross_usage_left.append(str(left_usage))
                    cross_usage_right.append(str(right_usage))
                left_condition = left.get("condition")
                right_condition = right.get("condition")
                if not _missing(left_condition) and not _missing(right_condition):
                    cross_condition_left.append(int(left_condition))
                    cross_condition_right.append(int(right_condition))
                if (
                    _missing(left_usage)
                    or _missing(right_usage)
                    or _missing(left_condition)
                    or _missing(right_condition)
                ):
                    continue
                same_view = int(left_condition) == int(right_condition)
                same_route = str(left_usage) == str(right_usage)
                if same_view and same_route:
                    view_counts["same_view_same_route"] += 1
                elif same_view:
                    view_counts["same_view_different_route"] += 1
                elif same_route:
                    view_counts["different_view_same_route"] += 1
                else:
                    view_counts["different_view_different_route"] += 1
                garment_view = True
        if garment_view:
            view_garments += 1
    usage = _pair_summary(cross_usage_left, cross_usage_right)
    usage["confusion_matrix"] = {
        "rows": "wargon",
        "columns": "myrorna",
        "labels": list(USAGE_LABELS),
        "matrix": confusion_matrix(cross_usage_left, cross_usage_right, labels=list(USAGE_LABELS)).tolist()
        if cross_usage_left
        else [],
    }
    condition = _pair_summary(cross_condition_left, cross_condition_right, quadratic=True)
    if cross_condition_left:
        difference = np.asarray(cross_condition_right, dtype=float) - np.asarray(cross_condition_left, dtype=float)
        condition["mean_difference_myrorna_minus_wargon"] = float(difference.mean())
        condition["mean_absolute_difference"] = float(np.abs(difference).mean())
    else:
        condition["mean_difference_myrorna_minus_wargon"] = None
        condition["mean_absolute_difference"] = None
    within_condition = _pair_summary(within_condition_left, within_condition_right, quadratic=True)
    if within_condition_left:
        within_gap = np.abs(
            np.asarray(within_condition_right, dtype=float) - np.asarray(within_condition_left, dtype=float)
        )
        within_condition["mean_absolute_difference"] = float(within_gap.mean())
    else:
        within_condition["mean_absolute_difference"] = None
    report["cross_site"] = {"usage": usage, "condition": condition}
    report["within_wargon"] = {
        "usage": _pair_summary(within_usage_left, within_usage_right),
        "condition": within_condition,
    }
    report["view_versus_route"] = {
        "n_pairs": int(sum(view_counts.values())),
        "n_garments": int(view_garments),
        **view_counts,
    }
    return report
