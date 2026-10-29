"""Label-shift correction for a change in the usage prior.

The class-conditional features are left as the model learned them. Only the
class prior changes. Two estimates of the station 3 prior are reported. A
Laplace-smoothed labelled sample, held out of the scored rows, and an EM
estimate that uses the model's probabilities and no station 3 labels
(Saerens, Latinne, and Decaestecker, Neural Computation, 2002).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from garmentroute.labels import USAGE_CLASSES, USAGE_LABELS, usage_class
from garmentroute.metrics import classification_metrics


def empirical_prior(labels: Sequence[Any], classes: Sequence[str]) -> tuple[dict[str, float], dict[str, int]]:
    """Class shares over ``classes``. A class that does not occur has share 0."""
    counts = {str(label): 0 for label in classes}
    total = 0
    for label in labels:
        if label is None or (isinstance(label, float) and label != label):
            continue
        key = str(label)
        if key not in counts:
            continue
        counts[key] += 1
        total += 1
    if total == 0:
        return {key: 0.0 for key in counts}, counts
    return {key: counts[key] / total for key in counts}, counts


def laplace_prior(counts: dict[str, int], alpha: float = 1.0) -> dict[str, float]:
    """Add ``alpha`` to every class count, then divide by the new total."""
    if alpha < 0:
        raise ValueError("laplace alpha must be non-negative")
    total = sum(int(count) for count in counts.values()) + alpha * len(counts)
    if total <= 0:
        share = 1.0 / len(counts) if counts else 0.0
        return {key: share for key in counts}
    return {key: (int(count) + alpha) / total for key, count in counts.items()}


def saerens_prior(
    proba: np.ndarray,
    classes: Sequence[str],
    source_prior: dict[str, float],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[dict[str, float], int, bool]:
    """EM estimate of the target prior. Labels are not an argument.

    Starts from the training prior and repeats q(y|x) proportional to
    p(y|x) * pi_t(y) / pi_s(y), then pi_t(y) as the mean of q.
    """
    names = [str(label) for label in classes]
    if not names:
        return {}, 0, False
    matrix = np.clip(np.asarray(proba, dtype=float), 0.0, None)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("probability columns must match classes")
    usable = matrix.sum(axis=1) > 0
    matrix = matrix[usable]
    source = np.array([max(float(source_prior.get(name, 0.0)), 1e-8) for name in names], dtype=float)
    source = source / source.sum()
    if len(matrix) == 0:
        return {name: float(source[index]) for index, name in enumerate(names)}, 0, False
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    prior = source.copy()
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        weighted = matrix * (prior / source)
        post = weighted / np.clip(weighted.sum(axis=1, keepdims=True), 1e-12, None)
        updated = post.mean(axis=0)
        updated = updated / updated.sum()
        if float(np.max(np.abs(updated - prior))) < tol:
            prior = updated
            converged = True
            break
        prior = updated
    return {name: float(prior[index]) for index, name in enumerate(names)}, n_iter, converged


def draw_prior_sample(frame: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into a prior sample and the rows that will be scored.

    The draw is a simple random sample after sorting by ``item_id``, so it
    estimates the class prior. It is not stratified.
    """
    if "item_id" not in frame.columns:
        raise ValueError("prior sample requires item_id")
    ordered = frame.sort_values("item_id").reset_index(drop=True)
    if n < 1 or n >= len(ordered):
        raise ValueError(f"prior sample size {n} does not leave a scored set from {len(ordered)} rows")
    chosen = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    mask = np.zeros(len(ordered), dtype=bool)
    mask[np.sort(chosen)] = True
    return ordered.loc[mask].copy(), ordered.loc[~mask].copy()


def reweight_labels(
    proba: np.ndarray,
    classes: Sequence[str],
    source_prior: dict[str, float],
    target_prior: dict[str, float],
) -> tuple[list[str], int]:
    """Argmax of p(y|x) times target prior over source prior.

    A source prior of 0 leaves that class at weight 0. A row whose reweighted
    mass is 0 keeps the uncorrected argmax and is counted in the second return.
    """
    weights = np.array(
        [
            (float(target_prior.get(label, 0.0)) / float(source_prior[label]))
            if float(source_prior.get(label, 0.0)) > 0
            else 0.0
            for label in classes
        ],
        dtype=float,
    )
    matrix = np.asarray(proba, dtype=float)
    adjusted = matrix * weights
    raw = matrix.argmax(axis=1)
    chosen = adjusted.argmax(axis=1)
    undefined = adjusted.sum(axis=1) <= 0
    chosen = np.where(undefined, raw, chosen)
    names = [str(label) for label in classes]
    return [names[int(index)] for index in chosen], int(undefined.sum())


def score_usage(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Six-way usage metrics plus the same metrics after collapsing rare routes."""
    truth = [str(label) for label in y_true]
    pred = [str(label) for label in y_pred]
    report = classification_metrics(truth, pred, list(USAGE_LABELS))
    mapped_truth: list[str] = []
    mapped_pred: list[str] = []
    for left, right in zip(truth, pred):
        collapsed_left = usage_class(left)
        collapsed_right = usage_class(right)
        if collapsed_left is None or collapsed_right is None:
            continue
        mapped_truth.append(collapsed_left)
        mapped_pred.append(collapsed_right)
    report["usage_class"] = classification_metrics(mapped_truth, mapped_pred, list(USAGE_CLASSES))
    return report


def station3_context(frame: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train prior, station 3 prior sample, and the station 3 rows left to score."""
    train = frame.loc[frame["split"].eq("train") & frame["usage"].notna()]
    cross = frame.loc[frame["split"].eq("cross_site") & frame["usage"].notna()]
    sample, score = draw_prior_sample(cross, n, seed)
    classes = list(USAGE_LABELS)
    source_prior, source_counts = empirical_prior(train["usage"].astype(str).tolist(), classes)
    target_prior, target_counts = empirical_prior(sample["usage"].astype(str).tolist(), classes)
    public = {
        "prior_sample_size": int(n),
        "prior_sample_seed": int(seed),
        "prior_sample_item_ids": sample["item_id"].astype(str).tolist(),
        "n_scored": int(len(score)),
        "n_cross_site": int(len(cross)),
        "source_prior": source_prior,
        "source_prior_counts": source_counts,
        "target_prior": target_prior,
        "target_prior_counts": target_counts,
        "laplace_alpha": 1,
        "target_prior_laplace": laplace_prior(target_counts, 1.0),
        "method": "p(y|x) * target_prior / source_prior, then argmax",
        "em_reference": "Saerens, Latinne, and Decaestecker, Neural Computation 14(1), 2002",
    }
    return sample, score, public


def score_shift_variants(
    y_true: Sequence[str],
    proba: np.ndarray,
    classes: Sequence[str],
    source_prior: dict[str, float],
    sample_counts: dict[str, int],
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Uncorrected scores, Laplace-prior scores, and EM-prior scores."""
    matrix = np.asarray(proba, dtype=float)
    names = [str(label) for label in classes]
    raw = [names[int(index)] for index in matrix.argmax(axis=1)]
    smoothed = laplace_prior(sample_counts, alpha)
    laplace_labels, n_laplace = reweight_labels(matrix, names, source_prior, smoothed)
    em, n_iter, converged = saerens_prior(matrix, names, source_prior)
    em_labels, n_em = reweight_labels(matrix, names, source_prior, em)
    return {
        "uncorrected": score_usage(y_true, raw),
        "label_shift_laplace": score_usage(y_true, laplace_labels),
        "label_shift_em": score_usage(y_true, em_labels),
        "n_undefined_reweight": {"label_shift_laplace": n_laplace, "label_shift_em": n_em},
        "em_prior": em,
        "em_iterations": n_iter,
        "em_converged": converged,
    }
