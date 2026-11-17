"""Calibration and abstention.

Temperature scaling and the abstention threshold are fit on the validation
split only. Test, cross-site, and gold rows are never used to choose them.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 15,
) -> dict:
    """Expected calibration error of a binary correctness-versus-confidence diagram."""
    confidence = np.asarray(list(confidences), dtype=float)
    hits = np.asarray(list(correct), dtype=float)
    if confidence.size == 0:
        return {"n": 0, "ece": None, "n_bins": n_bins, "bins": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidence, edges[1:-1], right=False)
    total = confidence.size
    ece = 0.0
    bins = []
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin": index,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "n": 0,
                    "confidence": None,
                    "accuracy": None,
                }
            )
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(hits[mask].mean())
        ece += (count / total) * abs(bin_accuracy - bin_confidence)
        bins.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "n": count,
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )
    return {"n": int(total), "ece": float(ece), "n_bins": n_bins, "bins": bins}


def coverage_accuracy(
    confidences: Sequence[float],
    correct: Sequence[bool],
    fractions: Sequence[float] | None = None,
) -> list[dict]:
    """Accuracy when the model keeps only its most confident fraction of rows.

    Rows are sorted by confidence, highest first. At each coverage the accuracy
    is computed on that prefix alone.
    """
    confidence = np.asarray(list(confidences), dtype=float)
    hits = np.asarray(list(correct), dtype=float)
    if confidence.size == 0:
        return []
    if fractions is None:
        fractions = tuple(round(value, 2) for value in np.linspace(0.05, 1.0, 20))
    order = np.argsort(-confidence)
    hits = hits[order]
    cumulative = np.cumsum(hits)
    rows = []
    n = hits.size
    for fraction in fractions:
        keep = max(1, int(np.ceil(float(fraction) * n)))
        keep = min(keep, n)
        rows.append(
            {
                "coverage": float(keep / n),
                "requested_coverage": float(fraction),
                "n": int(keep),
                "accuracy": float(cumulative[keep - 1] / keep),
            }
        )
    return rows


def area_under_risk_coverage(confidences: Sequence[float], correct: Sequence[bool]) -> float | None:
    """AURC: integral of selective risk (1 - accuracy) over coverage in (0, 1].

    Lower is better. Uses the stepwise risk curve after sorting by confidence
    descending (Geifman & El-Yaniv style).
    """
    confidence = np.asarray(list(confidences), dtype=float)
    hits = np.asarray(list(correct), dtype=float)
    n = int(confidence.size)
    if n == 0:
        return None
    order = np.argsort(-confidence)
    hits = hits[order]
    risks = 1.0 - (np.cumsum(hits) / np.arange(1, n + 1))
    # Each prefix of size k covers k/n; step width is 1/n.
    return float(np.mean(risks))


def coverage_at_target_accuracy(
    confidences: Sequence[float],
    correct: Sequence[bool],
    target: float,
) -> float | None:
    """Largest coverage whose most-confident prefix still has accuracy >= ``target``.

    Returns ``None`` when even the single most confident prediction misses the target.
    """
    confidence = np.asarray(list(confidences), dtype=float)
    hits = np.asarray(list(correct), dtype=float)
    n = int(confidence.size)
    if n == 0:
        return None
    order = np.argsort(-confidence)
    hits = hits[order]
    best: float | None = None
    for keep in range(1, n + 1):
        accuracy = float(hits[:keep].mean())
        if accuracy >= float(target):
            best = float(keep / n)
        else:
            # Once the prefix falls below target, larger prefixes only get worse or
            # stay mixed; keep scanning in case of non-monotone noise is rare, but
            # for the standard risk-coverage curve accuracy is not guaranteed mono.
            continue
    return best


def _prefix_accuracy(confidence: np.ndarray, correct: np.ndarray, fraction: float) -> float:
    n = int(confidence.size)
    keep = max(1, int(np.ceil(float(fraction) * n)))
    keep = min(keep, n)
    order = np.argsort(-confidence)
    return float(correct[order][:keep].mean())


def bootstrap_selective_accuracy(
    confidences: Sequence[float],
    correct: Sequence[bool],
    fractions: Sequence[float] = (0.25, 0.5),
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Garment-level bootstrap CIs for selective accuracy at fixed coverage fractions."""
    from garmentroute.metrics import bootstrap_ci

    confidence = np.asarray(list(confidences), dtype=float)
    hits = np.asarray(list(correct), dtype=float)
    n = int(confidence.size)
    out: dict[str, Any] = {"n_garments": n, "n_boot": int(n_boot), "alpha": float(alpha), "fractions": {}}
    if n == 0:
        for fraction in fractions:
            out["fractions"][f"{float(fraction):.2f}"] = {
                "point": None,
                **bootstrap_ci([], alpha=alpha),
            }
        return out
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    for fraction in fractions:
        point = _prefix_accuracy(confidence, hits, fraction)
        draws = []
        for _ in range(int(n_boot)):
            sample = rng.choice(indices, size=n, replace=True)
            draws.append(_prefix_accuracy(confidence[sample], hits[sample], fraction))
        out["fractions"][f"{float(fraction):.2f}"] = {
            "point": point,
            "requested_coverage": float(fraction),
            **bootstrap_ci(draws, alpha=alpha),
        }
    return out


def selective_coverage_report(
    confidences: Sequence[float],
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any],
    fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    accuracy_targets: Sequence[float] = (0.60, 0.70),
    bootstrap_fractions: Sequence[float] = (0.25, 0.5),
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Accuracy, macro-F1, AURC, coverage targets, and bootstrap CIs at key coverages."""
    from sklearn.metrics import accuracy_score, f1_score

    confidence = np.asarray(list(confidences), dtype=float)
    truth = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    label_list = list(labels)
    n = int(confidence.size)
    if n == 0:
        return {
            "n": 0,
            "points": [],
            "aurc": None,
            "coverage_at_accuracy": {f"{float(t):.2f}": None for t in accuracy_targets},
            "accuracy_bootstrap": bootstrap_selective_accuracy(
                [], [], fractions=bootstrap_fractions, n_boot=n_boot, seed=seed
            ),
        }
    correct = pred == truth
    order = np.argsort(-confidence)
    truth_o = truth[order]
    pred_o = pred[order]
    points = []
    for fraction in fractions:
        keep = max(1, int(np.ceil(float(fraction) * n)))
        keep = min(keep, n)
        points.append(
            {
                "requested_coverage": float(fraction),
                "coverage": float(keep / n),
                "n": int(keep),
                "accuracy": float(accuracy_score(truth_o[:keep], pred_o[:keep])),
                "macro_f1": float(
                    f1_score(
                        truth_o[:keep],
                        pred_o[:keep],
                        labels=label_list,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    coverage_targets = {
        f"{float(target):.2f}": coverage_at_target_accuracy(confidence, correct, target)
        for target in accuracy_targets
    }
    return {
        "n": n,
        "points": points,
        "aurc": area_under_risk_coverage(confidence, correct),
        "coverage_at_accuracy": coverage_targets,
        "accuracy_bootstrap": bootstrap_selective_accuracy(
            confidence, correct, fractions=bootstrap_fractions, n_boot=n_boot, seed=seed
        ),
    }


def confidence_from_probs(probabilities: np.ndarray, signal: str) -> np.ndarray:
    """Per-row confidence scores. Higher means keep earlier on the coverage curve."""
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probabilities must be (n, c)")
    ordered = np.sort(probs, axis=1)
    if signal == "max_prob":
        return ordered[:, -1]
    if signal == "margin":
        top1 = ordered[:, -1]
        top2 = ordered[:, -2] if ordered.shape[1] > 1 else np.zeros(len(ordered))
        return top1 - top2
    if signal == "entropy":
        clipped = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(clipped * np.log(clipped), axis=1)
        return -entropy
    raise ValueError(f"unknown confidence signal: {signal}")


def threshold_at_coverage(
    confidences: Sequence[float],
    target_coverage: float = 0.5,
) -> float | None:
    """Confidence cutoff that keeps about ``target_coverage`` of garments (selective curve)."""
    confidence = np.asarray(list(confidences), dtype=float)
    if confidence.size == 0:
        return None
    ordered = np.sort(confidence)[::-1]
    keep = max(1, int(round(float(target_coverage) * len(ordered))))
    keep = min(keep, len(ordered))
    return float(ordered[keep - 1])


def select_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    min_coverage: float,
    *,
    mode: str = "coverage",
) -> dict:
    """Pick a confidence threshold on validation rows.

    ``mode="coverage"`` (default): threshold on the selective curve that keeps
    about ``min_coverage`` of val garments (e.g. 50% coverage → not the looser
    accuracy-max threshold that often lands near 42%).

    ``mode="max_accuracy"``: highest accuracy among thresholds that still cover
    at least ``min_coverage`` (ties keep the lower threshold).
    """
    confidence = np.asarray(list(confidences), dtype=float)
    hits = np.asarray(list(correct), dtype=float)
    if confidence.size == 0:
        return {"threshold": None, "coverage": None, "accuracy": None, "min_coverage": min_coverage}
    if mode == "coverage":
        cutoff = threshold_at_coverage(confidence, target_coverage=float(min_coverage))
        if cutoff is None:
            return {
                "threshold": None,
                "coverage": None,
                "accuracy": None,
                "min_coverage": float(min_coverage),
                "mode": mode,
            }
        mask = confidence >= cutoff
        return {
            "threshold": float(cutoff),
            "coverage": float(mask.mean()),
            "accuracy": float(hits[mask].mean()) if mask.any() else None,
            "n": int(mask.sum()),
            "min_coverage": float(min_coverage),
            "mode": mode,
            "target_coverage": float(min_coverage),
        }
    best: dict | None = None
    for threshold in np.unique(confidence):
        mask = confidence >= threshold
        coverage = float(mask.mean())
        if coverage < min_coverage or mask.sum() == 0:
            continue
        accuracy = float(hits[mask].mean())
        candidate = {
            "threshold": float(threshold),
            "coverage": coverage,
            "accuracy": accuracy,
            "n": int(mask.sum()),
            "min_coverage": float(min_coverage),
            "mode": mode,
        }
        if best is None or accuracy > best["accuracy"] or (
            accuracy == best["accuracy"] and threshold < best["threshold"]
        ):
            best = candidate
    if best is None:
        return {
            "threshold": None,
            "coverage": None,
            "accuracy": None,
            "min_coverage": float(min_coverage),
            "mode": mode,
            "note": "no threshold met the minimum coverage",
        }
    return best


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by a positive temperature."""
    return np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)


def fit_temperature(
    logits: np.ndarray,
    labels: Sequence[int],
    steps: int = 100,
    t_min: float = 0.05,
    t_max: float = 20.0,
) -> float:
    """Fit one scalar temperature by minimising validation NLL with LBFGS.

    Optimises ``log(T)`` and clamps ``T`` to ``[t_min, t_max]``. Requires
    PyTorch. The labels are class indices aligned with ``logits``.
    """
    import torch
    import torch.nn.functional as F

    scores = torch.tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    target = torch.tensor(list(labels), dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=steps)
    lower = float(np.log(t_min))
    upper = float(np.log(t_max))

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        temperature = log_temperature.clamp(lower, upper).exp()
        loss = F.cross_entropy(scores / temperature, target)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().clamp(lower, upper).exp())
