"""Calibration and abstention.

Temperature scaling and the abstention threshold are fit on the validation
split only. Test, cross-site, and gold rows are never used to choose them.
"""

from __future__ import annotations

from typing import Sequence

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


def select_threshold(
    confidences: Sequence[float],
    correct: Sequence[bool],
    min_coverage: float,
) -> dict:
    """Pick a confidence threshold on validation rows.

    The chosen threshold is the one with the highest accuracy among thresholds
    that still cover at least ``min_coverage`` of the validation rows. Ties
    keep the lower threshold, which abstains less.
    """
    confidence = np.asarray(list(confidences), dtype=float)
    hits = np.asarray(list(correct), dtype=float)
    if confidence.size == 0:
        return {"threshold": None, "coverage": None, "accuracy": None, "min_coverage": min_coverage}
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
            "note": "no threshold met the minimum coverage",
        }
    return best


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by a positive temperature."""
    return np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)


def fit_temperature(logits: np.ndarray, labels: Sequence[int], steps: int = 100) -> float:
    """Fit one scalar temperature by minimising validation NLL with LBFGS.

    Requires PyTorch. The labels are class indices aligned with ``logits``.
    """
    import torch
    import torch.nn.functional as F

    scores = torch.tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    target = torch.tensor(list(labels), dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimiser = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=steps)

    def closure() -> torch.Tensor:
        optimiser.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 100.0)
        loss = F.cross_entropy(scores / temperature, target)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 100.0))
