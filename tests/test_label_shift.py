"""Label-shift weights and balanced accuracy."""

import numpy as np
import pandas as pd

from garmentroute.label_shift import (
    draw_prior_sample,
    empirical_prior,
    laplace_prior,
    reweight_labels,
    saerens_prior,
)
from garmentroute.labels import USAGE_LABELS
from garmentroute.metrics import classification_metrics
from garmentroute.reporting import render_results


def test_balanced_accuracy_is_mean_recall_of_classes_that_occur():
    report = classification_metrics(["Reuse", "Reuse", "Reuse", "Export"], ["Reuse"] * 4, ["Reuse", "Export"])
    assert report["accuracy"] == 0.75
    assert report["balanced_accuracy"] == 0.5


def test_reweight_flips_the_call_when_the_target_prior_moves():
    proba = np.array([[0.6, 0.4]])
    same, undefined = reweight_labels(
        proba,
        ["Reuse", "Export"],
        {"Reuse": 0.5, "Export": 0.5},
        {"Reuse": 0.5, "Export": 0.5},
    )
    moved, _ = reweight_labels(
        proba,
        ["Reuse", "Export"],
        {"Reuse": 0.5, "Export": 0.5},
        {"Reuse": 0.1, "Export": 0.9},
    )
    assert undefined == 0
    assert same == ["Reuse"]
    assert moved == ["Export"]


def test_laplace_keeps_a_zero_count_class_possible():
    counts = {label: 0 for label in USAGE_LABELS}
    counts["Reuse"] = 191
    counts["Export"] = 9
    prior = laplace_prior(counts, 1.0)
    assert prior["Recycle"] == 1 / 206
    assert prior["Repair"] > 0
    assert abs(sum(prior.values()) - 1) < 1e-12
    raw = {label: counts[label] / 200 for label in counts}
    blocked, _ = reweight_labels(
        np.array([[0.004, 0.004, 0.992, 0.0, 0.0, 0.0]]),
        list(USAGE_LABELS),
        {label: 1 / 6 for label in USAGE_LABELS},
        raw,
    )
    opened, undefined = reweight_labels(
        np.array([[0.004, 0.004, 0.992, 0.0, 0.0, 0.0]]),
        list(USAGE_LABELS),
        {label: 1 / 6 for label in USAGE_LABELS},
        prior,
    )
    assert undefined == 0
    assert blocked == ["Reuse"]
    assert opened == ["Recycle"]


def test_saerens_em_moves_toward_the_model_posterior_without_labels():
    proba = np.tile([0.1, 0.9], (20, 1))
    prior, iterations, converged = saerens_prior(
        proba,
        ["Reuse", "Export"],
        {"Reuse": 0.5, "Export": 0.5},
    )
    assert converged
    assert iterations >= 1
    assert prior["Export"] > 0.8
    assert abs(sum(prior.values()) - 1) < 1e-9


def test_prior_sample_is_held_out_of_the_scored_rows():
    frame = pd.DataFrame({"item_id": [f"g{index}" for index in range(10)], "usage": ["Reuse"] * 10})
    sample, score = draw_prior_sample(frame, 3, 7)
    assert len(sample) == 3
    assert len(score) == 7
    assert set(sample["item_id"]).isdisjoint(set(score["item_id"]))
    again, _ = draw_prior_sample(frame, 3, 7)
    assert sample["item_id"].tolist() == again["item_id"].tolist()
    prior, counts = empirical_prior(sample["usage"], ["Reuse", "Export"])
    assert prior["Reuse"] == 1.0
    assert prior["Export"] == 0.0
    assert counts["Export"] == 0


def test_station3_section_quotes_both_variants():
    results = {
        "dataset": "clothing_v3",
        "smoke": False,
        "seed": 1,
        "split_meta": {"stratified": True},
        "majority": {"cross_site": {"usage": {"balanced_accuracy": 0.21, "macro_f1": 0.22, "accuracy": 0.97}}},
        "router": {"on_expert_attributes": {}},
        "attribute_baselines": {"models": {}},
        "station3": {
            "note": "Accuracy is secondary.",
            "prior_sample_size": 200,
            "prior_sample_seed": 42,
            "n_scored": 7216,
            "n_cross_site": 7416,
            "source_prior": {"Reuse": 0.5},
            "source_prior_counts": {"Reuse": 10},
            "target_prior": {"Reuse": 0.9},
            "target_prior_counts": {"Reuse": 180},
            "models": {
                "majority": {
                    "uncorrected": {
                        "balanced_accuracy": 0.31,
                        "macro_f1": 0.32,
                        "accuracy": 0.33,
                        "usage_class": {"balanced_accuracy": 0.34, "macro_f1": 0.35, "accuracy": 0.36},
                    },
                    "label_shift_laplace": {
                        "balanced_accuracy": 0.41,
                        "macro_f1": 0.42,
                        "accuracy": 0.43,
                        "usage_class": {"balanced_accuracy": 0.44, "macro_f1": 0.45, "accuracy": 0.46},
                    },
                    "label_shift_em": {
                        "balanced_accuracy": 0.51,
                        "macro_f1": 0.52,
                        "accuracy": 0.53,
                        "usage_class": {"balanced_accuracy": 0.54, "macro_f1": 0.55, "accuracy": 0.56},
                    },
                    "em_prior": {"Reuse": 0.8, "Export": 0.2},
                    "em_iterations": 4,
                    "em_converged": True,
                    "n_undefined_reweight": {"label_shift_laplace": 0, "label_shift_em": 0},
                }
            },
        },
        "gold": {"expert_vs_expert_usage": {"cohen_kappa": 0.1, "raw_agreement": 0.2, "n_pairs": 1}},
        "front": None,
        "fusion": None,
    }
    text = render_results(results, {"n_rows": 1, "n_gold": 0})
    assert "Accuracy is secondary." in text
    assert "0.310" in text
    assert "0.410" in text
    assert "Laplace prior" in text
    assert "EM prior (no labels)" in text
    assert "Macro-F1, all 6 expert classes" in text
    assert "Macro-F1, 4 training classes" in text
    assert "0.99" not in text
