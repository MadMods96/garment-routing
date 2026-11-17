"""Garment-level bootstrap intervals keep the point estimate inside the CI."""

import pandas as pd

from garmentroute.labels import USAGE_CLASSES
from garmentroute.metrics import bootstrap_macro_f1, bootstrap_pairwise_kappa, bootstrap_model_vs_site_kappas


def test_macro_f1_bootstrap_contains_the_point():
    truth = ["Reuse", "Export", "Recycle", "Other"] * 20
    pred = ["Reuse", "Export", "Reuse", "Other"] * 20
    report = bootstrap_macro_f1(truth, pred, USAGE_CLASSES, n_boot=200, seed=0)
    assert report["n_garments"] == 80
    assert report["ci_low"] <= report["point"] <= report["ci_high"]


def test_pairwise_kappa_bootstrap_resamples_garments():
    frame = pd.DataFrame(
        [
            {"gold_id": "1", "usage": "Reuse"},
            {"gold_id": "1", "usage": "Reuse"},
            {"gold_id": "2", "usage": "Export"},
            {"gold_id": "2", "usage": "Recycle"},
            {"gold_id": "3", "usage": "Reuse"},
            {"gold_id": "3", "usage": "Export"},
        ]
    )
    report = bootstrap_pairwise_kappa(frame, "usage", n_boot=100, seed=1)
    assert report["n_garments"] == 3
    assert report["point"] is not None
    assert report["ci_low"] <= report["point"] <= report["ci_high"]


def test_model_vs_site_bootstrap_uses_wargon_photos_only():
    gold = pd.DataFrame(
        [
            {"item_id": "a1", "gold_id": "1", "station": "station1", "usage": "Reuse", "condition": 4},
            {"item_id": "a3", "gold_id": "1", "station": "station3", "usage": "Export", "condition": 3},
            {"item_id": "b1", "gold_id": "2", "station": "station2", "usage": "Recycle", "condition": 2},
            {"item_id": "b3", "gold_id": "2", "station": "station3", "usage": "Recycle", "condition": 2},
        ]
    )
    preds = {
        "a1": {"usage": "Reuse", "condition": 4},
        "b1": {"usage": "Recycle", "condition": 2},
    }
    report = bootstrap_model_vs_site_kappas(gold, preds, n_boot=50, seed=2)
    assert report["model_vs_wargon_usage"]["n_garments"] == 2
    assert report["model_vs_myrorna_usage"]["point"] is not None
