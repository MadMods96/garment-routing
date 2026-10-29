"""Attribute baselines fit on train and keep missing holes missing."""

import numpy as np
import pandas as pd

from garmentroute.attribute_baseline import attribute_baseline_block, brand_frequency_map, design_frame
from garmentroute.reporting import render_results


def _config() -> dict:
    return {
        "target": "usage",
        "importance_repeats": 2,
        "logistic_regression": {
            "solver": "saga",
            "l1_ratio": 0.0,
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 200,
            "tol": 0.001,
        },
        "lightgbm": {
            "n_estimators": 40,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "class_weight": "balanced",
            "early_stopping_rounds": 5,
        },
        "station3_prior_sample_size": 4,
    }


def _row(split: str, usage: str, condition: int, brand: str, holes: str | None, item_id: str = "row") -> dict:
    return {
        "item_id": item_id,
        "split": split,
        "usage": usage,
        "condition": condition,
        "pilling": 5,
        "stains": "None",
        "holes": holes,
        "material": "100% cotton",
        "type": "Top",
        "brand": brand,
        "trend": "None",
    }


def test_brand_frequency_ignores_brands_outside_train():
    frequency = brand_frequency_map(pd.Series(["A", "A", "B", None]))
    assert frequency["A"] == 0.5
    assert frequency["B"] == 0.25
    assert frequency[""] == 0.25
    designed = design_frame(
        pd.DataFrame(
            [
                _row("test", "Reuse", 5, "A", "None"),
                _row("test", "Reuse", 5, "OnlyInTest", "None"),
                _row("test", "Reuse", 5, "A", None),
            ]
        ),
        frequency,
    )
    assert designed["brand_frequency"].tolist() == [0.5, 0.0, 0.5]
    assert designed.loc[0, "holes"] == "None"
    assert pd.isna(designed.loc[2, "holes"])
    assert designed.loc[0, "stains"] == "None"
    assert designed.loc[0, "trend"] == "None"


def test_models_fit_on_train_and_score_held_out_splits():
    rows = []
    serial = 0
    for usage, condition in (("Reuse", 5), ("Export", 3), ("Recycle", 2), ("Repair", 1)):
        for _ in range(12):
            rows.append(_row("train", usage, condition, "TrainBrand", "None", f"id-{serial}"))
            serial += 1
        for _ in range(4):
            rows.append(_row("val", usage, condition, "TrainBrand", "None", f"id-{serial}"))
            serial += 1
        for _ in range(4):
            rows.append(_row("test", usage, condition, "UnseenBrand", "None", f"id-{serial}"))
            serial += 1
        for _ in range(3):
            rows.append(_row("cross_site", usage, condition, "TrainBrand", "None", f"id-{serial}"))
            serial += 1
    rows.append(_row("gold", "Recycle", 5, "GoldBrand", "None", f"id-{serial}"))
    block = attribute_baseline_block(pd.DataFrame(rows), 0, _config())
    assert block["n_train"] == 48
    assert block["n_val"] == 16
    assert block["fit_on"] == "train"
    assert block["importance_split"] == "val"
    lightgbm = block["models"]["lightgbm"]
    assert set(lightgbm["splits"]) == {"test", "cross_site"}
    assert lightgbm["splits"]["test"]["accuracy"] == 1.0
    names = [row["feature"] for row in lightgbm["permutation_importance"]["features"]]
    assert names[0] == "condition"
    assert "brand_frequency" in names
    logistic = block["models"]["logistic_regression"]["splits"]["test"]
    assert logistic["macro_f1"] > 0.5
    assert logistic["usage_class"]["macro_f1"] > 0.5
    shift = block["station3_probability_models"]
    assert shift["prior_sample_size"] == 4
    assert shift["n_scored"] == 8
    assert shift["n_scored"] + shift["prior_sample_size"] == shift["n_cross_site"]
    assert set(shift["logistic_regression"]) >= {"uncorrected", "label_shift_laplace", "label_shift_em"}
    assert "balanced_accuracy" in shift["lightgbm"]["uncorrected"]
    assert "balanced_accuracy" in shift["lightgbm"]["label_shift_laplace"]
    assert "balanced_accuracy" in shift["lightgbm"]["label_shift_em"]
    assert shift["target_prior_laplace"]["Recycle"] > 0
    assert abs(sum(shift["target_prior_laplace"].values()) - 1) < 1e-9


def test_readme_quotes_attribute_macro_f1():
    results = {
        "dataset": "clothing_v3",
        "smoke": False,
        "seed": 1,
        "split_meta": {"stratified": True},
        "majority": {
            "test": {"usage": {"macro_f1": 0.11}, "usage_class": {"macro_f1": 0.22}},
            "cross_site": {"usage": {"macro_f1": 0.33}, "usage_class": {"macro_f1": 0.44}},
        },
        "gold": {"expert_vs_expert_usage": {"cohen_kappa": 0.1, "raw_agreement": 0.2, "n_pairs": 1}},
        "router": {
            "on_expert_attributes": {
                "test": {"macro_f1": 0.55, "usage_class_agreement": {"macro_f1": 0.66}},
                "cross_site": {"macro_f1": 0.77, "usage_class_agreement": {"macro_f1": 0.88}},
            }
        },
        "attribute_baselines": {
            "note": "Photographs are not used.",
            "features": ["condition", "brand_frequency"],
            "fit_on": "train",
            "brand_frequency": "train proportion",
            "early_stopping_split": "val",
            "importance_split": "val",
            "seed": 7,
            "models": {
                "logistic_regression": {
                    "splits": {
                        "test": {"macro_f1": 0.101, "usage_class": {"macro_f1": 0.102}},
                        "cross_site": {"macro_f1": 0.103, "usage_class": {"macro_f1": 0.104}},
                    },
                    "permutation_importance": {
                        "repeats": 5,
                        "features": [{"feature": "condition", "mean_macro_f1_drop": 0.2, "std_macro_f1_drop": 0.01}],
                    },
                },
                "lightgbm": {
                    "splits": {
                        "test": {"macro_f1": 0.201, "usage_class": {"macro_f1": 0.202}},
                        "cross_site": {"macro_f1": 0.203, "usage_class": {"macro_f1": 0.204}},
                    },
                    "permutation_importance": {
                        "repeats": 5,
                        "features": [{"feature": "condition", "mean_macro_f1_drop": 0.3, "std_macro_f1_drop": 0.02}],
                    },
                },
            },
        },
        "front": None,
        "fusion": None,
    }
    text = render_results(results, {"n_rows": 4, "n_gold": 1})
    assert "0.101" in text
    assert "0.201" in text
    assert "0.55" in text
    assert "0.99" not in text
    assert "Photographs are not used." in text
