"""Usage baselines that see annotation attributes and no photographs.

Logistic regression and LightGBM are fit on the train split. Brand frequency
is the proportion of that brand among training rows. Validation is used for
LightGBM early stopping and for permutation importance. Test, station 3, and
the gold set are scored or ignored, never used to fit or to rank features.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from garmentroute.label_shift import score_shift_variants, station3_context
from garmentroute.labels import USAGE_CLASSES, USAGE_LABELS, usage_class
from garmentroute.metrics import classification_metrics
from garmentroute.util import is_missing, set_seed

CATEGORICAL = ("stains", "holes", "material", "type", "trend")
NUMERIC = ("condition", "pilling", "brand_frequency")
FEATURES = ("condition", "pilling", "stains", "holes", "material", "type", "brand_frequency", "trend")


def _blank(value: Any) -> bool:
    if is_missing(value):
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if missing is True:
        return True
    return False


def _brand_key(value: Any) -> str:
    """Stable key for a brand string. A blank brand is one shared key."""
    if _blank(value):
        return ""
    return str(value).strip()


def _category_value(value: Any) -> Any:
    """Keep the class ``None`` as text. A missing field stays missing."""
    if _blank(value):
        return np.nan
    return str(value).strip()


def brand_frequency_map(brands: pd.Series) -> dict[str, float]:
    """Training-set proportion of each brand string. The keys sum to 1."""
    keys = brands.map(_brand_key)
    counts = keys.value_counts()
    total = float(counts.sum())
    if total == 0:
        return {}
    return {str(key): float(value) / total for key, value in counts.items()}


def design_frame(frame: pd.DataFrame, frequency_map: dict[str, float]) -> pd.DataFrame:
    """Columns the estimators see. Unseen brands get frequency 0.

    ``holes`` and ``stains`` keep a missing value when the annotation has none.
    The string ``None`` is the real class and is not used to fill gaps.
    """
    data: dict[str, Any] = {
        "condition": pd.to_numeric(frame["condition"], errors="coerce"),
        "pilling": pd.to_numeric(frame["pilling"], errors="coerce"),
        "brand_frequency": frame["brand"].map(_brand_key).map(lambda key: float(frequency_map.get(key, 0.0))),
    }
    for column in CATEGORICAL:
        data[column] = [_category_value(value) for value in frame[column].tolist()]
    return pd.DataFrame(data, index=frame.index)


def _levels(frame: pd.DataFrame) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for column in CATEGORICAL:
        values = sorted({str(value) for value in frame[column].tolist() if isinstance(value, str)})
        found[column] = values
    return found


def _for_lightgbm(frame: pd.DataFrame, levels: dict[str, list[str]]) -> pd.DataFrame:
    """Categorical dtypes from the training levels. Unseen levels become missing."""
    prepared = frame.copy()
    for column, categories in levels.items():
        prepared[column] = pd.Categorical(prepared[column], categories=categories)
    return prepared


def _logistic_pipeline(config: dict[str, Any], seed: int) -> Pipeline:
    logistic = config["logistic_regression"]
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            ),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC)),
            ("categorical", categorical, list(CATEGORICAL)),
        ]
    )
    classifier = LogisticRegression(
        solver=str(logistic["solver"]),
        l1_ratio=float(logistic.get("l1_ratio", 0.0)),
        C=float(logistic["C"]),
        class_weight=logistic["class_weight"],
        max_iter=int(logistic["max_iter"]),
        tol=float(logistic["tol"]),
        random_state=seed,
    )
    return Pipeline([("features", transform), ("classifier", classifier)])


def _macro_f1(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    return float(f1_score(list(y_true), list(y_pred), labels=list(USAGE_LABELS), average="macro", zero_division=0))


def _usage_class_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    truth = [usage_class(label) for label in y_true]
    pred = [usage_class(label) for label in y_pred]
    paired_truth = []
    paired_pred = []
    for left, right in zip(truth, pred):
        if left is None or right is None:
            continue
        paired_truth.append(left)
        paired_pred.append(right)
    return classification_metrics(paired_truth, paired_pred, list(USAGE_CLASSES))


def _score(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    report = classification_metrics(y_true, y_pred, list(USAGE_LABELS))
    report["usage_class"] = _usage_class_metrics(y_true, y_pred)
    return report


def _permutation(
    predict: Callable[[pd.DataFrame], Any],
    frame: pd.DataFrame,
    y_true: list[str],
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Drop in validation macro-F1 when one input column is shuffled."""
    baseline = _macro_f1(y_true, predict(frame))
    rng = np.random.default_rng(seed)
    ranked = []
    for column in FEATURES:
        original = frame[column].to_numpy().copy()
        drops = []
        for _ in range(repeats):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(original)
            drops.append(baseline - _macro_f1(y_true, predict(shuffled)))
        ranked.append(
            {
                "feature": column,
                "mean_macro_f1_drop": float(np.mean(drops)),
                "std_macro_f1_drop": float(np.std(drops)),
            }
        )
    ranked.sort(key=lambda item: item["mean_macro_f1_drop"], reverse=True)
    return {
        "split": "val",
        "scoring": "macro_f1",
        "repeats": int(repeats),
        "baseline_macro_f1": baseline,
        "features": ranked,
    }


def _split_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    work = frame[frame["split"] == split]
    return work[work["usage"].map(lambda value: not _blank(value))].copy()


def attribute_baseline_block(frame: pd.DataFrame, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Fit both models and return scores, hyperparameters, and importance.

    The returned dict is JSON-ready. Fitted estimators are not included.
    """
    set_seed(seed)
    train = _split_frame(frame, "train")
    validation = _split_frame(frame, "val")
    if train.empty or validation.empty:
        raise ValueError("attribute baselines need train and val rows")
    frequency_map = brand_frequency_map(train["brand"])
    train_x = design_frame(train, frequency_map)
    valid_x = design_frame(validation, frequency_map)
    levels = _levels(train_x)
    train_y = train["usage"].astype(str).tolist()
    valid_y = validation["usage"].astype(str).tolist()

    logistic = _logistic_pipeline(config, seed)
    logistic.fit(train_x, train_y)
    logistic_steps = int(np.max(logistic.named_steps["classifier"].n_iter_))

    label_encoder = LabelEncoder()
    train_codes = label_encoder.fit_transform(train_y)
    valid_codes = label_encoder.transform(valid_y)
    boosting_config = config["lightgbm"]
    boosting = LGBMClassifier(
        objective="multiclass",
        n_estimators=int(boosting_config["n_estimators"]),
        learning_rate=float(boosting_config["learning_rate"]),
        num_leaves=int(boosting_config["num_leaves"]),
        min_child_samples=int(boosting_config["min_child_samples"]),
        subsample=float(boosting_config["subsample"]),
        colsample_bytree=float(boosting_config["colsample_bytree"]),
        class_weight=boosting_config["class_weight"],
        random_state=seed,
        n_jobs=1,
        verbosity=-1,
    )
    train_trees = _for_lightgbm(train_x, levels)
    valid_trees = _for_lightgbm(valid_x, levels)
    boosting.fit(
        train_trees,
        train_codes,
        categorical_feature=list(CATEGORICAL),
        eval_X=valid_trees,
        eval_y=valid_codes,
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(int(boosting_config["early_stopping_rounds"]), verbose=False),
            log_evaluation(0),
        ],
    )

    def predict_logistic(features: pd.DataFrame) -> Any:
        return logistic.predict(features)

    def proba_logistic(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        classifier = logistic.named_steps["classifier"]
        return logistic.predict_proba(features), [str(label) for label in classifier.classes_]

    def predict_boosting(features: pd.DataFrame) -> Any:
        codes = np.asarray(boosting.predict(_for_lightgbm(features, levels))).astype(int)
        return label_encoder.inverse_transform(codes)

    def proba_boosting(features: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        matrix = boosting.predict_proba(_for_lightgbm(features, levels))
        names = [str(label_encoder.inverse_transform([int(code)])[0]) for code in boosting.classes_]
        return matrix, names

    repeats = int(config["importance_repeats"])
    models = {
        "logistic_regression": {
            "hyperparameters": dict(config["logistic_regression"]),
            "n_iter": logistic_steps,
            "converged": bool(logistic_steps < int(config["logistic_regression"]["max_iter"])),
            "predict": predict_logistic,
            "proba": proba_logistic,
        },
        "lightgbm": {
            "hyperparameters": dict(boosting_config),
            "best_iteration": int(boosting.best_iteration_),
            "predict": predict_boosting,
            "proba": proba_boosting,
        },
    }
    block: dict[str, Any] = {
        "target": str(config.get("target", "usage")),
        "inputs": (
            "annotation attributes only: condition, pilling, stains, holes, material, type, "
            "brand frequency, trend"
        ),
        "features": list(FEATURES),
        "brand_frequency": (
            "proportion of training rows with that brand string; a blank brand is one string; "
            "a brand absent from train is 0"
        ),
        "fit_on": "train",
        "early_stopping_split": "val",
        "importance_method": "permutation",
        "importance_split": "val",
        "seed": int(seed),
        "n_train": int(len(train)),
        "n_val": int(len(validation)),
        "note": (
            "Photographs are not used. Macro-F1, all 6 expert classes, is the ceiling for a model that is given these "
            "annotation attributes and predicts usage. Material, type, brand frequency, and trend "
            "are annotation fields."
        ),
        "models": {},
    }
    for name, spec in models.items():
        scored: dict[str, Any] = {
            "hyperparameters": spec["hyperparameters"],
            "splits": {},
            "permutation_importance": _permutation(spec["predict"], valid_x, valid_y, repeats, seed),
        }
        if "n_iter" in spec:
            scored["n_iter"] = spec["n_iter"]
            scored["converged"] = spec["converged"]
        if "best_iteration" in spec:
            scored["best_iteration"] = spec["best_iteration"]
        for split in ("test", "cross_site"):
            evaluation = _split_frame(frame, split)
            features = design_frame(evaluation, frequency_map)
            pred = [str(label) for label in spec["predict"](features)]
            truth = evaluation["usage"].astype(str).tolist()
            scored["splits"][split] = _score(truth, pred)
        block["models"][name] = scored
    sample_size = config.get("station3_prior_sample_size")
    if sample_size:
        _sample, score_rows, public = station3_context(frame, int(sample_size), seed)
        features = design_frame(score_rows, frequency_map)
        truth = score_rows["usage"].astype(str).tolist()
        probability_models: dict[str, Any] = {
            key: public[key]
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
        for name, spec in models.items():
            matrix, class_names = spec["proba"](features)
            probability_models[name] = score_shift_variants(
                truth,
                matrix,
                class_names,
                public["source_prior"],
                public["target_prior_counts"],
                public["laplace_alpha"],
            )
        block["station3_probability_models"] = probability_models
    return block
