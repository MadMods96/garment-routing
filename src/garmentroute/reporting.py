"""Fill the README results section from JSON files. This is the only writer of those numbers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from garmentroute.labels import USAGE_LABELS


def _fmt(value: Any) -> str:
    if value is None:
        return "not computed"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _get(payload: dict | None, *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _gold_split_line(eda: dict | None) -> str:
    census = _get(eda, "gold_census")
    if isinstance(census, dict):
        return (
            f"Gold-split annotation records: {_fmt(_get(census, 'n_split_rows'))}. "
            "The gold-set table below separates garments, JSON files, and annotation records."
        )
    return f"Gold-split annotation records: {_fmt(_get(eda, 'n_gold'))}."


def _gold_census_lines(eda: dict | None) -> list[str]:
    """Unit labels for the gold counts. Figures come from ``gold_census``."""
    census = _get(eda, "gold_census")
    if not isinstance(census, dict):
        return []
    comments = _get(census, "unnumbered_comment_text") or []
    quoted = ", ".join(f"`{text}`" for text in comments if text)
    unnumbered = (
        "Annotation records in the gold split with no item number. "
        "The comment contains `Test`, so the file is held out of training. "
        "Each of these is one JSON file, not one of the numbered garments."
    )
    if quoted:
        unnumbered += f" Comment text: {quoted}."
    rows = [
        (
            "n_split_rows",
            "Annotation records in the gold split. One record is one row of the index.",
        ),
        (
            "n_json_files",
            "Distinct JSON files in the gold split.",
        ),
        (
            "n_garments",
            "Garments. One garment is one item number parsed from a comment matching `Test` plus digits.",
        ),
        (
            "n_annotations_with_item_number",
            "Annotation records that carry such a number. The mean below divides this count by the garment count.",
        ),
        (
            "mean_annotations_per_garment",
            "Mean numbered annotation records per garment. This is not the gold-split size.",
        ),
        (
            "n_lowercase_annotations_of_same_garments",
            "Further annotation records of those same garments. The comment uses a lowercase `test` plus a number, which does not receive a garment id. The file is in the split because it is under `test100`. These records are not in the mean and not in the agreement pairs.",
        ),
        (
            "n_unnumbered_comments",
            unnumbered,
        ),
    ]
    other = _get(census, "n_other_numbered_comments")
    if isinstance(other, int) and other > 0:
        rows.append(
            (
                "n_other_numbered_comments",
                "Annotation records whose comment has a number that is not one of the garment ids above.",
            )
        )
    parts = [
        "numbered records",
        "lowercase records of the same garments",
    ]
    if isinstance(other, int) and other > 0:
        parts.append("other numbered comments")
    parts.append("unnumbered comments")
    if len(parts) == 2:
        total = f"{parts[0]} and {parts[1]}"
    else:
        total = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    lines = [
        "The garment count times the mean equals the numbered annotation records. "
        "It does not equal the gold-split size. "
        f"The {total} add up to the gold-split size.",
        "",
        "| Count | What it counts |",
        "| --- | --- |",
    ]
    for key, meaning in rows:
        lines.append(f"| {_fmt(_get(census, key))} | {meaning} |")
    lines.append("")
    return lines


def _attribute_baseline_lines(results: dict) -> list[str]:
    """Comparison of attribute-only models with the majority guess and the rule router."""
    block = _get(results, "attribute_baselines")
    lines = ["", "### Attribute baselines", ""]
    if not isinstance(block, dict):
        lines.append("Attribute baselines: not computed.")
        lines.append("")
        return lines
    note = _get(block, "note")
    if isinstance(note, str) and note:
        lines.append(note)
    features = _get(block, "features")
    if isinstance(features, list):
        lines.append("Features: " + ", ".join(str(feature) for feature in features) + ".")
    lines.append(
        f"Fit on {_fmt(_get(block, 'fit_on'))}. "
        f"Brand frequency: {_fmt(_get(block, 'brand_frequency'))}. "
        f"LightGBM early stopping split: {_fmt(_get(block, 'early_stopping_split'))}. "
        f"Permutation importance split: {_fmt(_get(block, 'importance_split'))}. "
        f"Seed {_fmt(_get(block, 'seed'))}."
    )
    lines.append("")
    lines.append("| Model | Split | Macro-F1, all 6 expert classes | Macro-F1, 4 training classes |")
    lines.append("| --- | --- | --- | --- |")
    for split in ("test", "cross_site"):
        lines.append(
            "| majority | "
            + split
            + f" | {_fmt(_get(results, 'majority', split, 'usage', 'macro_f1'))} | "
            + f"{_fmt(_get(results, 'majority', split, 'usage_class', 'macro_f1'))} |"
        )
        router = _get(results, "router", "on_expert_attributes", split)
        lines.append(
            "| rule router | "
            + split
            + f" | {_fmt(_get(router, 'macro_f1'))} | "
            + f"{_fmt(_get(router, 'usage_class_agreement', 'macro_f1'))} |"
        )
        for name, label in (("logistic_regression", "logistic regression"), ("lightgbm", "lightgbm")):
            scored = _get(block, "models", name, "splits", split)
            lines.append(
                f"| {label} | {split} | {_fmt(_get(scored, 'macro_f1'))} | "
                + f"{_fmt(_get(scored, 'usage_class', 'macro_f1'))} |"
            )
    lines.append("")
    lines.append(
        "Permutation importance is the drop in validation macro-F1, all 6 expert classes, when one feature is shuffled. "
        f"Repeats: {_fmt(_get(block, 'models', 'logistic_regression', 'permutation_importance', 'repeats'))}."
    )
    lines.append("")
    lines.append("| Model | Feature | Mean drop in macro-F1, all 6 expert classes | Std |")
    lines.append("| --- | --- | --- | --- |")
    for name, label in (("logistic_regression", "logistic regression"), ("lightgbm", "lightgbm")):
        features = _get(block, "models", name, "permutation_importance", "features")
        if not isinstance(features, list):
            lines.append(f"| {label} | not computed | not computed | not computed |")
            continue
        for row in features:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| {label} | {row.get('feature')} | {_fmt(row.get('mean_macro_f1_drop'))} | "
                + f"{_fmt(row.get('std_macro_f1_drop'))} |"
            )
    lines.append("")
    return lines


def _triple(block: Any) -> str:
    return (
        f"{_fmt(_get(block, 'balanced_accuracy'))} | {_fmt(_get(block, 'macro_f1'))} | {_fmt(_get(block, 'accuracy'))}"
    )


def _six_cells(usage: Any, usage_class: Any) -> str:
    return f"{_triple(usage)} | {_triple(usage_class)}"


def _prior_text(prior: Any, counts: Any) -> str:
    if not isinstance(prior, dict):
        return "not computed"
    labels = [label for label in USAGE_LABELS if label in prior]
    labels.extend(label for label in prior if label not in labels)
    parts = []
    for label in labels:
        share = prior[label]
        text = f"{label} {_fmt(share)}"
        if isinstance(counts, dict):
            text += f" (n={_fmt(counts.get(label))})"
        parts.append(text)
    return ", ".join(parts)


def _matrix_lines(matrix: Any) -> list[str]:
    if not isinstance(matrix, dict):
        return ["Route matrix: not computed.", ""]
    labels = matrix.get("labels")
    rows = matrix.get("matrix")
    if not isinstance(labels, list) or not isinstance(rows, list) or not rows:
        return ["Route matrix: not computed.", ""]
    lines = [
        "Rows are the Wargön usage label. Columns are the Myrorna usage label. "
        "A garment with two Wargön annotations and one Myrorna annotation contributes two cells.",
        "",
        "| Wargön route | " + " | ".join(str(label) for label in labels) + " |",
        "| --- | " + " | ".join("---" for _ in labels) + " |",
    ]
    for label, row in zip(labels, rows):
        if not isinstance(row, list):
            continue
        lines.append("| " + str(label) + " | " + " | ".join(_fmt(cell) for cell in row) + " |")
    lines.append("")
    return lines


def _site_agreement_lines(results: dict) -> list[str]:
    """Cross-site expert labels against within-Wargön expert labels on the same garments."""
    site = _get(results, "gold", "site_agreement")
    if not isinstance(site, dict):
        return ["", "Cross-site expert comparison: not computed.", ""]
    usage_cross = _get(site, "cross_site", "usage")
    usage_within = _get(site, "within_wargon", "usage")
    condition_cross = _get(site, "cross_site", "condition")
    condition_within = _get(site, "within_wargon", "condition")
    view = _get(site, "view_versus_route")
    lines = [
        "",
        (
            f"Numbered gold garments annotated at both Wargön and Myrorna: {_fmt(_get(site, 'n_garments'))}. "
            "Wargön is station 1 and station 2. Myrorna is station 3. "
            "Cross-site pairs join every Wargön annotation to every Myrorna annotation of the same garment. "
            "Within-Wargön pairs are unordered pairs of Wargön annotations on those same garments."
        ),
        (
            f"Usage Cohen's kappa, cross-site {_fmt(_get(usage_cross, 'cohen_kappa'))} "
            f"(raw agreement {_fmt(_get(usage_cross, 'raw_agreement'))}, "
            f"pairs {_fmt(_get(usage_cross, 'n_pairs'))}), "
            f"within Wargön {_fmt(_get(usage_within, 'cohen_kappa'))} "
            f"(raw agreement {_fmt(_get(usage_within, 'raw_agreement'))}, "
            f"pairs {_fmt(_get(usage_within, 'n_pairs'))})."
        ),
        (
            f"Condition quadratic weighted kappa, cross-site {_fmt(_get(condition_cross, 'cohen_kappa_quadratic'))}, "
            f"within Wargön {_fmt(_get(condition_within, 'cohen_kappa_quadratic'))}. "
            f"Mean condition difference, Myrorna minus Wargön, "
            f"{_fmt(_get(condition_cross, 'mean_difference_myrorna_minus_wargon'))}. "
            f"Mean absolute condition difference, cross-site "
            f"{_fmt(_get(condition_cross, 'mean_absolute_difference'))}, "
            f"within Wargön {_fmt(_get(condition_within, 'mean_absolute_difference'))}."
        ),
        "",
    ]
    lines.extend(_matrix_lines(_get(usage_cross, "confusion_matrix")))
    lines.append(
        "Same view means the condition scores are equal. Same route means the usage labels are equal. "
        f"Pairs with both fields present: {_fmt(_get(view, 'n_pairs'))}."
    )
    lines.append("")
    lines.append("| | Same route | Different route |")
    lines.append("| --- | --- | --- |")
    lines.append(
        "| Same condition | "
        f"{_fmt(_get(view, 'same_view_same_route'))} | {_fmt(_get(view, 'same_view_different_route'))} |"
    )
    lines.append(
        "| Different condition | "
        f"{_fmt(_get(view, 'different_view_same_route'))} | {_fmt(_get(view, 'different_view_different_route'))} |"
    )
    lines.append("")
    lines.append(
        "Same condition and a different route: the sites give the garment the same condition score and a different usage label."
    )
    lines.append("Different condition: the sites give the garment different condition scores.")
    lines.append("")
    return lines


def _station3_lines(results: dict) -> list[str]:
    """Station 3 usage with balanced accuracy and macro-F1 before accuracy."""
    lines = ["", "### Station 3", ""]
    block = _get(results, "station3")
    lines.append(
        "On station 3 the primary metrics are balanced accuracy and macro-F1. Accuracy is secondary. "
        "Balanced accuracy is the mean recall over classes that occur in the scored rows."
    )
    lines.append("")
    lines.append("Full station 3 split, uncorrected.")
    lines.append("")
    lines.append(
        "| Model | Balanced accuracy, 6 expert classes | Macro-F1, all 6 expert classes | Accuracy, 6 expert classes | "
        "Balanced accuracy, 4 training classes | Macro-F1, 4 training classes | Accuracy, 4 training classes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    full_rows = [
        (
            "majority",
            _get(results, "majority", "cross_site", "usage"),
            _get(results, "majority", "cross_site", "usage_class"),
        ),
        (
            "rule router",
            _get(results, "router", "on_expert_attributes", "cross_site"),
            _get(results, "router", "on_expert_attributes", "cross_site", "usage_class_agreement"),
        ),
        (
            "logistic regression",
            _get(results, "attribute_baselines", "models", "logistic_regression", "splits", "cross_site"),
            _get(results, "attribute_baselines", "models", "logistic_regression", "splits", "cross_site", "usage_class"),
        ),
        (
            "lightgbm",
            _get(results, "attribute_baselines", "models", "lightgbm", "splits", "cross_site"),
            _get(results, "attribute_baselines", "models", "lightgbm", "splits", "cross_site", "usage_class"),
        ),
    ]
    for label, usage, usage_class in full_rows:
        lines.append(f"| {label} | {_six_cells(usage, usage_class)} |")
    lines.append("")
    if not isinstance(block, dict):
        lines.append("Label-shift comparison: not computed.")
        lines.append("")
        return lines
    note = _get(block, "note")
    if isinstance(note, str) and note:
        lines.append(note)
        lines.append("")
    lines.append(
        f"Prior sample size {_fmt(_get(block, 'prior_sample_size'))}. "
        f"Seed {_fmt(_get(block, 'prior_sample_seed'))}. "
        f"Scored rows {_fmt(_get(block, 'n_scored'))}. "
        f"Station 3 rows before the hold-out {_fmt(_get(block, 'n_cross_site'))}."
    )
    lines.append(
        "Training class prior: " + _prior_text(_get(block, "source_prior"), _get(block, "source_prior_counts")) + "."
    )
    lines.append(
        "Station 3 sample class prior: "
        + _prior_text(_get(block, "target_prior"), _get(block, "target_prior_counts"))
        + "."
    )
    lines.append("")
    lines.append(
        "Laplace-smoothed station 3 prior: "
        + _prior_text(_get(block, "target_prior_laplace"), None)
        + f". Alpha {_fmt(_get(block, 'laplace_alpha'))}."
    )
    lines.append("")
    lines.append("Station 3 excluding the prior sample. Uncorrected, Laplace, and EM use these rows.")
    lines.append("")
    lines.append(
        "| Model | Variant | Balanced accuracy, 6 expert classes | Macro-F1, all 6 expert classes | Accuracy, 6 expert classes | "
        "Balanced accuracy, 4 training classes | Macro-F1, 4 training classes | Accuracy, 4 training classes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, label in (
        ("majority", "majority"),
        ("rule_router", "rule router"),
        ("logistic_regression", "logistic regression"),
        ("lightgbm", "lightgbm"),
    ):
        for variant, variant_label in (
            ("uncorrected", "uncorrected"),
            ("label_shift_laplace", "Laplace prior"),
            ("label_shift_em", "EM prior (no labels)"),
        ):
            scored = _get(block, "models", name, variant)
            lines.append(f"| {label} | {variant_label} | {_six_cells(scored, _get(scored, 'usage_class'))} |")
        undefined = _get(block, "models", name, "n_undefined_reweight")
        if isinstance(undefined, dict):
            for variant, count in undefined.items():
                if isinstance(count, int) and count > 0:
                    lines.append(f"{label} rows whose reweighted mass was 0 ({variant}): {_fmt(count)}.")
        elif isinstance(undefined, int) and undefined > 0:
            lines.append(f"{label} rows whose reweighted mass was 0: {_fmt(undefined)}.")
    lines.append("")
    lines.append(
        "EM prior from each model's p(y|x) on the scored rows. "
        + str(_get(block, "em_reference") or "Saerens, Latinne, and Decaestecker, 2002")
        + ". No station 3 labels."
    )
    lines.append("")
    lines.append("| Model | Iterations | Converged | Prior |")
    lines.append("| --- | --- | --- | --- |")
    for name, label in (
        ("majority", "majority"),
        ("rule_router", "rule router"),
        ("logistic_regression", "logistic regression"),
        ("lightgbm", "lightgbm"),
    ):
        lines.append(
            f"| {label} | {_fmt(_get(block, 'models', name, 'em_iterations'))} | "
            f"{_fmt(_get(block, 'models', name, 'em_converged'))} | "
            + _prior_text(_get(block, "models", name, "em_prior"), None)
            + " |"
        )
    lines.append("")
    return lines


def render_results(results: dict | None, eda: dict | None) -> str:
    """Markdown for the results section. Every figure is read from the JSON arguments."""
    if results is None or results.get("smoke") is True or results.get("dataset") != "clothing_v3":
        return (
            "No clothing_v3 `reports/results.json` has been written yet. "
            "Run `python -m garmentroute.evaluate` after `scripts/build_index.py`. "
            "This section is filled from that file and does not contain hand-typed scores.\n"
        )
    lines = [
        "Floats below are read from `reports/results.json` and `reports/eda_summary.json` and shown to 3 decimal places. The JSON files keep full precision.",
        "",
        f"Rows in the index: {_fmt(_get(eda, 'n_rows'))}.",
        _gold_split_line(eda),
        f"Stratified split: {_fmt(_get(results, 'split_meta', 'stratified'))}. Seed {_fmt(_get(results, 'seed'))}.",
        "",
    ]
    if isinstance(eda, dict) and "duplicate_train_test_dropped" in eda:
        lines.extend(
            [
                f"Near-duplicate fronts removed: {_fmt(_get(eda, 'duplicate_n'))}. "
                f"Test copies dropped because that front is also in train: {_fmt(eda.get('duplicate_train_test_dropped'))}.",
                "",
            ]
        )
    lines.extend(
        [
        "Macro-F1, all 6 expert classes, averages F1 over Reuse, Export, Recycle, Repair, Remake, and Energy recovery.",
        "Macro-F1, 4 training classes, averages F1 over Reuse, Export, Recycle, and Other, where Other groups Repair, Remake, and Energy recovery.",
        "",
        "| Task | Split | Accuracy | Macro-F1 | MAE | Within 1 |",
        "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    task_label = {"usage": "6 expert classes", "usage_class": "4 training classes"}
    for task in ("usage_class", "usage", "stains_3class", "holes", "condition", "pilling"):
        for split in ("test", "cross_site"):
            block = _get(results, "majority", split, task)
            lines.append(
                "| majority "
                + task_label.get(task, task)
                + f" | {split} | {_fmt(_get(block, 'accuracy'))} | {_fmt(_get(block, 'macro_f1'))} | "
                + f"{_fmt(_get(block, 'mae'))} | {_fmt(_get(block, 'within_1_accuracy'))} |"
            )
    cross_label = _get(results, "majority", "cross_site", "usage_class", "majority_label")
    cross_acc = _get(results, "majority", "cross_site", "usage_class", "accuracy")
    cross_f1 = _get(results, "majority", "cross_site", "usage_class", "macro_f1")
    lines.extend(
        [
            "",
            f"Cross-site majority label: {cross_label}. "
            f"Accuracy { _fmt(cross_acc) } is that label's share of station 3, not a trained model. "
            f"Macro-F1 on the same split is { _fmt(cross_f1) }.",
            "Station 3 usage is reported again in the station 3 section. "
            "There, balanced accuracy and macro-F1 come before accuracy.",
            "",
        ]
    )
    lines.extend(["", "### Gold set", ""])
    lines.extend(_gold_census_lines(eda))
    expert = _get(results, "gold", "expert_vs_expert_usage")
    condition = _get(results, "gold", "expert_vs_expert_condition")
    lines.append(
        f"Expert versus expert, usage: Cohen's kappa {_fmt(_get(expert, 'cohen_kappa'))}, "
        f"raw agreement {_fmt(_get(expert, 'raw_agreement'))}, "
        f"pairs {_fmt(_get(expert, 'n_pairs'))}. "
        f"Those pairs come from {_fmt(_get(expert, 'n_items_with_pairs'))} of "
        f"{_fmt(_get(expert, 'n_items'))} numbered garments. "
        "A garment with fewer than two numbered annotations contributes no pair. "
        "Lowercase `test` annotations are not in these pairs."
    )
    lines.append(
        f"Expert versus expert, condition: Cohen's kappa {_fmt(_get(condition, 'cohen_kappa'))}, "
        f"quadratic weighted kappa {_fmt(_get(condition, 'cohen_kappa_quadratic'))}."
    )
    lines.extend(_site_agreement_lines(results))
    model_kappa = _get(results, "gold", "model_vs_expert", "usage_cohen_kappa")
    if model_kappa is None:
        lines.append("Model versus expert kappa: not computed.")
    else:
        expert_kappa = _get(expert, "cohen_kappa")
        lines.append(f"Model versus expert, usage Cohen's kappa: {_fmt(model_kappa)}.")
        if isinstance(model_kappa, float) and isinstance(expert_kappa, float):
            gap = expert_kappa - model_kappa
            lines.append(f"Expert kappa minus model kappa: {_fmt(gap)}.")
            if gap <= 0.05:
                lines.append("The model kappa is within 0.05 of the expert-versus-expert kappa.")
            else:
                lines.append("The model kappa is not within 0.05 of the expert-versus-expert kappa.")
    lines.extend(["", "### Router on the expert damage labels", ""])
    lines.append(
        "These scores give the router the sorter's own condition and damage labels, "
        "plus the dataset material field as a mock passport. They are not a vision result."
    )
    lines.append("")
    lines.append("| Setting | Split | Accuracy | Macro-F1 |")
    lines.append("| --- | --- | --- | --- |")
    for split in ("test", "cross_site"):
        for suffix, label in (("", "unconstrained"), ("_capacity", "capacity")):
            block = _get(results, "router", "on_expert_attributes", f"{split}{suffix}")
            lines.append(
                f"| {label} | {split} | {_fmt(_get(block, 'accuracy'))} | {_fmt(_get(block, 'macro_f1'))} |"
            )
    lines.extend(_attribute_baseline_lines(results))
    lines.extend(_station3_lines(results))
    direct = _get(results, "fusion", "splits", "test", "after_temperature", "usage", "accuracy")
    routed = _get(
        results, "router", "on_predicted_attributes", "test", "usage_class_agreement", "accuracy"
    )
    lines.extend(["", "### Vision model", ""])
    if _get(results, "fusion") is None and _get(results, "front") is None:
        lines.append("Front-only and two-view checkpoints have not been scored.")
    else:
        for name in ("front", "fusion"):
            block = _get(results, name)
            if block is None:
                lines.append(f"{name}: not computed.")
                continue
            for split in ("test", "cross_site"):
                usage = _get(block, "splits", split, "after_temperature", "usage")
                lines.append(
                    f"{name} {split} usage accuracy {_fmt(_get(usage, 'accuracy'))}, "
                    f"macro-F1 {_fmt(_get(usage, 'macro_f1'))}."
                )
                ece_before = _get(block, "splits", split, "before_temperature", "calibration", "ece")
                ece_after = _get(block, "splits", split, "after_temperature", "calibration", "ece")
                lines.append(f"{name} {split} ECE before {_fmt(ece_before)}, after {_fmt(ece_after)}.")
    lines.append(
        f"Direct test usage-class accuracy: {_fmt(direct)}. "
        f"Router-from-predicted-attributes test usage-class accuracy: {_fmt(routed)}."
    )
    return "\n".join(lines) + "\n"


def splice_readme(readme: str, section: str) -> str:
    start = "<!-- RESULTS:START -->"
    end = "<!-- RESULTS:END -->"
    if start not in readme or end not in readme:
        raise ValueError("README is missing RESULTS markers")
    before, rest = readme.split(start, 1)
    _old, after = rest.split(end, 1)
    return f"{before}{start}\n{section}{end}{after}"


def sync(readme_path: Path, results_path: Path, eda_path: Path) -> None:
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else None
    eda = json.loads(eda_path.read_text(encoding="utf-8")) if eda_path.exists() else None
    text = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(splice_readme(text, render_results(results, eda)), encoding="utf-8")
