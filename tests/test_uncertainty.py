"""Calibration helpers do not look at a test split; these checks use synthetic scores."""

import numpy as np

from garmentroute.uncertainty import (
    coverage_accuracy,
    expected_calibration_error,
    fit_temperature,
    select_threshold,
)


def test_matching_confidence_and_accuracy_has_zero_ece():
    confidences = [0.5] * 10
    correct = [True, False] * 5
    report = expected_calibration_error(confidences, correct, n_bins=10)
    assert report["ece"] == 0


def test_coverage_curve_is_sorted_by_confidence():
    curve = coverage_accuracy([0.2, 0.9, 0.4], [False, True, False], fractions=(1 / 3, 1.0))
    assert curve[0]["accuracy"] == 1
    assert curve[-1]["coverage"] == 1


def test_threshold_is_chosen_only_from_the_rows_it_is_given():
    confidences = [0.9, 0.8, 0.2, 0.1]
    correct = [True, True, False, False]
    chosen = select_threshold(confidences, correct, min_coverage=0.5)
    assert chosen["threshold"] == 0.8
    assert chosen["accuracy"] == 1


def test_temperature_fit_clamps_away_from_zero():
    logits = np.asarray([[8.0, 0.0, -2.0, -3.0]] * 40, dtype=float)
    labels = [0] * 40
    temperature = fit_temperature(logits, labels, steps=50)
    assert 0.05 <= temperature <= 20.0

