"""Tests for threshold sweeps and validation threshold selection."""

from __future__ import annotations

from secom_ml.threshold import generate_threshold_sweep, select_threshold


def test_generate_threshold_sweep_returns_one_row_per_threshold() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.05, 0.20, 0.60, 0.70, 0.90]
    thresholds = [0.20, 0.50, 0.80]

    sweep = generate_threshold_sweep(y_true, y_score, thresholds=thresholds)

    assert sweep["threshold"].tolist() == thresholds
    assert len(sweep) == len(thresholds)
    assert {"balanced_accuracy", "f2", "review_rate"}.issubset(sweep.columns)


def test_select_threshold_returns_valid_balanced_accuracy_threshold() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.05, 0.20, 0.60, 0.70, 0.90]
    thresholds = [0.20, 0.50, 0.80]

    selection = select_threshold(
        y_true,
        y_score,
        metric="balanced_accuracy",
        thresholds=thresholds,
    )

    assert selection.threshold in thresholds
    assert selection.metric == "balanced_accuracy"
    assert selection.score == selection.row["balanced_accuracy"]


def test_select_threshold_supports_f2_score() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_score = [0.05, 0.20, 0.60, 0.70, 0.90]
    thresholds = [0.20, 0.50, 0.80]

    selection = select_threshold(
        y_true,
        y_score,
        metric="f2",
        thresholds=thresholds,
    )

    assert selection.threshold in thresholds
    assert selection.metric == "f2"
