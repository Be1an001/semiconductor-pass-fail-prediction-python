"""Tests for imbalanced screening metrics."""

from __future__ import annotations

import pytest

from secom_ml.metrics import calculate_binary_classification_metrics


def test_calculate_binary_classification_metrics_counts_and_rates() -> None:
    y_true = [0, 0, 1, 1]
    y_score = [0.10, 0.60, 0.80, 0.30]

    metrics = calculate_binary_classification_metrics(
        y_true,
        y_score,
        threshold=0.50,
    )

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["review_count"] == 2
    assert metrics["review_rate"] == pytest.approx(0.50)
    assert metrics["precision"] == pytest.approx(0.50)
    assert metrics["recall"] == pytest.approx(0.50)
    assert metrics["f2"] == pytest.approx(0.50)
    assert metrics["roc_auc"] == pytest.approx(0.75)
