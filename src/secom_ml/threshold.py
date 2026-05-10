"""Threshold sweep and validation-only threshold selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from secom_ml.metrics import calculate_binary_classification_metrics


SUPPORTED_SELECTION_METRICS = {"balanced_accuracy", "f2"}


@dataclass(frozen=True)
class ThresholdSelection:
    """Selected threshold, selected metric, and full validation sweep."""

    threshold: float
    metric: str
    score: float
    row: dict[str, float | int]
    sweep: pd.DataFrame


def default_threshold_grid(
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    num_thresholds: int = 181,
) -> np.ndarray:
    """Return the notebook-style threshold grid."""

    return np.round(np.linspace(min_threshold, max_threshold, num_thresholds), 6)


def generate_threshold_sweep(
    y_true: object,
    y_score: object,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Calculate metrics for every threshold in a validation sweep."""

    threshold_values = (
        list(thresholds) if thresholds is not None else default_threshold_grid()
    )
    rows = [
        calculate_binary_classification_metrics(y_true, y_score, threshold)
        for threshold in threshold_values
    ]
    return pd.DataFrame(rows)


def select_threshold(
    y_true: object,
    y_score: object,
    metric: str = "balanced_accuracy",
    thresholds: Iterable[float] | None = None,
) -> ThresholdSelection:
    """Select a threshold from validation data only."""

    if metric not in SUPPORTED_SELECTION_METRICS:
        supported = ", ".join(sorted(SUPPORTED_SELECTION_METRICS))
        raise ValueError(f"Unsupported threshold metric: {metric}. Use {supported}.")

    sweep = generate_threshold_sweep(y_true, y_score, thresholds=thresholds)
    ranked = sweep.sort_values(
        by=[metric, "review_rate", "threshold"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    selected_row = ranked.iloc[0].to_dict()

    return ThresholdSelection(
        threshold=float(selected_row["threshold"]),
        metric=metric,
        score=float(selected_row[metric]),
        row=selected_row,
        sweep=sweep,
    )
