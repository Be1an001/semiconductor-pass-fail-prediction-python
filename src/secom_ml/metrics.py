"""Metric helpers for imbalanced SECOM pass/fail screening."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_classification_metrics(
    y_true: object,
    y_score: object,
    threshold: float = 0.50,
) -> dict[str, float | int]:
    """Calculate screening metrics at a probability threshold."""

    y_true_array = np.asarray(y_true).astype(int)
    y_score_array = np.asarray(y_score, dtype=float)
    y_pred = (y_score_array >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        y_pred,
        labels=[0, 1],
    ).ravel()
    total = len(y_true_array)
    review_count = int(tp + fp)
    specificity = _safe_specificity(tn=tn, fp=fp)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_array, y_pred)
        ),
        "specificity": specificity,
        "precision": float(
            precision_score(y_true_array, y_pred, zero_division=0)
        ),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "f2": float(
            fbeta_score(y_true_array, y_pred, beta=2.0, zero_division=0)
        ),
        "roc_auc": _safe_score(roc_auc_score, y_true_array, y_score_array),
        "pr_auc": _safe_score(
            average_precision_score,
            y_true_array,
            y_score_array,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "review_count": review_count,
        "review_rate": float(review_count / total) if total else 0.0,
    }


def _safe_specificity(tn: int, fp: int) -> float:
    denominator = tn + fp
    if denominator == 0:
        return 0.0
    return float(tn / denominator)


def _safe_score(
    scorer: Callable[[object, object], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    try:
        return float(scorer(y_true, y_score))
    except ValueError:
        return float("nan")
