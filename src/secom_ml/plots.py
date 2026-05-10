"""Plotting helpers for SECOM model evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)


def plot_confusion_matrix(
    y_true: object,
    y_score: object,
    threshold: float,
    title: str = "Confusion Matrix",
    output_path: str | Path | None = None,
):
    """Plot a confusion matrix at a selected threshold."""

    y_pred = [1 if score >= threshold else 0 for score in y_score]
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Pass (0)", "Fail (1)"],
    )
    display.plot(cmap="Blues")
    display.ax_.set_title(title)
    _save_if_requested(output_path)
    return display.figure_, display.ax_


def plot_roc_curve(
    y_true: object,
    y_score: object,
    title: str = "ROC Curve",
    output_path: str | Path | None = None,
):
    """Plot a ROC curve from predicted fail probabilities."""

    display = RocCurveDisplay.from_predictions(y_true, y_score)
    display.ax_.set_title(title)
    _save_if_requested(output_path)
    return display.figure_, display.ax_


def plot_precision_recall_curve(
    y_true: object,
    y_score: object,
    title: str = "Precision-Recall Curve",
    output_path: str | Path | None = None,
):
    """Plot a precision-recall curve from predicted fail probabilities."""

    display = PrecisionRecallDisplay.from_predictions(y_true, y_score)
    display.ax_.set_title(title)
    _save_if_requested(output_path)
    return display.figure_, display.ax_


def plot_feature_importance(
    feature_names: list[str],
    importances: object,
    top_n: int = 20,
    title: str = "Top Random Forest Feature Importances",
    output_path: str | Path | None = None,
):
    """Plot top model-important sensor variables without causal interpretation."""

    importance = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Sensor feature")
    fig.tight_layout()
    _save_if_requested(output_path)
    return fig, ax


def _save_if_requested(output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
