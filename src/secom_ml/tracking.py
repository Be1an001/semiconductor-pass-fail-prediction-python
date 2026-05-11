"""MLflow helper utilities for SECOM experiment tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
):
    """Configure MLflow without starting a run."""

    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    return mlflow


def log_params_and_metrics(
    params: Mapping[str, object] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> None:
    """Log params and metrics to the currently active MLflow run."""

    mlflow = _require_mlflow()
    if params:
        mlflow.log_params(dict(params))
    if metrics:
        mlflow.log_metrics(dict(metrics))


def log_artifacts(artifact_paths: list[str | Path]) -> None:
    """Log artifacts to the currently active MLflow run."""

    mlflow = _require_mlflow()
    for artifact_path in artifact_paths:
        path = Path(artifact_path)
        if path.is_file():
            mlflow.log_artifact(str(path))


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for experiment tracking. "
            "Install project requirements before using tracking helpers."
        ) from exc
    return mlflow
