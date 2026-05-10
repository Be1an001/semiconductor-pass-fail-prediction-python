"""Run the final holdout evaluation for the selected SECOM Random Forest model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from secom_ml.data import load_secom_data
from secom_ml.metrics import calculate_binary_classification_metrics
from secom_ml.models import build_random_forest
from secom_ml.plots import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from secom_ml.preprocessing import (
    apply_missingness_selection,
    fit_tree_preprocessor,
    select_feature_columns_by_missingness,
    transform_tree_preprocessor,
)
from secom_ml.splitting import create_train_validation_test_split


DEFAULT_CONFIG = Path("configs/final_rf.yaml")
FINAL_METRIC_COLUMNS = [
    "selected_experiment_name",
    "threshold",
    "accuracy",
    "balanced_accuracy",
    "specificity",
    "precision",
    "recall",
    "f1",
    "f2",
    "roc_auc",
    "pr_auc",
    "tn",
    "fp",
    "fn",
    "tp",
    "review_count",
    "review_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the validation-selected SECOM Random Forest model once "
            "on the untouched holdout test set."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the final Random Forest YAML config.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configs.") from exc

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_yaml_config(config_path)
    selected_row = select_final_candidate(config)
    source_config = load_source_experiment_config(config)

    result = evaluate_final_model(
        config=config,
        source_config=source_config,
        selected_row=selected_row,
        config_path=config_path,
    )
    write_final_outputs(result=result, config=config)
    log_final_evaluation(result=result, config=config)
    print_final_summary(result)
    return 0


def select_final_candidate(config: dict[str, Any]) -> pd.Series:
    selection_config = config.get("selection", {})
    metrics_path = _project_path(
        selection_config.get(
            "validation_metrics_path",
            "outputs/metrics/validation_metrics.csv",
        )
    )
    if not metrics_path.exists():
        raise FileNotFoundError(
            "Validation metrics are required before final evaluation: "
            f"{metrics_path}"
        )

    validation_metrics = pd.read_csv(metrics_path)
    override_name = selection_config.get("selected_experiment_name")
    if override_name:
        candidates = validation_metrics[
            validation_metrics["experiment_name"] == override_name
        ].copy()
    else:
        candidates = validation_metrics[
            validation_metrics["model_type"]
            == selection_config.get("candidate_model_type", "random_forest")
        ].copy()
        if bool(selection_config.get("require_validation_threshold", True)):
            candidates = candidates[
                candidates["threshold_source"].astype(str).str.startswith("validation_")
            ].copy()

    if candidates.empty:
        raise ValueError("No validation candidate matched the final selection rules.")

    primary_metric = selection_config.get("primary_metric", "f2")
    sort_columns = [primary_metric]
    ascending = [False]
    for tie_breaker in selection_config.get("tie_breakers", []):
        sort_columns.append(tie_breaker["metric"])
        ascending.append(tie_breaker.get("direction", "desc") == "asc")
    sort_columns.append("experiment_name")
    ascending.append(True)

    ranked = candidates.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="mergesort",
    )
    return ranked.iloc[0]


def load_source_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    source_path = _project_path(
        config.get("selection", {}).get(
            "source_experiments_config",
            "configs/rf_experiments.yaml",
        )
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Source experiment config not found: {source_path}")
    return load_yaml_config(source_path)


def evaluate_final_model(
    config: dict[str, Any],
    source_config: dict[str, Any],
    selected_row: pd.Series,
    config_path: Path,
) -> dict[str, Any]:
    seed = int(config.get("random_seed", 42))
    X, y, _metadata = load_secom_data(data_dir=_data_dir_from_config(config))

    split_config = config.get("split", {})
    splits = create_train_validation_test_split(
        X=X,
        y=y,
        test_size=float(split_config.get("test_size", 0.20)),
        validation_size=float(split_config.get("validation_size", 0.20)),
        random_state=int(split_config.get("random_state", seed)),
        stratify=bool(split_config.get("stratify", True)),
    )

    preprocessing_config = config.get("preprocessing", {})
    missingness_selection = select_feature_columns_by_missingness(
        splits.X_train,
        missingness_threshold=float(
            preprocessing_config.get("missingness_threshold", 0.50)
        ),
    )
    X_train_keep = apply_missingness_selection(
        splits.X_train,
        missingness_selection,
    )
    X_test_keep = apply_missingness_selection(
        splits.X_test,
        missingness_selection,
    )
    tree_preprocessor, X_train_tree = fit_tree_preprocessor(
        X_train_keep,
        strategy=preprocessing_config.get("imputation_strategy", "median"),
    )
    X_test_tree = transform_tree_preprocessor(X_test_keep, tree_preprocessor)

    selected_experiment = find_source_experiment(
        source_config=source_config,
        experiment_name=str(selected_row["experiment_name"]),
    )
    model = build_selected_model(
        selected_experiment=selected_experiment,
        source_config=source_config,
        X_train_tree=X_train_tree,
        y_train=splits.y_train,
        seed=seed,
    )
    model.fit(X_train_tree, splits.y_train)

    threshold = float(
        selected_row[config.get("selection", {}).get("threshold_column", "threshold")]
    )
    test_prob = positive_class_probability(model, X_test_tree)
    metrics = calculate_binary_classification_metrics(
        splits.y_test,
        test_prob,
        threshold=threshold,
    )

    output_row = {
        "selected_experiment_name": selected_row["experiment_name"],
        "model_type": selected_row["model_type"],
        "threshold_source": selected_row["threshold_source"],
        "selection_metric": config.get("selection", {}).get("primary_metric", "f2"),
        **metrics,
    }

    output_paths = resolve_output_paths(config)
    write_figures(
        output_paths=output_paths,
        y_test=splits.y_test,
        test_prob=test_prob,
        threshold=threshold,
        model=model,
        feature_columns=tree_preprocessor.feature_columns,
    )
    feature_importance = write_feature_importance(
        output_path=output_paths["final_feature_importance"],
        model=model,
        feature_columns=tree_preprocessor.feature_columns,
    )

    return {
        "metrics": output_row,
        "selected_validation_row": selected_row.to_dict(),
        "selected_experiment": selected_experiment,
        "output_paths": output_paths,
        "config_path": str(config_path),
        "train_rows": len(splits.y_train),
        "validation_rows_reserved": len(splits.y_validation),
        "test_rows": len(splits.y_test),
        "kept_features": len(missingness_selection.keep_columns),
        "dropped_features": len(missingness_selection.drop_columns),
        "feature_importance": feature_importance,
    }


def find_source_experiment(
    source_config: dict[str, Any],
    experiment_name: str,
) -> dict[str, Any]:
    for experiment in source_config.get("experiments", []):
        if experiment.get("name") == experiment_name:
            return experiment
    raise ValueError(f"Selected experiment is missing from source config: {experiment_name}")


def build_selected_model(
    selected_experiment: dict[str, Any],
    source_config: dict[str, Any],
    X_train_tree: Any,
    y_train: pd.Series,
    seed: int,
) -> Any:
    if selected_experiment.get("search") == "random_search":
        return fit_random_search_model(
            source_config=source_config,
            X_train_tree=X_train_tree,
            y_train=y_train,
            seed=seed,
        )

    params = dict(selected_experiment.get("params", {}))
    params.setdefault("random_state", seed)
    params.setdefault("n_jobs", -1)
    return build_random_forest(**params)


def fit_random_search_model(
    source_config: dict[str, Any],
    X_train_tree: Any,
    y_train: pd.Series,
    seed: int,
) -> Any:
    search_config = source_config.get("random_search", {})
    cv = StratifiedKFold(
        n_splits=int(search_config.get("cv_splits", 3)),
        shuffle=True,
        random_state=int(search_config.get("random_state", seed)),
    )
    search = RandomizedSearchCV(
        estimator=build_random_forest(random_state=seed, n_jobs=-1),
        param_distributions=search_config.get("params", {}),
        n_iter=int(search_config.get("n_iter", 8)),
        scoring=make_scorer(fbeta_score, beta=2.0, zero_division=0),
        cv=cv,
        random_state=int(search_config.get("random_state", seed)),
        n_jobs=int(search_config.get("n_jobs", 1)),
        refit=True,
    )
    search.fit(X_train_tree, y_train)
    return search.best_estimator_


def positive_class_probability(model: Any, X: Any) -> Any:
    probabilities = model.predict_proba(X)
    class_labels = list(model.classes_)
    if 1 not in class_labels:
        raise ValueError("Model does not expose probability for positive class 1.")
    return probabilities[:, class_labels.index(1)]


def write_final_outputs(result: dict[str, Any], config: dict[str, Any]) -> None:
    output_paths = result["output_paths"]
    metrics_path = output_paths["final_test_metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result["metrics"]]).to_csv(metrics_path, index=False)


def write_figures(
    output_paths: dict[str, Path],
    y_test: pd.Series,
    test_prob: Any,
    threshold: float,
    model: Any,
    feature_columns: list[str],
) -> None:
    import matplotlib.pyplot as plt

    figures_dir = output_paths["confusion_matrix"].parent
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, _ax = plot_confusion_matrix(
        y_test,
        test_prob,
        threshold=threshold,
        title="Final Holdout Confusion Matrix - Random Forest",
        output_path=output_paths["confusion_matrix"],
    )
    plt.close(fig)

    fig, _ax = plot_roc_curve(
        y_test,
        test_prob,
        title="Final Holdout ROC Curve - Random Forest",
        output_path=output_paths["roc_curve"],
    )
    plt.close(fig)

    fig, _ax = plot_precision_recall_curve(
        y_test,
        test_prob,
        title="Final Holdout Precision-Recall Curve - Random Forest",
        output_path=output_paths["pr_curve"],
    )
    plt.close(fig)

    fig, _ax = plot_feature_importance(
        feature_names=feature_columns,
        importances=model.feature_importances_,
        top_n=20,
        title="Final Random Forest Feature Importances",
        output_path=output_paths["feature_importance"],
    )
    plt.close(fig)


def write_feature_importance(
    output_path: Path,
    model: Any,
    feature_columns: list[str],
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.insert(0, "rank", importance.index + 1)
    importance.head(20).to_csv(output_path, index=False)
    return importance


def log_final_evaluation(result: dict[str, Any], config: dict[str, Any]) -> None:
    mlflow_config = config.get("mlflow", {})
    if not bool(mlflow_config.get("enabled", True)):
        return

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for final evaluation tracking. "
            "Install project requirements or set mlflow.enabled to false."
        ) from exc

    mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "sqlite:///mlflow.db"))
    mlflow.set_experiment(
        mlflow_config.get("experiment_name", "secom-pass-fail-screening")
    )
    output_paths = result["output_paths"]

    with mlflow.start_run(
        run_name=mlflow_config.get("run_name", "final_holdout_evaluation")
    ):
        mlflow.set_tags(
            {
                "phase": "phase_3_final_holdout",
                "split": "test",
                "test_set_used_for_selection": "false",
                "project_positioning": "mlops_lite_portfolio",
            }
        )
        mlflow.log_params(
            _stringify_params(
                {
                    "selected_experiment_name": result["metrics"][
                        "selected_experiment_name"
                    ],
                    "threshold_source": result["metrics"]["threshold_source"],
                    "threshold": result["metrics"]["threshold"],
                    "config_path": result["config_path"],
                    "train_rows": result["train_rows"],
                    "validation_rows_reserved": result["validation_rows_reserved"],
                    "test_rows": result["test_rows"],
                    "kept_features": result["kept_features"],
                    "dropped_features": result["dropped_features"],
                }
            )
        )
        final_metrics = {
            key: value
            for key, value in result["metrics"].items()
            if key in FINAL_METRIC_COLUMNS and _is_finite_number(value)
        }
        mlflow.log_metrics(final_metrics)
        for artifact_path in output_paths.values():
            if artifact_path.exists() and artifact_path.is_file():
                mlflow.log_artifact(str(artifact_path))


def print_final_summary(result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    print("Final holdout evaluation completed.")
    print(f"Selected validation experiment: {metrics['selected_experiment_name']}")
    print(f"Validation-selected threshold: {metrics['threshold']:.3f}")
    print(
        "Test recall={recall:.4f}, F2={f2:.4f}, balanced_accuracy={bal:.4f}, "
        "PR-AUC={pr_auc:.4f}".format(
            recall=metrics["recall"],
            f2=metrics["f2"],
            bal=metrics["balanced_accuracy"],
            pr_auc=metrics["pr_auc"],
        )
    )
    print(
        "Confusion counts: TP={tp}, FP={fp}, FN={fn}, TN={tn}; "
        "review_rate={review_rate:.4f}".format(**metrics)
    )
    print("Saved final metrics and figures. No model or threshold was selected on test data.")


def resolve_output_paths(config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("outputs", {})
    return {
        "final_test_metrics": _project_path(
            outputs.get("final_test_metrics", "outputs/metrics/final_test_metrics.csv")
        ),
        "final_feature_importance": _project_path(
            outputs.get(
                "final_feature_importance",
                "outputs/metrics/final_feature_importance.csv",
            )
        ),
        "confusion_matrix": _project_path(
            outputs.get("confusion_matrix", "outputs/figures/final_confusion_matrix.png")
        ),
        "roc_curve": _project_path(
            outputs.get("roc_curve", "outputs/figures/final_roc_curve.png")
        ),
        "pr_curve": _project_path(
            outputs.get("pr_curve", "outputs/figures/final_pr_curve.png")
        ),
        "feature_importance": _project_path(
            outputs.get(
                "feature_importance",
                "outputs/figures/final_feature_importance.png",
            )
        ),
    }


def _data_dir_from_config(config: dict[str, Any]) -> Path | None:
    feature_path = config.get("data", {}).get("feature_path")
    if not feature_path:
        return None
    return _project_path(feature_path).parent


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _stringify_params(params: dict[str, Any]) -> dict[str, str | int | float | bool]:
    safe_params: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        if value is None:
            safe_params[key] = "null"
        elif isinstance(value, str | int | float | bool):
            safe_params[key] = value
        else:
            safe_params[key] = json.dumps(value, sort_keys=True, default=str)
    return safe_params


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


if __name__ == "__main__":
    raise SystemExit(main())
