"""Run validation-only SECOM Random Forest experiments with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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
from secom_ml.models import (
    build_dummy_classifier,
    build_logistic_regression_baseline,
    build_random_forest,
)
from secom_ml.preprocessing import (
    apply_missingness_selection,
    fit_linear_preprocessor,
    fit_tree_preprocessor,
    select_feature_columns_by_missingness,
    transform_linear_preprocessor,
    transform_tree_preprocessor,
)
from secom_ml.splitting import create_train_validation_test_split
from secom_ml.threshold import default_threshold_grid, generate_threshold_sweep


DEFAULT_CONFIG = Path("configs/rf_experiments.yaml")
METRIC_COLUMNS = [
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


@dataclass(frozen=True)
class ExperimentResult:
    """Validation result and threshold sweep for one experiment variant."""

    name: str
    model_type: str
    threshold_source: str
    metrics: dict[str, float | int | str]
    sweep: pd.DataFrame
    params: dict[str, Any]
    tags: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run validation-only SECOM Random Forest experiments and track "
            "metrics with local MLflow."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the Random Forest experiment YAML config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned experiment names without training models.",
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
    experiments = config.get("experiments", [])
    experiment_names = [item.get("name", "<unnamed>") for item in experiments]
    if args.dry_run:
        print("Configured experiments:")
        for name in experiment_names:
            print(f"- {name}")
        return 0

    results = run_experiments(config=config, config_path=config_path)
    write_outputs(results=results, config=config)
    print_run_summary(results)
    return 0


def run_experiments(
    config: dict[str, Any],
    config_path: Path,
) -> list[ExperimentResult]:
    seed = int(config.get("random_seed", 42))
    data_dir = _data_dir_from_config(config)
    X, y, _metadata = load_secom_data(data_dir=data_dir)

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
    X_validation_keep = apply_missingness_selection(
        splits.X_validation,
        missingness_selection,
    )

    imputation_strategy = preprocessing_config.get("imputation_strategy", "median")
    tree_preprocessor, X_train_tree = fit_tree_preprocessor(
        X_train_keep,
        strategy=imputation_strategy,
    )
    X_validation_tree = transform_tree_preprocessor(
        X_validation_keep,
        tree_preprocessor,
    )

    linear_config = preprocessing_config.get("linear_path", {})
    linear_preprocessor, X_train_linear = fit_linear_preprocessor(
        X_train_keep,
        strategy=imputation_strategy,
        pca_variance=float(linear_config.get("pca_variance", 0.95)),
        random_state=seed,
    )
    X_validation_linear = transform_linear_preprocessor(
        X_validation_keep,
        linear_preprocessor,
    )

    thresholds = _threshold_grid_from_config(config)
    mlflow_client = configure_mlflow(config)
    random_search_model, random_search_details = fit_random_search_model(
        config=config,
        X_train_tree=X_train_tree,
        y_train=splits.y_train,
        seed=seed,
    )

    shared_context = {
        "config_path": str(config_path),
        "seed": seed,
        "train_rows": len(splits.y_train),
        "validation_rows": len(splits.y_validation),
        "test_rows_reserved": len(splits.y_test),
        "kept_features": len(missingness_selection.keep_columns),
        "dropped_features": len(missingness_selection.drop_columns),
        "linear_pca_components": int(X_train_linear.shape[1]),
        "missingness_threshold": missingness_selection.missingness_threshold,
    }

    results: list[ExperimentResult] = []
    for experiment in config.get("experiments", []):
        result = run_single_experiment(
            experiment=experiment,
            config=config,
            thresholds=thresholds,
            X_train_tree=X_train_tree,
            X_validation_tree=X_validation_tree,
            X_train_linear=X_train_linear,
            X_validation_linear=X_validation_linear,
            y_train=splits.y_train,
            y_validation=splits.y_validation,
            seed=seed,
            random_search_model=random_search_model,
            random_search_details=random_search_details,
            shared_context=shared_context,
        )
        results.append(result)
        log_mlflow_result(
            mlflow_client=mlflow_client,
            result=result,
            shared_context=shared_context,
        )

    return results


def run_single_experiment(
    experiment: dict[str, Any],
    config: dict[str, Any],
    thresholds: list[float],
    X_train_tree: Any,
    X_validation_tree: Any,
    X_train_linear: Any,
    X_validation_linear: Any,
    y_train: pd.Series,
    y_validation: pd.Series,
    seed: int,
    random_search_model: Any,
    random_search_details: dict[str, Any],
    shared_context: dict[str, Any],
) -> ExperimentResult:
    name = experiment["name"]
    model_type = experiment["model_type"]

    if model_type == "dummy_classifier":
        model = build_dummy_classifier()
        X_train_model = X_train_tree
        X_validation_model = X_validation_tree
    elif model_type == "logistic_regression_pca":
        model = build_logistic_regression_baseline(random_state=seed)
        X_train_model = X_train_linear
        X_validation_model = X_validation_linear
    elif experiment.get("search") == "random_search":
        model = random_search_model
        X_train_model = None
        X_validation_model = X_validation_tree
    elif model_type == "random_forest":
        params = _with_default_rf_params(experiment.get("params", {}), seed=seed)
        model = build_random_forest(**params)
        X_train_model = X_train_tree
        X_validation_model = X_validation_tree
    else:
        raise ValueError(f"Unsupported experiment model_type: {model_type}")

    if X_train_model is not None:
        model.fit(X_train_model, y_train)

    validation_prob = positive_class_probability(model, X_validation_model)
    threshold, threshold_source, selected_sweep = resolve_threshold(
        experiment=experiment,
        config=config,
        y_validation=y_validation,
        validation_prob=validation_prob,
        thresholds=thresholds,
    )
    metrics = calculate_binary_classification_metrics(
        y_validation,
        validation_prob,
        threshold=threshold,
    )
    metrics_row = {
        "experiment_name": name,
        "model_type": model_type,
        "threshold_source": threshold_source,
        **metrics,
    }

    model_params = model.get_params(deep=False)
    if experiment.get("search") == "random_search":
        model_params = {
            **model_params,
            "random_search_best_score": random_search_details["best_score"],
            **{
                f"random_search_best_{key}": value
                for key, value in random_search_details["best_params"].items()
            },
        }

    sweep = selected_sweep.copy()
    sweep.insert(0, "experiment_name", name)
    sweep.insert(1, "model_type", model_type)
    sweep["selected_threshold"] = sweep["threshold"].eq(threshold)
    sweep["threshold_source"] = threshold_source

    tags = {
        "phase": "phase_2_validation_experiments",
        "split": "validation",
        "test_set_used": "false",
        "project_positioning": "mlops_lite_portfolio",
    }

    return ExperimentResult(
        name=name,
        model_type=model_type,
        threshold_source=threshold_source,
        metrics=metrics_row,
        sweep=sweep,
        params={**shared_context, **model_params},
        tags=tags,
    )


def fit_random_search_model(
    config: dict[str, Any],
    X_train_tree: Any,
    y_train: pd.Series,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    search_config = config.get("random_search", {})
    base_model = build_random_forest(random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(
        n_splits=int(search_config.get("cv_splits", 3)),
        shuffle=True,
        random_state=int(search_config.get("random_state", seed)),
    )
    f2_scorer = make_scorer(fbeta_score, beta=2.0, zero_division=0)

    # The search is intentionally small and uses training folds only.
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_config.get("params", {}),
        n_iter=int(search_config.get("n_iter", 8)),
        scoring=f2_scorer,
        cv=cv,
        random_state=int(search_config.get("random_state", seed)),
        n_jobs=int(search_config.get("n_jobs", 1)),
        refit=True,
        verbose=0,
    )
    search.fit(X_train_tree, y_train)
    details = {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "cv_splits": cv.n_splits,
        "n_iter": int(search_config.get("n_iter", 8)),
        "scoring": search_config.get("scoring", "f2"),
    }
    return search.best_estimator_, details


def resolve_threshold(
    experiment: dict[str, Any],
    config: dict[str, Any],
    y_validation: pd.Series,
    validation_prob: Any,
    thresholds: list[float],
) -> tuple[float, str, pd.DataFrame]:
    sweep = generate_threshold_sweep(
        y_validation,
        validation_prob,
        thresholds=thresholds,
    )

    if "threshold" in experiment:
        return float(experiment["threshold"]), "fixed", sweep

    threshold_config = config.get("threshold_search", {})
    metric = threshold_config.get("selection_metric", "f2")
    ranked = sweep.sort_values(
        by=[metric, "review_rate", "threshold"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    selected = ranked.iloc[0]
    return float(selected["threshold"]), f"validation_{metric}", sweep


def positive_class_probability(model: Any, X: Any) -> Any:
    probabilities = model.predict_proba(X)
    class_labels = list(model.classes_)
    if 1 not in class_labels:
        raise ValueError("Model does not expose probability for positive class 1.")
    positive_index = class_labels.index(1)
    return probabilities[:, positive_index]


def configure_mlflow(config: dict[str, Any]):
    mlflow_config = config.get("mlflow", {})
    if not bool(mlflow_config.get("enabled", True)):
        return None

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for Phase 2 tracking. "
            "Install project requirements or set mlflow.enabled to false."
        ) from exc

    tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_config.get(
        "experiment_name",
        "secom-pass-fail-screening",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_mlflow_result(
    mlflow_client: Any,
    result: ExperimentResult,
    shared_context: dict[str, Any],
) -> None:
    if mlflow_client is None:
        return

    with mlflow_client.start_run(run_name=result.name):
        mlflow_client.set_tags(result.tags)
        mlflow_client.log_params(_stringify_params(result.params))
        mlflow_client.log_params(
            {
                "experiment_name": result.name,
                "model_type": result.model_type,
                "threshold_source": result.threshold_source,
                "test_set_used_for_selection": "false",
            }
        )
        numeric_metrics = {
            key: value
            for key, value in result.metrics.items()
            if key in METRIC_COLUMNS and _is_finite_number(value)
        }
        mlflow_client.log_metrics(numeric_metrics)
        mlflow_client.log_text(
            pd.DataFrame([result.metrics]).to_csv(index=False),
            artifact_file="validation_metrics.csv",
        )
        mlflow_client.log_text(
            result.sweep.to_csv(index=False),
            artifact_file="threshold_sweep.csv",
        )
        mlflow_client.log_dict(
            shared_context,
            artifact_file="run_context.json",
        )


def write_outputs(results: list[ExperimentResult], config: dict[str, Any]) -> None:
    outputs = config.get("outputs", {})
    metrics_path = Path(
        outputs.get("validation_metrics", "outputs/metrics/validation_metrics.csv")
    )
    threshold_sweep_path = Path(
        outputs.get("threshold_sweep", "outputs/metrics/threshold_sweep.csv")
    )
    improvement_path = Path(
        outputs.get("rf_improvement_table", "outputs/metrics/rf_improvement_table.csv")
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    validation_metrics = pd.DataFrame([result.metrics for result in results])
    validation_metrics.to_csv(metrics_path, index=False)

    threshold_sweep = pd.concat(
        [result.sweep for result in results],
        ignore_index=True,
    )
    threshold_sweep.to_csv(threshold_sweep_path, index=False)

    rf_improvement = validation_metrics[
        validation_metrics["experiment_name"].str.startswith("rf_")
    ].copy()
    ordered_columns = [
        "experiment_name",
        "threshold_source",
        "threshold",
        "recall",
        "f2",
        "balanced_accuracy",
        "pr_auc",
        "roc_auc",
        "tp",
        "fp",
        "fn",
        "tn",
        "review_count",
        "review_rate",
    ]
    rf_improvement = rf_improvement.loc[:, ordered_columns]
    rf_improvement.to_csv(improvement_path, index=False)


def print_run_summary(results: list[ExperimentResult]) -> None:
    summary = pd.DataFrame([result.metrics for result in results])
    columns = [
        "experiment_name",
        "threshold",
        "recall",
        "f2",
        "balanced_accuracy",
        "pr_auc",
        "tp",
        "fp",
        "fn",
        "tn",
        "review_rate",
    ]
    print("Validation experiment summary:")
    print(summary.loc[:, columns].to_string(index=False))
    print("\nSaved latest metrics CSV files under outputs/metrics.")
    print("No final test evaluation was run.")


def _data_dir_from_config(config: dict[str, Any]) -> Path | None:
    data_config = config.get("data", {})
    feature_path = data_config.get("feature_path")
    if not feature_path:
        return None
    return (PROJECT_ROOT / feature_path).resolve().parent


def _threshold_grid_from_config(config: dict[str, Any]) -> list[float]:
    threshold_config = config.get("threshold_search", {})
    return default_threshold_grid(
        min_threshold=float(threshold_config.get("min_threshold", 0.05)),
        max_threshold=float(threshold_config.get("max_threshold", 0.95)),
        num_thresholds=int(threshold_config.get("num_thresholds", 181)),
    ).tolist()


def _with_default_rf_params(params: dict[str, Any], seed: int) -> dict[str, Any]:
    normalized = dict(params or {})
    normalized.setdefault("random_state", seed)
    normalized.setdefault("n_jobs", -1)
    return normalized


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
