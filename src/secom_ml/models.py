"""Model factory functions for SECOM baseline and Random Forest experiments."""

from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_dummy_classifier(strategy: str = "most_frequent") -> DummyClassifier:
    """Build a majority-class sanity baseline."""

    return DummyClassifier(strategy=strategy)


def build_logistic_regression_baseline(
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 5000,
    solver: str = "liblinear",
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """Build the class-weighted Logistic Regression baseline."""

    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
        **kwargs,
    )


def build_random_forest(
    n_estimators: int = 400,
    min_samples_leaf: int = 3,
    class_weight: str | dict[int, float] | None = "balanced_subsample",
    n_jobs: int = -1,
    random_state: int = 42,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Build the current Random Forest reference model."""

    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )


def build_random_forest_from_config(
    params: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Build a Random Forest from a parameter dictionary."""

    return build_random_forest(**(params or {}))
