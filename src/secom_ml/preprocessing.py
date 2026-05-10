"""Leakage-aware preprocessing helpers for SECOM tabular experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MissingnessSelection:
    """Feature selection learned from the training split only."""

    keep_columns: list[str]
    drop_columns: list[str]
    train_missing_ratio: pd.Series
    missingness_threshold: float


@dataclass
class TreePreprocessor:
    """Imputation-only preprocessing bundle for tree models."""

    imputer: SimpleImputer
    feature_columns: list[str]


@dataclass
class LinearPreprocessor:
    """Imputation, variance filtering, scaling, and PCA bundle."""

    imputer: SimpleImputer
    variance_selector: VarianceThreshold
    scaler: StandardScaler
    pca: PCA
    feature_columns: list[str]


def select_feature_columns_by_missingness(
    X_train: pd.DataFrame,
    missingness_threshold: float = 0.50,
) -> MissingnessSelection:
    """Select columns using only training-split missing-value ratios."""

    if not 0 <= missingness_threshold <= 1:
        raise ValueError("missingness_threshold must be between 0 and 1.")

    train_missing_ratio = X_train.isna().mean()
    keep_columns = train_missing_ratio[
        train_missing_ratio <= missingness_threshold
    ].index.tolist()
    drop_columns = train_missing_ratio[
        train_missing_ratio > missingness_threshold
    ].index.tolist()

    return MissingnessSelection(
        keep_columns=keep_columns,
        drop_columns=drop_columns,
        train_missing_ratio=train_missing_ratio,
        missingness_threshold=missingness_threshold,
    )


def apply_missingness_selection(
    X: pd.DataFrame,
    selection: MissingnessSelection,
) -> pd.DataFrame:
    """Apply a training-learned missingness selection to any split."""

    return X.loc[:, selection.keep_columns].copy()


def fit_tree_preprocessor(
    X_train: pd.DataFrame,
    strategy: str = "median",
) -> tuple[TreePreprocessor, object]:
    """Fit an imputation-only preprocessing path for tree models."""

    imputer = SimpleImputer(strategy=strategy)
    X_train_transformed = imputer.fit_transform(X_train)
    preprocessor = TreePreprocessor(
        imputer=imputer,
        feature_columns=list(X_train.columns),
    )
    return preprocessor, X_train_transformed


def transform_tree_preprocessor(
    X: pd.DataFrame,
    preprocessor: TreePreprocessor,
) -> object:
    """Transform a split using a fitted tree preprocessing bundle."""

    return preprocessor.imputer.transform(X.loc[:, preprocessor.feature_columns])


def fit_linear_preprocessor(
    X_train: pd.DataFrame,
    strategy: str = "median",
    pca_variance: float = 0.95,
    random_state: int = 42,
) -> tuple[LinearPreprocessor, object]:
    """Fit the linear and neural-network preprocessing path."""

    imputer = SimpleImputer(strategy=strategy)
    variance_selector = VarianceThreshold(threshold=0.0)
    scaler = StandardScaler()
    pca = PCA(n_components=pca_variance, random_state=random_state)

    X_imputed = imputer.fit_transform(X_train)
    X_variable = variance_selector.fit_transform(X_imputed)
    X_scaled = scaler.fit_transform(X_variable)
    X_pca = pca.fit_transform(X_scaled)

    preprocessor = LinearPreprocessor(
        imputer=imputer,
        variance_selector=variance_selector,
        scaler=scaler,
        pca=pca,
        feature_columns=list(X_train.columns),
    )
    return preprocessor, X_pca


def transform_linear_preprocessor(
    X: pd.DataFrame,
    preprocessor: LinearPreprocessor,
) -> object:
    """Transform a split using a fitted linear preprocessing bundle."""

    X_imputed = preprocessor.imputer.transform(
        X.loc[:, preprocessor.feature_columns]
    )
    X_variable = preprocessor.variance_selector.transform(X_imputed)
    X_scaled = preprocessor.scaler.transform(X_variable)
    return preprocessor.pca.transform(X_scaled)
