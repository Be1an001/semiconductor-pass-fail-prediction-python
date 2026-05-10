"""Train, validation, and test splitting utilities for SECOM experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SecomSplits:
    """Container for the 60/20/20 stratified split used by the notebook."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray


def create_train_validation_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify: bool = True,
) -> SecomSplits:
    """Create a stratified train/validation/test split with 60/20/20 defaults."""

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError("test_size plus validation_size must be less than 1.")

    row_index = np.arange(len(X))
    stratify_target = y if stratify else None

    X_train_full, X_test, y_train_full, y_test, idx_train_full, idx_test = (
        train_test_split(
            X,
            y,
            row_index,
            test_size=test_size,
            stratify=stratify_target,
            random_state=random_state,
        )
    )

    relative_validation_size = validation_size / (1.0 - test_size)
    train_stratify_target = y_train_full if stratify else None

    X_train, X_validation, y_train, y_validation, idx_train, idx_validation = (
        train_test_split(
            X_train_full,
            y_train_full,
            idx_train_full,
            test_size=relative_validation_size,
            stratify=train_stratify_target,
            random_state=random_state,
        )
    )

    splits = SecomSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        train_index=idx_train,
        validation_index=idx_validation,
        test_index=idx_test,
    )
    check_split_overlap(splits, total_rows=len(X))
    return splits


def check_split_overlap(splits: SecomSplits, total_rows: int | None = None) -> None:
    """Raise an error if train, validation, or test rows overlap."""

    train_rows = set(splits.train_index.tolist())
    validation_rows = set(splits.validation_index.tolist())
    test_rows = set(splits.test_index.tolist())

    if train_rows & validation_rows:
        raise ValueError("Train and validation splits overlap.")
    if train_rows & test_rows:
        raise ValueError("Train and test splits overlap.")
    if validation_rows & test_rows:
        raise ValueError("Validation and test splits overlap.")

    if total_rows is not None:
        combined = train_rows | validation_rows | test_rows
        if len(combined) != total_rows:
            raise ValueError("Split rows do not cover the full dataset.")
