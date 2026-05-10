"""Tests for SECOM data loading and label mapping."""

from __future__ import annotations

import pandas as pd

from secom_ml.data import load_secom_data, map_secom_labels


def test_map_secom_labels_maps_pass_and_fail_values() -> None:
    raw = pd.Series([-1, 1, -1, 1])

    mapped = map_secom_labels(raw)

    assert mapped.tolist() == [0, 1, 0, 1]
    assert mapped.dtype == "int32" or mapped.dtype == "int64"


def test_load_secom_data_uses_repo_data_files() -> None:
    X, y, metadata = load_secom_data()

    assert X.shape == (1567, 590)
    assert y.shape == (1567,)
    assert y.value_counts().sort_index().to_dict() == {0: 1463, 1: 104}
    assert len(metadata.timestamps) == 1567
    assert metadata.feature_columns[0] == "sensor_001"
    assert metadata.feature_columns[-1] == "sensor_590"
