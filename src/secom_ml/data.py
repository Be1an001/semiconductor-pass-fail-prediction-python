"""Data loading helpers for the UCI SECOM pass/fail dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


RAW_TO_BINARY_LABEL = {-1: 0, 1: 1}
DEFAULT_DATA_FILES = ("secom.data", "secom_labels.data", "secom.names")


@dataclass(frozen=True)
class SecomMetadata:
    """Metadata returned with the loaded SECOM feature matrix and target."""

    data_dir: Path
    feature_path: Path
    labels_path: Path
    names_path: Path
    timestamps: pd.Series
    raw_labels: pd.Series
    label_mapping: dict[int, int]
    feature_columns: list[str]


def project_root() -> Path:
    """Return the repository root for the current source-layout package."""

    return Path(__file__).resolve().parents[2]


def candidate_data_dirs() -> list[Path]:
    """Return common data directories used by the notebook and local scripts."""

    root = project_root()
    return [
        root / "data",
        Path.cwd() / "data",
        Path.cwd() / "../data",
        Path("/content/secom_project"),
    ]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """Resolve a directory containing the required SECOM data files."""

    if data_dir is not None:
        resolved = Path(data_dir).expanduser().resolve()
        _validate_data_dir(resolved)
        return resolved

    for candidate in candidate_data_dirs():
        resolved = candidate.expanduser().resolve()
        if _has_required_files(resolved):
            return resolved

    searched = ", ".join(str(path) for path in candidate_data_dirs())
    raise FileNotFoundError(
        "Could not find a complete SECOM data directory. "
        f"Searched: {searched}"
    )


def map_secom_labels(raw_labels: pd.Series) -> pd.Series:
    """Map raw SECOM labels to binary pass/fail labels."""

    mapped = raw_labels.replace(RAW_TO_BINARY_LABEL)
    unexpected = sorted(set(mapped.dropna().unique()) - {0, 1})
    if unexpected:
        raise ValueError(f"Unexpected SECOM labels after mapping: {unexpected}")
    return mapped.astype(int)


def load_secom_data(
    data_dir: str | Path | None = None,
    feature_file: str = "secom.data",
    labels_file: str = "secom_labels.data",
    names_file: str = "secom.names",
) -> tuple[pd.DataFrame, pd.Series, SecomMetadata]:
    """Load SECOM features, mapped labels, timestamps, and metadata."""

    resolved_data_dir = resolve_data_dir(data_dir)
    feature_path = resolved_data_dir / feature_file
    labels_path = resolved_data_dir / labels_file
    names_path = resolved_data_dir / names_file

    for path in (feature_path, labels_path, names_path):
        if not path.exists():
            raise FileNotFoundError(f"Required SECOM file is missing: {path}")

    X = pd.read_csv(feature_path, sep=r"\s+", header=None, engine="python")
    X.columns = [f"sensor_{index:03d}" for index in range(1, X.shape[1] + 1)]

    labels_df = pd.read_csv(labels_path, sep=r"\s+", header=None, engine="python")
    labels_df.columns = ["label", "date_part", "time_part"]

    timestamps = pd.to_datetime(
        labels_df["date_part"].astype(str).str.replace('"', "", regex=False)
        + " "
        + labels_df["time_part"].astype(str).str.replace('"', "", regex=False),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    y = map_secom_labels(labels_df["label"])
    y.name = "target"

    metadata = SecomMetadata(
        data_dir=resolved_data_dir,
        feature_path=feature_path,
        labels_path=labels_path,
        names_path=names_path,
        timestamps=timestamps,
        raw_labels=labels_df["label"].copy(),
        label_mapping=RAW_TO_BINARY_LABEL.copy(),
        feature_columns=list(X.columns),
    )
    return X, y, metadata


def _has_required_files(data_dir: Path, files: Iterable[str] = DEFAULT_DATA_FILES) -> bool:
    return data_dir.exists() and all((data_dir / file_name).exists() for file_name in files)


def _validate_data_dir(data_dir: Path) -> None:
    if not _has_required_files(data_dir):
        required = ", ".join(DEFAULT_DATA_FILES)
        raise FileNotFoundError(
            f"Data directory must contain {required}. Received: {data_dir}"
        )
