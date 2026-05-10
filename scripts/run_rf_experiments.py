"""CLI skeleton for future SECOM Random Forest experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/rf_experiments.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the SECOM Random Forest experiment workflow. "
            "Phase 1 validates configuration shape but does not train models."
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
        help="Print the planned experiment settings without training models.",
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
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_yaml_config(args.config)
    experiment_names = [
        item.get("name", "<unnamed>")
        for item in config.get("experiments", [])
    ]

    print("Phase 1 placeholder: Random Forest experiments are not run yet.")
    print(f"Config: {args.config}")
    print(f"Configured experiments: {', '.join(experiment_names)}")
    print("Next phase will connect data loading, preprocessing, metrics, and MLflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
