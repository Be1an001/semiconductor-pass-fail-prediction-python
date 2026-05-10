"""CLI skeleton for future final holdout evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/final_rf.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the final SECOM Random Forest evaluation workflow. "
            "Phase 1 does not evaluate the holdout test set."
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
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_yaml_config(args.config)
    model_name = config.get("final_model", {}).get("name", "random_forest")

    print("Phase 1 placeholder: final holdout evaluation is not run yet.")
    print(f"Config: {args.config}")
    print(f"Configured final model: {model_name}")
    print("The test set must remain untouched until model and threshold are fixed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
