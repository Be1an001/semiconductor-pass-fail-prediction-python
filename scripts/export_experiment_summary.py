"""CLI skeleton for future SECOM experiment summary export."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the SECOM experiment summary export. "
            "Phase 1 does not generate reports yet."
        )
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("outputs/metrics"),
        help="Directory containing generated metric CSV files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where Markdown reports will be written later.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Phase 1 placeholder: experiment summary export is not run yet.")
    print(f"Metrics directory: {args.metrics_dir}")
    print(f"Reports directory: {args.reports_dir}")
    print("Next phase will summarize tracked validation and final test artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
