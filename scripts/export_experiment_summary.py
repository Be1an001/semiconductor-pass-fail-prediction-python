"""Export portfolio-ready summaries from SECOM experiment metric CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SECOM experiment summary and model card reports."
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
        help="Directory where Markdown reports will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_dir = _project_path(args.metrics_dir)
    reports_dir = _project_path(args.reports_dir)

    validation_metrics = _read_required_csv(metrics_dir / "validation_metrics.csv")
    rf_improvement = _read_required_csv(metrics_dir / "rf_improvement_table.csv")
    final_test_metrics = _read_required_csv(metrics_dir / "final_test_metrics.csv")

    reports_dir.mkdir(parents=True, exist_ok=True)
    experiment_summary = build_experiment_summary(
        validation_metrics=validation_metrics,
        rf_improvement=rf_improvement,
        final_test_metrics=final_test_metrics,
    )
    model_card = build_model_card(
        validation_metrics=validation_metrics,
        final_test_metrics=final_test_metrics,
    )

    summary_path = reports_dir / "experiment_summary.md"
    model_card_path = reports_dir / "model_card.md"
    summary_path.write_text(experiment_summary, encoding="utf-8")
    model_card_path.write_text(model_card, encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {model_card_path}")
    return 0


def build_experiment_summary(
    validation_metrics: pd.DataFrame,
    rf_improvement: pd.DataFrame,
    final_test_metrics: pd.DataFrame,
) -> str:
    final_row = final_test_metrics.iloc[0]
    selected_name = final_row["selected_experiment_name"]
    selected_validation = validation_metrics[
        validation_metrics["experiment_name"] == selected_name
    ].iloc[0]

    validation_table = markdown_table(
        validation_metrics,
        [
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
        ],
    )
    rf_table = markdown_table(
        rf_improvement,
        [
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
        ],
    )
    final_table = markdown_table(
        final_test_metrics,
        [
            "selected_experiment_name",
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
        ],
    )

    return f"""# SECOM Random Forest Experiment Summary

## Project Objective

This project applies a reproducible machine learning workflow to the public UCI SECOM semiconductor dataset. The objective is to study whether sensor measurements can support pass/fail screening while keeping the interpretation appropriate for an imbalanced, anonymous, public dataset.

The workflow is MLOps-lite rather than a deployed manufacturing system. It focuses on reproducible scripts, validation-based threshold selection, MLflow tracking, generated metric files, and careful reporting.

## Dataset Summary

- Rows: 1,567
- Loaded sensor features: 590
- Pass samples: 1,463
- Fail samples: 104
- Fail rate: 6.64%
- Target mapping: raw `-1` to pass class `0`, raw `1` to fail class `1`

The sensor variables are anonymous. Feature importance values should be read as model-driven signals, not as process explanations.

## Imbalanced Classification Challenge

The fail class is rare, so raw accuracy can be misleading. A model can obtain high accuracy by predicting nearly every sample as pass while missing all fail cases. For this reason, the evaluation emphasizes fail-class recall, F2-score, balanced accuracy, PR-AUC, ROC-AUC, confusion matrix counts, and review workload.

Review workload is reported as:

`review_rate = (TP + FP) / total_samples`

## Validation Experiment Design

The script-based workflow uses the same stratified 60/20/20 train, validation, and test split logic as the original notebook. Preprocessing is fit on the training split only. The validation split is used for model comparison and threshold selection. The holdout test split is reserved for one final evaluation after the model and threshold are selected.

Threshold sweeps use validation probabilities from 0.05 to 0.95. Tuned thresholds are selected by maximizing F2-score, with ties resolved toward lower review rate.

## Validation Metrics

{validation_table}

## Random Forest Iteration Comparison

{rf_table}

The validation results show that threshold choice has a larger practical effect than the small RandomizedSearchCV improvement in this run. The selected final candidate is `{selected_name}`, which used a validation-selected threshold of {format_number(selected_validation["threshold"])}.

## Final Holdout Test Evaluation

The selected model and threshold were evaluated once on the untouched holdout test set.

{final_table}

On the final holdout split, the model detected {int(final_row["tp"])} of {int(final_row["tp"] + final_row["fn"])} fail cases. It also flagged {int(final_row["fp"])} pass cases for review. This supports a screening interpretation: the model can surface a subset of higher-risk samples, but it is not an automated accept/reject system.

## Key Interpretation

- The useful story is not raw accuracy.
- The practical story is fail-class screening performance and the review workload created by lower thresholds.
- The validation-selected threshold improves fail recall compared with the default 0.50 threshold.
- The false-positive count means downstream review capacity and cost would matter in any real operating setting.

## Limitations

- The fail class is small, with only 104 fail cases overall.
- Model selection is based on one stratified random split.
- The final test split has only 21 fail cases, so recall and F2 estimates can move if a few cases change.
- The workflow does not use a time-based validation split.
- Threshold selection is validation-based and not tied to a real engineering cost function.
- Feature importance does not establish process causality.
- No fab stakeholder validation, operational rollout, monitoring, or cost savings are claimed.

## Next Steps

- Add repeated stratified validation or time-based validation.
- Compare threshold options against explicit review-capacity assumptions.
- Add calibration checks for predicted probabilities.
- Add stability checks for feature importance across resamples.
- Preserve the original notebook and create a final interview notebook only after the script-based workflow is stable.
"""


def build_model_card(
    validation_metrics: pd.DataFrame,
    final_test_metrics: pd.DataFrame,
) -> str:
    final_row = final_test_metrics.iloc[0]
    selected_name = final_row["selected_experiment_name"]
    validation_row = validation_metrics[
        validation_metrics["experiment_name"] == selected_name
    ].iloc[0]

    validation_table = markdown_table(
        pd.DataFrame([validation_row]),
        [
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
        ],
    )
    final_table = markdown_table(
        final_test_metrics,
        [
            "selected_experiment_name",
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
        ],
    )

    return f"""# Model Card: SECOM Random Forest Screening Model

## Model Name

SECOM Random Forest fail-screening model selected from `{selected_name}`.

## Intended Use

This model is intended for portfolio analysis of imbalanced semiconductor pass/fail screening using the public UCI SECOM dataset. It is designed to demonstrate a reproducible applied ML workflow with validation-based threshold selection, traceable metrics, and cautious interpretation.

## Not Intended Use

This model is not intended for operational deployment, automated fab pass/fail decisions, safety-critical decisions, or process diagnosis. It should not be used to make real manufacturing decisions without domain validation, cost analysis, monitoring, and process-engineering review.

## Dataset

- Source: public UCI SECOM dataset
- Rows: 1,567
- Loaded sensor features: 590
- Pass samples: 1,463
- Fail samples: 104
- Fail rate: 6.64%
- Sensor names: anonymous

## Target Definition

The raw label is mapped as follows:

- `-1` maps to pass class `0`
- `1` maps to fail class `1`

The positive class is the fail class.

## Preprocessing Summary

- Stratified train, validation, and test split
- Training-only missingness filtering at a 50% threshold
- Median imputation fit on training data only
- Tree-model path without scaling or PCA
- Test data is transformed using training-fitted preprocessing only

## Model Family

Random Forest classifier with class imbalance handling from the selected validation experiment.

## Threshold Selection Method

The final threshold was selected from validation probabilities only. The selection metric was F2-score, which emphasizes fail-class recall more than precision. Ties are resolved toward lower review rate. The test set was not used for threshold selection.

## Validation Metrics Summary

{validation_table}

## Final Test Metrics Summary

{final_table}

## Operational Interpretation

At the selected threshold, the final holdout evaluation detected {int(final_row["tp"])} fail cases and missed {int(final_row["fn"])} fail cases. It also flagged {int(final_row["fp"])} pass cases for review, producing a review rate of {format_percent(final_row["review_rate"])}.

This supports a screening interpretation rather than an automated decision interpretation.

## Ethical and Operational Limitations

- The dataset is small and highly imbalanced.
- False negatives could represent missed fail cases.
- False positives create review workload.
- The data is historical, anonymous, and not tied to live operational context.
- Feature importance values are not process-causal explanations.
- The threshold is not based on a real business or engineering cost function.

## Monitoring and Future Validation Needs

- Repeated split or cross-validation checks
- Time-based validation using timestamp information
- Probability calibration review
- Threshold review under explicit false-positive and false-negative costs
- Drift monitoring before any operational use
- Stability review for model-important sensor variables

## Non-Production Status

This model is an analytical and portfolio workflow artifact. It is not an operational manufacturing system and has not been validated with fab stakeholders.
"""


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    headers = columns
    rows = []
    for _, row in data.loc[:, columns].iterrows():
        rows.append([format_cell(row[column]) for column in columns])

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return format_number(value)
    return str(value)


def format_number(value: object) -> str:
    numeric = float(value)
    if numeric == 0:
        return "0.0000"
    if abs(numeric) >= 1:
        return f"{numeric:.3f}"
    return f"{numeric:.4f}"


def format_percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {path}")
    return pd.read_csv(path)


def _project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
