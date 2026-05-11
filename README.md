# Semiconductor Pass/Fail Prediction with the UCI SECOM Dataset

This project applies a reproducible machine learning workflow to the UCI SECOM semiconductor pass/fail dataset, focusing on imbalanced classification, Random Forest tuning, threshold selection, experiment tracking, and model interpretation.

The project is best read as a master's-level applied machine learning portfolio workflow. It is a screening prototype for a public, anonymous dataset, not an operational semiconductor quality-control system.

## Project Overview

The main question is:

**Can sensor measurements help flag units that are more likely to fail downstream testing?**

The project started as a notebook-first analysis and now also includes a script-based workflow. The current version archives the original notebook and adds a final portfolio notebook, reusable Python modules, configuration files, MLflow tracking, generated metrics, final holdout evaluation, and portfolio-ready reports.

The main story is not raw accuracy. The dataset is highly imbalanced, so the more useful story is the fail-class screening trade-off: recall, F2-score, balanced accuracy, PR-AUC, confusion matrix counts, and flagged sample rate.

## Problem Framing

Semiconductor manufacturing can produce many sensor and process measurements. A useful analytics task is to identify units that may deserve extra review before final downstream testing.

The modeling goal is to identify patterns that can help flag likely fail cases under class imbalance. A lower threshold can catch more fail cases, but it also creates more false positives and a higher flagged sample rate. That threshold trade-off is the main modeling decision.

## Dataset

This repository includes the public UCI SECOM files used by the workflow.

| File | Purpose |
|---|---|
| `data/secom.data` | Sensor feature matrix |
| `data/secom_labels.data` | Raw labels and timestamps |
| `data/secom.names` | UCI metadata |

Dataset summary used by this project:

- Rows: 1,567
- Loaded sensor features: 590
- Pass samples: 1,463
- Fail samples: 104
- Fail rate: 6.64%
- Label mapping: `-1 -> 0` for pass, `1 -> 1` for fail

The UCI metadata describes 591 attributes. This project loads 590 sensor columns from `secom.data` and reads labels and timestamps separately from `secom_labels.data`.

## Why Accuracy Is Misleading

Only about 6.6% of samples are fail cases. A model can look good by raw accuracy while missing the fail class.

For example, on the validation split, the current Random Forest configuration at threshold `0.50` missed every fail case:

| Experiment | Threshold | Recall | F2 | Balanced accuracy | TP | FP | FN | TN | Flagged sample rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rf_current_config_threshold_050` | 0.500 | 0.0000 | 0.0000 | 0.5000 | 0 | 0 | 21 | 293 | 0.0000 |
| `rf_current_config_threshold_tuned` | 0.110 | 0.5714 | 0.3371 | 0.6458 | 12 | 82 | 9 | 211 | 0.2994 |

The tuned threshold changed the operating point. It caught more fail cases, but it also sent more samples to review. That is the trade-off this project documents.

## Workflow

The current workflow has two layers:

1. The archived original notebook remains available as the first notebook-first baseline.
2. The script-based workflow runs reproducible experiments and produces metrics, figures, and reports.
3. The final portfolio notebook reads the generated outputs and summarizes the project.

Script workflow:

1. Load SECOM data from `data/`.
2. Map labels to pass/fail target values.
3. Create a stratified train/validation/test split.
4. Fit preprocessing on training data only.
5. Run validation-only baseline and Random Forest experiments.
6. Select thresholds using validation probabilities only.
7. Track runs with local MLflow.
8. Evaluate the selected final model once on the holdout test set.
9. Export reports from generated CSV outputs.

The test set is not used for model selection, threshold selection, or hyperparameter tuning.

## Repository Structure

| Path | Description |
|---|---|
| `data/` | Public SECOM data files and dataset note |
| `notebooks/` | Final portfolio notebook and archived original notebook |
| `src/secom_ml/` | Reusable data, split, preprocessing, model, metric, threshold, plot, and tracking helpers |
| `scripts/` | Command-line scripts for experiments, final evaluation, and report export |
| `configs/` | YAML configuration files for experiments and final evaluation |
| `outputs/metrics/` | Generated CSV metrics from the latest local script run |
| `outputs/figures/` | Notebook figures and final script-generated figures |
| `reports/` | PDF reports plus generated experiment summary and model card |
| `walkthrough/` | Project walkthrough |
| `tests/` | Lightweight tests for data loading, metrics, and threshold selection |

## Experiment Design

The script-based workflow uses the same split idea as the notebook:

- 60% train
- 20% validation
- 20% holdout test
- stratified by target label
- random seed `42`

Preprocessing is fit on training data only:

- drop high-missing columns using the training split
- median imputation
- tree-model path without scaling or PCA
- linear baseline path with imputation, variance filtering, scaling, and PCA

Threshold sweeps use validation probabilities from `0.05` to `0.95`. Tuned thresholds are selected by F2-score, with ties resolved toward lower flagged sample rate.

## Random Forest Iterative Results

The table below comes from `outputs/metrics/rf_improvement_table.csv`.

| Experiment | Threshold | Recall | F2 | Balanced accuracy | PR-AUC | TP | FP | FN | TN | Flagged sample rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rf_default_threshold_050` | 0.500 | 0.0000 | 0.0000 | 0.5000 | 0.1417 | 0 | 0 | 21 | 293 | 0.0000 |
| `rf_balanced_threshold_050` | 0.500 | 0.0000 | 0.0000 | 0.5000 | 0.1329 | 0 | 0 | 21 | 293 | 0.0000 |
| `rf_current_config_threshold_050` | 0.500 | 0.0000 | 0.0000 | 0.5000 | 0.1329 | 0 | 0 | 21 | 293 | 0.0000 |
| `rf_current_config_threshold_tuned` | 0.110 | 0.5714 | 0.3371 | 0.6458 | 0.1329 | 12 | 82 | 9 | 211 | 0.2994 |
| `rf_random_search_best_threshold_050` | 0.500 | 0.0000 | 0.0000 | 0.4983 | 0.1446 | 0 | 1 | 21 | 292 | 0.0032 |
| `rf_random_search_best_threshold_tuned` | 0.130 | 0.5714 | 0.3297 | 0.6390 | 0.1446 | 12 | 86 | 9 | 207 | 0.3121 |

In this run, threshold tuning had a larger practical effect than the small RandomizedSearchCV result. The final selected validation candidate was `rf_current_config_threshold_tuned`.

## Final Holdout Test Result

The selected validation candidate was evaluated once on the untouched holdout test set.

Source file: `outputs/metrics/final_test_metrics.csv`

| Metric | Value |
|---|---:|
| Selected experiment | `rf_current_config_threshold_tuned` |
| Threshold | 0.110 |
| Recall | 0.5238 |
| F2 | 0.3642 |
| Balanced accuracy | 0.6663 |
| PR-AUC | 0.2192 |
| ROC-AUC | 0.7978 |
| TP | 11 |
| FP | 56 |
| FN | 10 |
| TN | 237 |
| Flagged sample rate | 0.2134 |

The final model detected 11 of 21 fail cases in the holdout test split. At this threshold, it also flagged 56 pass cases. This is a screening result, not an accept/reject rule.

## MLflow Experiment Tracking

The experiment scripts use local MLflow tracking.

- Tracking URI: `sqlite:///mlflow.db`
- Experiment name: `secom-pass-fail-screening`
- One run is logged for each validation experiment.
- A separate run is logged for final holdout evaluation.

Local MLflow files such as `mlflow.db`, `mlruns/`, and `mlartifacts/` are ignored by Git.

To open the local MLflow UI after running experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Key Artifacts

Generated metrics:

- `outputs/metrics/validation_metrics.csv`
- `outputs/metrics/threshold_sweep.csv`
- `outputs/metrics/rf_improvement_table.csv`
- `outputs/metrics/final_test_metrics.csv`
- `outputs/metrics/final_feature_importance.csv`

Generated figures:

- `outputs/figures/final_confusion_matrix.png`
- `outputs/figures/final_roc_curve.png`
- `outputs/figures/final_pr_curve.png`
- `outputs/figures/final_feature_importance.png`

Reports:

- `reports/experiment_summary.md`
- `reports/model_card.md`

Original project materials:

- `notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb`
- `notebooks/archive/EAI6010_Module_4_Assignment_V2_Cheng_L_original.ipynb`
- `reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf`
- `reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf`

## How to Run Locally

Install requirements:

```bash
pip install -r requirements.txt
```

Run tests and linting:

```bash
python -m pytest
python -m ruff check .
```

Run validation experiments:

```bash
python scripts/run_rf_experiments.py --config configs/rf_experiments.yaml
```

Run final holdout evaluation:

```bash
python scripts/evaluate_final_model.py --config configs/final_rf.yaml
```

Export the Markdown reports:

```bash
python scripts/export_experiment_summary.py
```

Open MLflow locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The CSV outputs are overwritten by the latest local run. The project does not create timestamped metrics files.

## Limitations

- The fail class is small, with only 104 fail cases overall.
- The validation and test splits each contain 21 fail cases.
- Results are based on one stratified random split.
- A time-based validation split is not yet included.
- Threshold selection uses validation metrics, not a real engineering cost function.
- Feature importance values are model-driven signals, not process explanations.
- The dataset is public and anonymous, so operational sensor meaning is limited.
- No fab stakeholder validation, operational rollout, monitoring system, or cost savings are claimed.

## Portfolio Takeaway

This project shows a practical imbalanced-classification workflow:

- start with a notebook-first analysis
- move core logic into reusable Python modules
- run validation-only Random Forest experiments
- track runs with MLflow
- select thresholds using validation data
- evaluate once on a holdout test split
- report the screening trade-off clearly

The strongest portfolio message is that threshold choice matters for fail-class screening. The validation-selected operating point caught more fail cases than the default `0.50` threshold, while also increasing the flagged sample rate. That trade-off is the central result.

## Related Files

- Final notebook: [`notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb`](notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb)
- Archived original notebook: [`notebooks/archive/EAI6010_Module_4_Assignment_V2_Cheng_L_original.ipynb`](notebooks/archive/EAI6010_Module_4_Assignment_V2_Cheng_L_original.ipynb)
- Dataset note: [`data/README.md`](data/README.md)
- Output note: [`outputs/README.md`](outputs/README.md)
- Walkthrough: [`walkthrough/project-walkthrough.md`](walkthrough/project-walkthrough.md)
- Experiment summary: [`reports/experiment_summary.md`](reports/experiment_summary.md)
- Model card: [`reports/model_card.md`](reports/model_card.md)
