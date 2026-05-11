# Semiconductor Pass/Fail Prediction with the UCI SECOM Dataset

This project analyzes semiconductor pass/fail screening using the public UCI SECOM sensor dataset. The goal was to compare classification models under severe class imbalance and explain the screening trade-off in a careful, reproducible way.

This is an individual applied machine learning portfolio project. It uses a Python/Jupyter notebook, reusable scripts, generated metrics, local MLflow tracking, and final Markdown reports. It is not a production semiconductor quality-control system.

## Project Type / Status / Tools

- **Project type:** Applied machine learning / manufacturing analytics
- **Status:** Individual portfolio project
- **Main workflow:** Script-based experiments plus final portfolio notebook
- **Dataset:** Public UCI SECOM semiconductor sensor data
- **Main model family:** Random Forest
- **Tracking:** Local MLflow experiment tracking
- **Main tools:** Python, pandas, scikit-learn, matplotlib, seaborn, MLflow, pytest, ruff
- **Production status:** Screening-style ML prototype, not a deployed system

## Business Problem

Semiconductor manufacturing can generate many sensor and process measurements. A useful analytics question is whether those measurements can help flag units that are more likely to fail downstream testing.

The challenge is that fail cases are rare. A model can look strong by raw accuracy while missing the fail class. For this reason, this project focuses on fail-class recall, F2-score, balanced accuracy, PR-AUC, confusion matrix counts, and flagged sample rate.

The final result should be interpreted as a screening signal. It is not an automated accept/reject rule and not a real fab deployment.

## Project Objective

The objective was to build a reproducible applied ML workflow that:

- loads and validates the SECOM data
- handles missing sensor values without preprocessing leakage
- compares baseline and Random Forest experiments on validation data
- selects thresholds using validation probabilities only
- evaluates the selected candidate once on the holdout test split
- tracks local experiment runs with MLflow
- exports metrics, figures, and documentation artifacts for review

## Dataset

This repository includes the public UCI SECOM files used by the workflow.

| File | Purpose |
|---|---|
| `data/secom.data` | Sensor feature matrix |
| `data/secom_labels.data` | Raw labels and timestamps |
| `data/secom.names` | UCI metadata |

Dataset summary:

- **Rows:** 1,567
- **Loaded anonymous sensor features:** 590
- **Pass samples:** 1,463
- **Fail samples:** 104
- **Fail rate:** 6.64%
- **Label mapping:** `-1 -> 0` for pass, `1 -> 1` for fail

The UCI metadata describes 591 attributes. This project loads 590 sensor columns from `secom.data` and reads labels and timestamps separately from `secom_labels.data`.

## My Role / Contribution

This was an individual portfolio project. I organized the project around a script-based workflow, reusable Python modules, validation experiments, local MLflow tracking, final holdout evaluation, and a final notebook that reads the generated outputs.

## Methodology

The workflow separates validation model comparison from final holdout evaluation.

1. Load SECOM data from `data/`.
2. Map raw labels into binary pass/fail values.
3. Create a stratified 60/20/20 train, validation, and test split.
4. Fit preprocessing on the training split only.
5. Drop high-missing columns using the training split.
6. Apply median imputation.
7. Use a tree-model path for Random Forest experiments.
8. Use a linear baseline path with imputation, variance filtering, scaling, and PCA.
9. Run validation-only baseline and Random Forest experiments.
10. Select thresholds using validation probabilities only.
11. Track runs with local MLflow.
12. Evaluate the selected model and threshold once on the holdout test split.
13. Export CSV metrics, final figures, experiment summary, and model card.

The test split is not used for model selection, threshold selection, or hyperparameter tuning.

## Key Findings

- The fail class is rare: 104 fail cases out of 1,567 rows.
- At the default `0.50` threshold, the current Random Forest configuration missed every fail case on the validation split.
- Validation threshold tuning changed the operating point from "flag nothing" to "catch more fail cases but flag more samples."
- The selected experiment was `rf_current_config_threshold_tuned`.
- The final validation-selected threshold was `0.110`.
- On the final holdout split, the model detected **11 of 21 fail cases**.
- The same threshold also flagged **56 pass cases** as fail.
- The final flagged sample rate was **0.2134**, meaning about 21% of test samples would be sent for review at this threshold.
- PR-AUC and flagged sample rate are important because the fail class is rare.

## Visual Highlights

### Final confusion matrix

The confusion matrix shows the screening trade-off: the model detected some fail cases but also flagged many pass cases.

![Final confusion matrix](outputs/figures/final_confusion_matrix.png)

### Final precision-recall curve

The PR curve is important because the fail class is rare.

![Final precision-recall curve](outputs/figures/final_pr_curve.png)

### Final feature importance

Feature importance shows model-driven signal ranking, not physical root-cause proof.

![Final feature importance](outputs/figures/final_feature_importance.png)

The ROC curve is also available in [`outputs/figures/final_roc_curve.png`](outputs/figures/final_roc_curve.png).

## Model Evaluation Note

The final threshold was selected on validation data and then evaluated on the holdout test split. The result should be treated as split-specific.

Final holdout test metrics from `outputs/metrics/final_test_metrics.csv`:

| Metric | Value |
|---|---:|
| Selected experiment | `rf_current_config_threshold_tuned` |
| Threshold | 0.110 |
| Recall | 0.5238 |
| F2-score | 0.3642 |
| Balanced accuracy | 0.6663 |
| PR-AUC | 0.2192 |
| ROC-AUC | 0.7978 |
| True positives | 11 |
| False positives | 56 |
| False negatives | 10 |
| True negatives | 237 |
| Flagged sample rate | 0.2134 |

This suggests useful screening signal, not a production quality decision system.

## Repository Structure

| Path | Description |
|---|---|
| `data/` | Public SECOM data files and dataset note |
| `notebooks/` | Final portfolio notebook |
| `src/secom_ml/` | Reusable data, split, preprocessing, model, metric, threshold, plot, and tracking helpers |
| `scripts/` | Command-line scripts for experiments, final evaluation, and report export |
| `configs/` | YAML configuration files for experiments and final evaluation |
| `outputs/metrics/` | Generated CSV metrics from the latest local script run |
| `outputs/figures/` | Final script-generated figures |
| `reports/` | Generated experiment summary and model card |
| `walkthrough/` | Project walkthrough |
| `tests/` | Lightweight tests for data loading, metrics, and threshold selection |

## How to Reproduce

This repository has a script-based workflow, generated outputs, and a final notebook. The commands below reproduce the local analysis workflow without implying production deployment.

Install requirements:

```bash
python -m pip install -r requirements.txt
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

Notes:

- MLflow local files are ignored by Git.
- Script runs overwrite CSV and Markdown report outputs deterministically with the current configs and random seed.
- MLflow keeps local run history in ignored local files.
- The final notebook can be opened in VS Code or Jupyter after the generated outputs exist.

## MLflow Tracking Notes

The experiment scripts use local MLflow tracking.

- Tracking URI: `sqlite:///mlflow.db`
- Experiment name: `secom-pass-fail-screening`
- One run is logged for each validation experiment.
- A separate run is logged for final holdout evaluation.

This is a local reproducibility layer. It is not a cloud deployment, model registry, or production monitoring setup.

## Evidence and Key Artifacts

Generated metrics:

- [`outputs/metrics/validation_metrics.csv`](outputs/metrics/validation_metrics.csv)
- [`outputs/metrics/threshold_sweep.csv`](outputs/metrics/threshold_sweep.csv)
- [`outputs/metrics/rf_improvement_table.csv`](outputs/metrics/rf_improvement_table.csv)
- [`outputs/metrics/final_test_metrics.csv`](outputs/metrics/final_test_metrics.csv)
- [`outputs/metrics/final_feature_importance.csv`](outputs/metrics/final_feature_importance.csv)

Generated figures:

- [`outputs/figures/final_confusion_matrix.png`](outputs/figures/final_confusion_matrix.png)
- [`outputs/figures/final_pr_curve.png`](outputs/figures/final_pr_curve.png)
- [`outputs/figures/final_roc_curve.png`](outputs/figures/final_roc_curve.png)
- [`outputs/figures/final_feature_importance.png`](outputs/figures/final_feature_importance.png)

Reports:

- [`reports/experiment_summary.md`](reports/experiment_summary.md)
- [`reports/model_card.md`](reports/model_card.md)

Notebook:

- [`notebooks/EAI6010_SECOM_Pass_Fail_Portfolio.ipynb`](notebooks/EAI6010_SECOM_Pass_Fail_Portfolio.ipynb)

## Limitations

- The dataset is public and anonymous.
- The fail class is small, with only 104 fail cases overall.
- The validation and test splits each contain only 21 fail cases.
- Results are based on one stratified random split.
- A time-based validation split is not yet included.
- Threshold selection uses validation metrics, not a real engineering cost function.
- Feature importance values are model-driven signals, not physical root-cause proof.
- No real fab validation, stakeholder adoption, operational rollout, monitoring system, or cost savings are claimed.
- The project does not include a GenAI/LLM component, dashboard, SQL layer, data warehouse, full MLOps platform, or deployed app.

## Future Improvements

Useful next steps would be:

- add repeated split or time-based validation
- compare thresholds against review-capacity assumptions
- add calibration checks for predicted probabilities
- check feature-importance stability across resamples
- document threshold trade-offs with a simple cost or review-capacity example
- keep the final portfolio notebook updated when script results change

## Related Files

- Final notebook: [`notebooks/EAI6010_SECOM_Pass_Fail_Portfolio.ipynb`](notebooks/EAI6010_SECOM_Pass_Fail_Portfolio.ipynb)
- Dataset note: [`data/README.md`](data/README.md)
- Output guide: [`outputs/README.md`](outputs/README.md)
- Walkthrough: [`walkthrough/README.md`](walkthrough/README.md)
- Experiment summary: [`reports/experiment_summary.md`](reports/experiment_summary.md)
- Model card: [`reports/model_card.md`](reports/model_card.md)
