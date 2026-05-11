# Project Walkthrough

This walkthrough summarizes the main analysis decisions behind the SECOM semiconductor pass/fail prediction project. It is meant to be read with the root README, generated reports, data note, output guide, and final notebook.

See:

- Root README: [`../README.md`](../README.md)
- Experiment summary: [`../reports/experiment_summary.md`](../reports/experiment_summary.md)
- Model card: [`../reports/model_card.md`](../reports/model_card.md)
- Final portfolio notebook: [`../notebooks/EAI6010_SECOM_Pass_Fail_Portfolio.ipynb`](../notebooks/EAI6010_SECOM_Pass_Fail_Portfolio.ipynb)
- Data note: [`../data/README.md`](../data/README.md)
- Output guide: [`../outputs/README.md`](../outputs/README.md)

## Project Overview

This project uses the public UCI SECOM dataset to study semiconductor pass/fail screening with sensor data.

The final notebook presents the project for portfolio review. The script-based workflow handles reusable preprocessing, Random Forest validation experiments, validation-based threshold selection, local MLflow tracking, final holdout evaluation, and generated reports.

This project is best understood as a portfolio workflow and screening-style ML prototype. It is not an operational manufacturing decision system.

## Business Problem

The project asks:

**Can sensor measurements help flag units that are more likely to fail?**

The fail class is rare, so the main question is not whether the model can produce high raw accuracy. The more useful question is whether a validation-selected threshold can catch more fail cases while keeping the flagged sample rate understandable.

## Dataset

Main files:

- [`../data/secom.data`](../data/secom.data)
- [`../data/secom_labels.data`](../data/secom_labels.data)
- [`../data/secom.names`](../data/secom.names)

Dataset summary:

- 1,567 rows
- 590 loaded anonymous sensor features
- 1,463 pass samples
- 104 fail samples
- 6.64% fail rate
- label mapping: `-1 -> 0` for pass, `1 -> 1` for fail

The sensor variables are anonymous, so feature importance should be interpreted carefully.

## Methodology

The project workflow is:

1. Load SECOM sensor and label files.
2. Map raw labels to binary pass/fail values.
3. Parse timestamps for exploratory checks.
4. Review class imbalance, missing values, and simple time patterns.
5. Create a stratified train/validation/test split.
6. Fit preprocessing on training data only.
7. Compare Dummy Classifier, Logistic Regression + PCA, and Random Forest variants.
8. Tune thresholds on validation data.
9. Track experiments with local MLflow.
10. Evaluate the selected model once on the holdout test set.
11. Review Random Forest feature importance as model-driven signals.

The latest reproducible results come from the script-based workflow. The final notebook presents those results by reading generated data, metrics, figures, and reports. It does not rerun model tuning.

## Script-Based Workflow

The newer workflow adds reusable code under [`../src/secom_ml`](../src/secom_ml) and command-line scripts under [`../scripts`](../scripts).

Main scripts:

- `scripts/run_rf_experiments.py`
- `scripts/evaluate_final_model.py`
- `scripts/export_experiment_summary.py`

Main configs:

- `configs/rf_experiments.yaml`
- `configs/final_rf.yaml`

The workflow uses:

- stratified 60/20/20 train, validation, and test split
- training-only preprocessing
- validation-only threshold selection
- small RandomizedSearchCV on training folds only
- local MLflow tracking
- generated CSV metrics and figures

## Validation Experiments

The validation experiments compare simple baselines and Random Forest variants.

The most important Random Forest comparison is:

| Experiment | Threshold | Recall | F2 | Balanced accuracy | TP | FP | FN | TN | Flagged sample rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `rf_current_config_threshold_050` | 0.500 | 0.0000 | 0.0000 | 0.5000 | 0 | 0 | 21 | 293 | 0.0000 |
| `rf_current_config_threshold_tuned` | 0.110 | 0.5714 | 0.3371 | 0.6458 | 12 | 82 | 9 | 211 | 0.2994 |

At threshold `0.50`, the current Random Forest configuration missed all validation fail cases. The validation-selected threshold caught more fail cases, but it also increased the flagged sample rate.

That is the main trade-off: better fail-case screening behavior comes with more false positives.

## Final Holdout Evaluation

The final candidate is selected from validation results and then evaluated once on the untouched test split.

Selected candidate:

- `rf_current_config_threshold_tuned`
- validation-selected threshold: `0.110`

Final holdout test metrics:

| Metric | Value |
|---|---:|
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

The model detected 11 of 21 fail cases in the test split and flagged 56 pass cases.

![Final confusion matrix](../outputs/figures/final_confusion_matrix.png)

## Visual Evidence

### Precision-recall curve

The PR curve is important because the fail class is rare.

![Final precision-recall curve](../outputs/figures/final_pr_curve.png)

### ROC curve

The ROC curve summarizes ranking behavior, but it should be read together with PR-AUC and confusion matrix counts.

![Final ROC curve](../outputs/figures/final_roc_curve.png)

### Feature importance

Feature importance shows model-driven signal ranking, not physical root-cause proof.

![Final feature importance](../outputs/figures/final_feature_importance.png)

## Generated Reports

The generated Markdown reports summarize the script-based results:

- [`../reports/experiment_summary.md`](../reports/experiment_summary.md)
- [`../reports/model_card.md`](../reports/model_card.md)

The reports are generated from CSV outputs under [`../outputs/metrics`](../outputs/metrics).

## MLflow Tracking

The experiment scripts use local MLflow tracking:

- tracking URI: `sqlite:///mlflow.db`
- experiment name: `secom-pass-fail-screening`

Local MLflow files are ignored by Git. This is useful for local reproducibility, but it is not a production monitoring setup.

## Interpretation

The useful story is not raw accuracy. The useful story is threshold-based fail screening:

- how many fail cases are caught
- how many fail cases are missed
- how many pass cases are flagged
- the flagged sample rate created by the threshold

Feature importance values are useful for model review, but they are not process explanations.

## Limitations

- The fail class is small.
- The validation and test splits each contain only 21 fail cases.
- Results are based on one stratified random split.
- A time-based validation split is not yet included.
- Threshold choice is not tied to a real engineering cost function.
- The dataset is public and anonymous.
- No fab stakeholder validation, operational rollout, dashboard, SQL layer, GenAI component, or full MLOps platform is claimed.

## Future Improvements

Good next steps would be:

- add repeated split or time-based validation
- compare thresholds against review-capacity assumptions
- add calibration checks
- check feature-importance stability across resamples
- keep the final portfolio notebook updated when script results change
