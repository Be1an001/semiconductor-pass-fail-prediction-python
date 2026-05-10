# SECOM Random Forest Experiment Summary

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

| experiment_name | threshold | recall | f2 | balanced_accuracy | pr_auc | tp | fp | fn | tn | review_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy_majority_baseline | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.0669 | 0 | 0 | 21 | 293 | 0.0000 |
| logistic_regression_pca_baseline | 0.3050 | 0.3810 | 0.3008 | 0.6205 | 0.1509 | 8 | 41 | 13 | 252 | 0.1561 |
| rf_default_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.1417 | 0 | 0 | 21 | 293 | 0.0000 |
| rf_balanced_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.1329 | 0 | 0 | 21 | 293 | 0.0000 |
| rf_current_config_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.1329 | 0 | 0 | 21 | 293 | 0.0000 |
| rf_current_config_threshold_tuned | 0.1100 | 0.5714 | 0.3371 | 0.6458 | 0.1329 | 12 | 82 | 9 | 211 | 0.2994 |
| rf_random_search_best_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.4983 | 0.1446 | 0 | 1 | 21 | 292 | 0.0032 |
| rf_random_search_best_threshold_tuned | 0.1300 | 0.5714 | 0.3297 | 0.6390 | 0.1446 | 12 | 86 | 9 | 207 | 0.3121 |

## Random Forest Iteration Comparison

| experiment_name | threshold | recall | f2 | balanced_accuracy | pr_auc | tp | fp | fn | tn | review_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf_default_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.1417 | 0 | 0 | 21 | 293 | 0.0000 |
| rf_balanced_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.1329 | 0 | 0 | 21 | 293 | 0.0000 |
| rf_current_config_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.5000 | 0.1329 | 0 | 0 | 21 | 293 | 0.0000 |
| rf_current_config_threshold_tuned | 0.1100 | 0.5714 | 0.3371 | 0.6458 | 0.1329 | 12 | 82 | 9 | 211 | 0.2994 |
| rf_random_search_best_threshold_050 | 0.5000 | 0.0000 | 0.0000 | 0.4983 | 0.1446 | 0 | 1 | 21 | 292 | 0.0032 |
| rf_random_search_best_threshold_tuned | 0.1300 | 0.5714 | 0.3297 | 0.6390 | 0.1446 | 12 | 86 | 9 | 207 | 0.3121 |

The validation results show that threshold choice has a larger practical effect than the small RandomizedSearchCV improvement in this run. The selected final candidate is `rf_current_config_threshold_tuned`, which used a validation-selected threshold of 0.1100.

## Final Holdout Test Evaluation

The selected model and threshold were evaluated once on the untouched holdout test set.

| selected_experiment_name | threshold | recall | f2 | balanced_accuracy | pr_auc | tp | fp | fn | tn | review_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf_current_config_threshold_tuned | 0.1100 | 0.5238 | 0.3642 | 0.6663 | 0.2192 | 11 | 56 | 10 | 237 | 0.2134 |

On the final holdout split, the model detected 11 of 21 fail cases. It also flagged 56 pass cases for review. This supports a screening interpretation: the model can surface a subset of higher-risk samples, but it is not an automated accept/reject system.

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
