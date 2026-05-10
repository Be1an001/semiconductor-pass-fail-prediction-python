# Model Card: SECOM Random Forest Screening Model

## Model Name

SECOM Random Forest fail-screening model selected from `rf_current_config_threshold_tuned`.

## Intended Use

This model is intended for portfolio analysis of imbalanced semiconductor pass/fail screening using the public UCI SECOM dataset. It is designed to demonstrate a reproducible applied ML workflow with validation-based threshold selection, traceable metrics, and cautious interpretation.

## Not Intended Use

This model is not intended for operational deployment, automated fab pass/fail decisions, safety-critical decisions, or root-cause diagnosis. It should not be used to make real manufacturing decisions without domain validation, cost analysis, monitoring, and process-engineering review.

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

| experiment_name | threshold | recall | f2 | balanced_accuracy | pr_auc | tp | fp | fn | tn | review_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf_current_config_threshold_tuned | 0.1100 | 0.5714 | 0.3371 | 0.6458 | 0.1329 | 12 | 82 | 9 | 211 | 0.2994 |

## Final Test Metrics Summary

| selected_experiment_name | threshold | recall | f2 | balanced_accuracy | pr_auc | tp | fp | fn | tn | review_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rf_current_config_threshold_tuned | 0.1100 | 0.5238 | 0.3642 | 0.6663 | 0.2192 | 11 | 56 | 10 | 237 | 0.2134 |

## Operational Interpretation

At the selected threshold, the final holdout evaluation detected 11 fail cases and missed 10 fail cases. It also flagged 56 pass cases for review, producing a review rate of 21.3%.

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

This model is an analytical and portfolio workflow artifact. It is not production-ready and has not been validated with real fab stakeholders.
