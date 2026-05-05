# Output Figure Note

This folder contains selected exported figures from the final notebook. They are included to make the main project evidence easier to review from the repository.

## Figure Files

| File | What it shows | How to read it |
|---|---|---|
| `figures/eda-overview.png` | Target class distribution, missing-ratio distribution, and weekly fail-rate trend | Shows why the dataset needs careful handling: imbalance, missingness, and possible time variation |
| `figures/mlp-training-history.png` | Training loss and validation balanced accuracy for the weighted MLP | Supports the model comparison story; the MLP was not the final selected model |
| `figures/test-confusion-matrix-random-forest.png` | Final holdout confusion matrix for the selected Random Forest | Shows the screening trade-off: some fail cases detected, but many false positives |
| `figures/roc-curve-random-forest.png` | ROC curve for the final Random Forest test evaluation | Useful for model ranking signal, but should be read with the PR curve because the target is imbalanced |
| `figures/precision-recall-curve-random-forest.png` | Precision-recall curve for the final Random Forest test evaluation | Important for understanding performance on the small fail class |
| `figures/random-forest-feature-importances.png` | Top 20 Random Forest feature importances | Shows model-important anonymous sensor variables, not confirmed physical root causes |

## Notes on Interpretation

- The confusion matrix is the clearest visual for the final screening trade-off.
- The ROC curve should not be shown alone because the fail class is small.
- The precision-recall curve gives a more direct view of the minority fail class.
- The feature importance chart should be described carefully because the sensor variables are anonymous.
- The MLP history is included as part of model comparison, not because the project is mainly a deep learning project.

## Reproducibility Note

These figures are selected notebook-result visuals that were preserved for the GitHub version of the project. The current notebook displays these figures during analysis, but it is not set up as an automated export pipeline that regenerates every file in `outputs/figures/`.
