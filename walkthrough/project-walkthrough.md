# Project Walkthrough

This walkthrough explains the main analysis decisions behind the semiconductor pass/fail prediction project. It is meant to be read with the notebook, reports, data note, and output figures.

See:

- Main notebook: [`../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb`](../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb)
- Assignment report: [`../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf`](../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf)
- Portfolio PDF: [`../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf`](../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf)

## Project Overview

This was an individual Module 4 project for **EAI6010: Applications of Artificial Intelligence**. I used the UCI SECOM dataset to study a semiconductor pass/fail prediction problem with sensor data.

The assignment started from the idea of adapting an off-the-shelf model. I changed the project into a tabular applied ML workflow because the SECOM data is made of numeric sensor measurements, not images.

This project is best understood as a screening-style ML prototype. It is not a deployed manufacturing system.

## Business Problem

The project asks:

**Given semiconductor sensor measurements, can a model help flag units that are more likely to fail?**

This kind of analysis can support:

- early defect screening
- quality-risk review
- manual review prioritization
- discussion of process signals that may deserve engineering follow-up

The model result is not treated as an automatic pass/fail decision. In a real manufacturing setting, the threshold, sensor availability, drift risk, and false-positive/false-negative costs would need review with process or quality engineers.

## Dataset

Main data files:

- [`../data/secom.data`](../data/secom.data)
- [`../data/secom_labels.data`](../data/secom_labels.data)
- [`../data/secom.names`](../data/secom.names)

Notebook-loaded data summary:

- 1,567 rows
- 590 loaded sensor features
- 1,463 pass samples
- 104 fail samples
- 6.64% fail rate
- timestamp range from July 2008 to October 2008

The raw label mapping is:

- `-1 -> 0` for pass
- `1 -> 1` for fail

More detail is in [`../data/README.md`](../data/README.md).

## Methodology

The final notebook follows this workflow:

1. Load the SECOM sensor and label files.
2. Convert the raw labels into a binary pass/fail target.
3. Parse timestamps for exploratory analysis.
4. Audit class balance, missing values, and low-information features.
5. Create a stratified train/validation/test split.
6. Fit preprocessing steps on the training data only.
7. Drop columns with more than 50% missing values based on the training split.
8. Apply median imputation to the remaining features.
9. Remove constant features for the linear and neural-network path.
10. Apply scaling and PCA for Logistic Regression and MLP.
11. Use a simpler imputation-only path for Random Forest.
12. Compare several classification models.
13. Tune the classification threshold on validation balanced accuracy.
14. Evaluate the selected model once on the holdout test set.

## Data Checks

Before modeling, the notebook checks:

- target class distribution
- missing-value ratios across sensors
- features with high missingness
- constant and near-constant columns
- timestamp-based weekly fail-rate trend

These checks show why the dataset is difficult:

- the fail class is rare
- many features contain missing values
- some features carry very little information
- weekly fail rates change over time, so future validation should consider time-based splits

Figure:

![EDA overview](../outputs/figures/eda-overview.png)

## Preprocessing Design

The preprocessing design was one of the most important parts of this project.

Main choices:

- split the data before fitting preprocessing
- use training-only missingness filtering
- use median imputation
- remove constant features for the linear and MLP paths
- use scaling and PCA for Logistic Regression and MLP
- keep Random Forest on an imputation-only tree path

This design helps reduce preprocessing leakage. It does not remove all validation risk, because the project still uses a stratified random split rather than a time-based validation setup.

## Models Compared

The notebook compares four models:

| Model | Role in the project |
|---|---|
| Dummy Classifier | Majority-class sanity baseline |
| Logistic Regression + PCA | Simple class-weighted linear baseline |
| Random Forest | Nonlinear tree-based reference model |
| Weighted MLP | PyTorch neural-network comparison model |

The MLP was included partly to stay aligned with the AI course, but the project is not mainly a deep learning project. The final selected model was Random Forest.

## Selected Code Examples

### Data loading and label mapping

```python
X_raw = pd.read_csv(data_file, sep=r"\s+", header=None, engine="python")
X_raw.columns = [f"sensor_{i:03d}" for i in range(1, X_raw.shape[1] + 1)]

labels_df = pd.read_csv(labels_file, sep=r"\s+", header=None, engine="python")
labels_df.columns = ["label", "date_part", "time_part"]

y = labels_df["label"].replace({-1: 0, 1: 1}).astype(int)
```

### Training-only missingness filter

```python
def select_feature_columns_by_missingness(X_train_df, missing_threshold=0.50):
    train_missing_ratio = X_train_df.isna().mean()
    keep_cols = train_missing_ratio[train_missing_ratio <= missing_threshold].index.tolist()
    drop_cols = train_missing_ratio[train_missing_ratio > missing_threshold].index.tolist()
    return keep_cols, drop_cols, train_missing_ratio
```

### Random Forest model

```python
rf_model = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)
```

## Validation Model Comparison

The validation comparison used balanced accuracy as the main selection metric because the target is highly imbalanced.

| Model | Threshold | Accuracy | Balanced accuracy | Precision | Recall | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.110 | 0.7102 | 0.6458 | 0.1277 | 0.5714 | 0.1329 |
| Weighted MLP | 0.125 | 0.3535 | 0.6315 | 0.0901 | 0.9524 | 0.1253 |
| Logistic Regression + PCA | 0.305 | 0.8280 | 0.6205 | 0.1633 | 0.3810 | 0.1509 |
| Dummy Classifier | 0.500 | 0.9331 | 0.5000 | 0.0000 | 0.0000 | 0.0669 |

Random Forest had the strongest validation balanced accuracy on this split. The MLP reached high recall, but with very low precision and lower overall accuracy. Logistic Regression + PCA remained a useful simpler baseline.

Figure:

![MLP training history](../outputs/figures/mlp-training-history.png)

## Final Holdout Evaluation

Random Forest was selected for the holdout test evaluation because it had the strongest validation balanced accuracy.

This selection should be read as split-specific. It does not prove that Random Forest would always be the best model across future samples or time periods.

Final Random Forest test metrics:

| Metric | Value |
|---|---:|
| Threshold | 0.110 |
| Accuracy | 0.7898 |
| Balanced accuracy | 0.6663 |
| Specificity | 0.8089 |
| Precision | 0.1642 |
| Recall | 0.5238 |
| F1 | 0.2500 |
| ROC-AUC | 0.7978 |
| PR-AUC | 0.2192 |

Final test confusion matrix:

|  | Predicted pass | Predicted fail |
|---|---:|---:|
| True pass | 237 | 56 |
| True fail | 10 | 11 |

The model detected 11 of the 21 fail cases in the test set. It also incorrectly flagged 56 pass cases as fail. This is why the result is better described as a screening signal than as an automated decision system.

Figures:

![Confusion matrix](../outputs/figures/test-confusion-matrix-random-forest.png)

![ROC curve](../outputs/figures/roc-curve-random-forest.png)

![Precision-recall curve](../outputs/figures/precision-recall-curve-random-forest.png)

## Feature Importance

The notebook reviews Random Forest feature importances as a model interpretation step.

The most important caution is that these are anonymous sensor variables. The chart can suggest which model inputs were influential, but it does not prove physical root causes in the semiconductor process.

Figure:

![Random Forest feature importances](../outputs/figures/random-forest-feature-importances.png)

## Limitations

This project is not a production-ready manufacturing model.

Main limitations:

- only 104 fail cases are available in the full dataset
- the validation and test splits each contain only 21 fail cases
- model selection is based on one validation split
- the workflow uses a stratified random split, not a time-based validation split
- the final threshold is not based on real business or engineering costs
- feature importance does not prove root causes
- no calibration, deployment, monitoring, dashboard, SQL layer, GenAI component, or MLOps workflow is included
- no real fab stakeholder validation or cost savings are confirmed

## Future Improvements

Good next steps would be:

- repeated stratified cross-validation
- time-based validation using the timestamp fields
- cost-based threshold testing for false positives and false negatives
- probability calibration checks
- clearer documentation of feature anonymity and operational sensor availability
- permutation importance or SHAP with careful non-causal interpretation
- a simple read-only review app for threshold trade-offs, if this project were extended as a portfolio data product

## Related Files

- Main notebook: [`../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb`](../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb)
- Assignment report: [`../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf`](../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf)
- Portfolio PDF: [`../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf`](../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf)
- Data note: [`../data/README.md`](../data/README.md)
- Output figure note: [`../outputs/README.md`](../outputs/README.md)
- Root README: [`../README.md`](../README.md)
