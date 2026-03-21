# Project Walkthrough

## 1. Project Overview

This is my Module 4 project for **EAI6010: Applications of Artificial Intelligence**.

I used the **UCI SECOM dataset** to study a semiconductor pass/fail prediction problem with sensor data. The main idea of the assignment was to take an existing tutorial and revise it for a different dataset.

For this project, I did not want to just swap in a new file and keep the same logic. I wanted to rebuild the workflow so it fit this dataset better and was easier to defend.

See:
- Final notebook: [`../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb`](../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb)
- Final assignment report: [`../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf`](../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf)
- Portfolio PDF version: [`../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf`](../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf)

## 2. Business Problem

I framed this as a semiconductor quality-monitoring problem.

The question is simple:

**Given sensor measurements from a manufacturing process, can I predict whether a sample is more likely to pass or fail?**

This kind of setup can be useful for:
- early defect screening
- quality-risk review
- identifying suspicious process patterns earlier

At the same time, I do not treat this model as a final automated decision tool. I see it more as a prototype or screening workflow.

## 3. Why I Picked This Dataset

I picked the UCI SECOM dataset because I wanted a semiconductor-related problem that was closer to a real manufacturing setting.

The dataset also has the kind of challenges that make the project more meaningful:
- high dimensionality
- many missing values
- strong class imbalance
- timestamps that may suggest process change over time

That made it a better fit for careful workflow design, not just model training.

## 4. Dataset Used

Main files:
- [`../data/secom.data`](../data/secom.data)
- [`../data/secom_labels.data`](../data/secom_labels.data)
- [`../data/secom.names`](../data/secom.names)

Main project data:
- 1567 rows
- 590 sensor features
- 1463 pass samples
- 104 fail samples

More details:
- [`../data/README.md`](../data/README.md)

## 5. What I Changed from the Original Tutorial Idea

The original inspiration was a semiconductor-related tutorial idea, but I did not directly reuse an image-based workflow.

This dataset is tabular sensor data, so I changed the project in these ways:

- I used a tabular modeling pipeline
- I split the data before fitting preprocessing
- I dropped high-missing columns based on the training split only
- I used median imputation instead of a simpler full-data shortcut
- I removed constant features
- I used PCA only where it made sense
- I compared multiple models instead of trusting one model only
- I tuned the classification threshold on the validation set

That made the final workflow cleaner and more realistic.

## 6. My Workflow

My final workflow was:

1. Load the SECOM sensor and label files
2. Create the pass/fail target
3. Parse timestamp fields for exploratory analysis
4. Audit missing values, class balance, and low-information features
5. Split into train / validation / test
6. Fit preprocessing on training data only
7. Build separate preprocessing paths for:
   - linear models / MLP
   - tree-based model
8. Compare multiple models
9. Tune threshold on validation balanced accuracy
10. Evaluate the selected model once on the untouched test set

## 7. Key Data Checks

Before modeling, I looked at:
- target class distribution
- missing ratio across sensors
- timestamp-based weekly fail-rate trend

This part helped me see that:
- the fail class is rare
- many features have missing values
- some features likely carry very little signal
- the process may not be perfectly stationary over time

Figure:
- ![EDA overview](../outputs/figures/eda-overview.png)

## 8. Preprocessing Design

One of the biggest improvements in this project was the preprocessing logic.

### Main choices
- training-only missingness filter
- median imputation
- constant-feature removal
- scaling for linear / neural-network path
- PCA for linear / neural-network path
- simpler imputation-only path for Random Forest

### Why this mattered
I wanted the workflow to be easier to explain and less vulnerable to leakage.

A simple shortcut may look fine in a classroom setting, but it becomes harder to defend when the dataset is small, imbalanced, and noisy.

## 9. Models Compared

I compared four models:

### 1. Dummy Classifier
This was my sanity-check baseline.

### 2. Logistic Regression + PCA
This was my simple and more interpretable baseline.

### 3. Random Forest
This was my stronger nonlinear baseline.

### 4. Weighted MLP
I also built a class-weighted PyTorch MLP to stay aligned with the AI course.

## 10. Selected Code

### Data loading
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

## 11. Validation Results

On the validation split:

- **Random Forest** had the best balanced accuracy
- **Logistic Regression + PCA** stayed competitive on some other metrics
- **Weighted MLP** pushed recall high, but with very low precision and much lower overall accuracy

That was actually one of the useful findings from this project:

**A neural network is not automatically the best choice for a small, imbalanced tabular problem.**

Figure:
- ![MLP training history](../outputs/figures/mlp-training-history.png)

## 12. Final Model Selection

I selected **Random Forest** for final test evaluation because it had the strongest validation balanced accuracy.

I treated that as a split-specific decision, not as proof that it would always win on every future resample.

## 13. Final Test Results

### Random Forest test metrics
- Accuracy: 0.7898
- Balanced accuracy: 0.6663
- Recall: 0.5238
- ROC-AUC: 0.7978
- PR-AUC: 0.2192

### Confusion matrix read
The model correctly detected **11 of the 21 fail cases** on the test set, but it also produced many false positives.

Figures:
- ![Confusion matrix](../outputs/figures/test-confusion-matrix-random-forest.png)
- ![ROC curve](../outputs/figures/roc-curve-random-forest.png)
- ![Precision-recall curve](../outputs/figures/precision-recall-curve-random-forest.png)

## 14. Feature Importance

I also reviewed the top Random Forest feature importances.

I used this only as a model interpretation step, not as proof of physical causality. In a real semiconductor setting, these signals would still need process knowledge and engineering validation.

Figure:
- ![Random Forest feature importances](../outputs/figures/random-forest-feature-importances.png)

## 15. What I Learned

This project helped me practice a more realistic analyst mindset.

I learned that:
- a cleaner workflow matters more than flashy results
- imbalanced classification needs better metrics than plain accuracy
- threshold choice changes the real operating behavior of a model
- tabular industrial data can punish overly simple modeling choices
- strong deployment thinking includes limits, monitoring, and retraining questions

## 16. Final Conclusion

This project is a stronger analytical prototype, but not a production-ready manufacturing model yet.

I would still want:
- repeated validation or cross-validation
- time-based validation
- drift monitoring
- threshold setting with engineering or business cost in mind
- more validation of the most important sensor signals

Overall, I think this is a good example of taking an off-the-shelf modeling idea and turning it into a more careful and more honest workflow.

## 17. Related Files

- Notebook: [`../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb`](../notebooks/EAI6010_Module_4_Assignment_V2_Cheng_L.ipynb)
- Assignment report: [`../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf`](../reports/EAI6010_Module_4_Assignment_V2_Cheng_L.pdf)
- Portfolio PDF: [`../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf`](../reports/EAI6010_SECOM_Portfolio_Cheng_Liu.pdf)
- Data note: [`../data/README.md`](../data/README.md)
- Figures note: [`../outputs/README.md`](../outputs/README.md)