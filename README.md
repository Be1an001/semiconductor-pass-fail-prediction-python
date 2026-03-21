\# Semiconductor Pass/Fail Prediction with the UCI SECOM Dataset



This is my Module 4 project for \*\*EAI6010: Applications of Artificial Intelligence\*\*.



I used the \*\*UCI SECOM dataset\*\* to study a semiconductor pass/fail prediction problem with sensor data. The main goal was not just to run one model. I wanted to adapt an off-the-shelf tutorial idea into a cleaner workflow that fit this dataset better.



In this final version, I used a leakage-safe train/validation/test setup, compared multiple models, and evaluated results with metrics that make more sense for an imbalanced pass/fail problem.



\## Project Type



\- \*\*Type:\*\* Individual course project

\- \*\*Course:\*\* EAI6010: Applications of Artificial Intelligence

\- \*\*Module:\*\* Module 4 - Using an Off-the-shelf Model

\- \*\*Focus:\*\* Semiconductor quality-monitoring style classification



\## Business Problem



This project is framed as a semiconductor quality-monitoring problem.



Given sensor measurements from a manufacturing process, can I identify whether a sample is more likely to pass or fail? In practice, a model like this could be more useful as an early screening tool than as a fully automated final decision system.



\## Project Goal



My goal was to take an existing tutorial idea and rebuild it for a different dataset and a more realistic tabular workflow.



I wanted to:



\- use a semiconductor-related dataset

\- handle missing values more carefully

\- avoid data leakage

\- compare strong baselines instead of depending on only one model

\- use evaluation metrics that fit an imbalanced classification problem

\- think honestly about deployment limits



\## Dataset



This project uses the \*\*UCI SECOM dataset\*\*.



Main files in this repo:



\- `data/secom.data` - sensor features

\- `data/secom\_labels.data` - labels and timestamp-related fields

\- `data/secom.names` - dataset metadata from UCI



Based on the notebook:



\- \*\*Rows:\*\* 1567

\- \*\*Sensor features:\*\* 590

\- \*\*Target distribution:\*\* 1463 pass / 104 fail



More details are in \[`data/README.md`](data/README.md).



\## Tools and Methods



\### Tools

\- Python

\- Jupyter Notebook

\- pandas

\- numpy

\- scikit-learn

\- matplotlib

\- seaborn

\- PyTorch



\### Workflow

\- data loading

\- missing-value audit

\- timestamp-based exploratory analysis

\- train / validation / test split

\- training-only missingness filtering

\- median imputation

\- constant-feature removal

\- scaling and PCA

\- model comparison

\- threshold tuning

\- final holdout test evaluation



\### Models Compared

\- Dummy Classifier

\- Logistic Regression + PCA

\- Random Forest

\- class-weighted PyTorch MLP



\## Main Results



On the validation split, \*\*Random Forest\*\* had the strongest balanced accuracy among the compared models, so I selected it for final test evaluation.



\### Final test results for Random Forest

\- \*\*Accuracy:\*\* 0.7898

\- \*\*Balanced accuracy:\*\* 0.6663

\- \*\*Recall:\*\* 0.5238

\- \*\*ROC-AUC:\*\* 0.7978

\- \*\*PR-AUC:\*\* 0.2192



\### Quick interpretation

The model correctly detected \*\*11 of the 21 fail cases\*\* on the test set, but it also produced many false positives.



Because of that, I see this as a stronger analytical prototype or screening model, not something I would deploy directly as a final automated pass/fail decision system.



\## Key Figures



\### EDA overview

!\[EDA overview](outputs/figures/eda-overview.png)



\### MLP training history

!\[MLP training history](outputs/figures/mlp-training-history.png)



\### Random Forest confusion matrix

!\[Random Forest confusion matrix](outputs/figures/test-confusion-matrix-random-forest.png)



\### ROC curve

!\[ROC curve](outputs/figures/roc-curve-random-forest.png)



\### Precision-recall curve

!\[Precision-recall curve](outputs/figures/precision-recall-curve-random-forest.png)



\### Random Forest feature importances

!\[Random Forest feature importances](outputs/figures/random-forest-feature-importances.png)



\## Repository Guide



\- Final notebook: \[`notebooks/EAI6010\_Module\_4\_Assignment\_V2\_Cheng\_L.ipynb`](notebooks/EAI6010\_Module\_4\_Assignment\_V2\_Cheng\_L.ipynb)

\- Final assignment report: \[`reports/EAI6010\_Module\_4\_Assignment\_V2\_Cheng\_L.pdf`](reports/EAI6010\_Module\_4\_Assignment\_V2\_Cheng\_L.pdf)

\- Portfolio PDF version: \[`reports/EAI6010\_SECOM\_Portfolio\_Cheng\_Liu.pdf`](reports/EAI6010\_SECOM\_Portfolio\_Cheng\_Liu.pdf)

\- Walkthrough version: \[`walkthrough/project-walkthrough.md`](walkthrough/project-walkthrough.md)

\- Dataset note: \[`data/README.md`](data/README.md)

\- Figure note: \[`outputs/README.md`](outputs/README.md)



\## How to Run



1\. Clone or download this repository.

2\. Install the required Python packages:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt

&#x20;  ```

3\. Open the notebook:

&#x20;  - `notebooks/EAI6010\_Module\_4\_Assignment\_V2\_Cheng\_L.ipynb`

4\. Run the notebook cells in order.



The notebook can use the local files in the `data/` folder. It also includes logic to download the UCI files if needed.



\## What This Project Shows



This project shows that I can:



\- work with messy tabular sensor data

\- handle missing values and imbalanced classes more carefully

\- build a leakage-safe workflow

\- compare simple and nonlinear models

\- evaluate results beyond plain accuracy

\- explain model limits honestly instead of overselling the result



\## Limitations



This is still not a production-ready manufacturing model.



Main limits:

\- small minority class

\- many missing values

\- high-dimensional noisy features

\- possible process shift over time

\- split-specific model selection

\- threshold still needs business or engineering cost discussion



For a more deployment-oriented version, I would still want:

\- repeated validation or cross-validation

\- time-based validation

\- drift monitoring

\- threshold setting with domain stakeholders

\- more engineering validation for important sensor signals



\## Contribution Note



This is an \*\*individual course project\*\*.



I selected the dataset, rebuilt the workflow, ran the analysis, compared the models, interpreted the results, and wrote the final report and portfolio version.



\## References



\- UCI Machine Learning Repository. SECOM Dataset.

\- Vinit. Semiconductor/IOT-MachineLearning. Kaggle.

\- Howard, J., \& Gugger, S. \*Deep Learning for Coders with Fastai and PyTorch\*. O'Reilly.

