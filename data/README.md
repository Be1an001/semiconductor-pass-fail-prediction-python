# Data Note

This folder contains the public UCI SECOM dataset files used by the notebook and script workflow.

## Files

| File | Purpose |
|---|---|
| `secom.data` | Main sensor feature matrix used for modeling |
| `secom_labels.data` | Pass/fail labels and timestamp fields |
| `secom.names` | UCI metadata and dataset background |

## Dataset Source

These files come from the **UCI Machine Learning Repository - SECOM dataset**.

Direct source links:

- `secom.data`: https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data
- `secom_labels.data`: https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data
- `secom.names`: https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.names

## How the Project Uses the Files

### `secom.data`

The notebook and scripts load this file as the feature matrix and rename the columns as:

- `sensor_001`
- `sensor_002`
- ...
- `sensor_590`

The sensor variables are anonymous, so they should be treated as model input features rather than named process measurements.

### `secom_labels.data`

The notebook and scripts load this file to create:

- the binary pass/fail target
- timestamp-based exploratory checks

The label is converted as:

- `-1 -> 0` for pass
- `1 -> 1` for fail

### `secom.names`

This file is kept as the UCI metadata reference. It describes the semiconductor manufacturing setting, the public dataset source, missing values, and the original SECOM task background.

## Project Data Summary

Based on the final notebook:

- **Rows:** 1,567
- **Loaded sensor features:** 590
- **Pass samples:** 1,463
- **Fail samples:** 104
- **Fail rate:** 6.64%
- **Timestamp range:** July 2008 to October 2008

The UCI metadata may describe the dataset at a slightly different attribute-count level. For this project, the README follows the actual notebook-loaded structure: 590 sensor columns from `secom.data`, with labels and timestamps read from `secom_labels.data`.

## Public Repository Note

The raw SECOM files are included here because they are public academic dataset files and small enough for this repository. Keeping them in the repo makes the final notebook and scripts easier to review and rerun.

The script workflow and final portfolio notebook expect these local files to be present and check for them before running.

## Caution

The dataset uses anonymous sensor variables. Feature importance results in this project should be read as model-important inputs, not as confirmed physical root causes of semiconductor failure.
