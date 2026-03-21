# Data Note

This folder contains the public dataset files used in this project.

## Files

- `secom.data`
- `secom_labels.data`
- `secom.names`

## What each file is for

### `secom.data`
This is the main sensor dataset used for modeling.

In the notebook, it is loaded as the feature matrix and the columns are renamed to:
- `sensor_001`
- `sensor_002`
- ...
- `sensor_590`

### `secom_labels.data`
This file contains the target label and timestamp-related fields.

In the notebook, it is used to create:
- the pass/fail target
- timestamp-based exploratory checks

The label is converted as:
- `-1 -> 0` for pass
- `1 -> 1` for fail

### `secom.names`
This file is the metadata file from UCI. It gives the dataset description and background information.

I kept it here as a reference file for documentation and reproducibility.

## Source

These files come from the **UCI Machine Learning Repository - SECOM dataset**.

Direct download links:

- `secom.data`  
  https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data

- `secom_labels.data`  
  https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data

- `secom.names`  
  https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.names

## Why I included the raw files

I included the raw files in this repo because:

- they are public academic data
- the files are small enough for GitHub
- it makes the notebook easier to rerun
- it helps keep the project reproducible

## Public repo note

For this portfolio repo, I am including the original SECOM files directly because they are public files from the UCI repository and were already used as the source in my notebook workflow.

If needed, the notebook can also re-download the same files from UCI using the links above.

## Small note on feature count

In my notebook, `secom.data` is loaded as **590 sensor columns**.

The UCI metadata file may describe the dataset slightly differently at the documentation level, so I follow the actual notebook-loaded file structure here because that is what the project code uses.