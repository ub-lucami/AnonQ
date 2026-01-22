# lightGBM_model.py — Script Description

## Summary
Simple, self-contained script that trains a LightGBM classifier on the anonymized events dataset and prints evaluation results. It demonstrates feature engineering (cyclical time features), categorical encoding, model training, and basic prediction.

## Purpose
- Provide a lightweight example of training a LightGBM model on the project's event data.
- Produce a quick classification report and confusion matrix for a test split and show a single example prediction.

## Inputs
- Hard-coded CSV: `G:/TS_2025/anonymized_events.csv` (semicolon-separated). Expected columns: `GUID`, `week_number`, `weekday`, `time_period`, `generalized_event`.

## Outputs
- Printed `classification_report` and `confusion_matrix` to stdout.
- Printed one example prediction for a test-row.

## Main Steps / Flow
- Load dataset with `pd.read_csv()`.
- Feature engineering: add cyclical features for `week_number` and `weekday` via sine/cosine transforms.
- Label-encode `GUID`, `time_period`, and `generalized_event`.
- Build feature matrix `X` and target `y`.
- Train/test split with stratification on `y`.
- Train `LGBMClassifier` and fit on training set.
- Predict on test set and print evaluation results.

## Key Code Sections (function-level style)

- Data loading
  - Reads the CSV using `pd.read_csv("G:/TS_2025/anonymized_events.csv", sep=';')`.

- Feature engineering
  - Adds `week_sin`, `week_cos`, `weekday_sin`, `weekday_cos` using numpy trig transforms.

- Encoding
  - `LabelEncoder` applied to `GUID`, `time_period`, and `generalized_event`.

- Modeling
  - Constructs `LGBMClassifier(n_estimators=500, learning_rate=0.03, subsample=0.9)` and fits on train split.

- Evaluation
  - Uses `classification_report` and `confusion_matrix` to summarise performance.
  - Shows one example prediction decoded back to original event label.

## Dependencies
- pandas, numpy, scikit-learn, lightgbm.

## Usage
Run directly with Python:

```bash
python lightGBM_model.py
```

Notes:
- The script uses a hard-coded absolute path for the input CSV; change the path or parameterize it before running on another machine.
- For reproducibility, the random seed for `train_test_split` is fixed to `42`.
- Consider persisting trained models to disk or converting this script into functions for reuse.
