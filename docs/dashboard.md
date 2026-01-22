# dashboard.py — Script Description

## Summary
Interactive Streamlit dashboard for exploring anonymized event data and training local models (LightGBM classifier and an LSTM sequence model). Provides visualization, per-GUID inspection, on-demand training, and a small prediction demo.

## Purpose
- Enable secure, local analysis of `anonymized_events.csv` via an easy UI.
- Visualize distributions and heatmaps of temporal/event features.
- Train and evaluate a LightGBM classifier for event prediction.
- Train an LSTM to predict the next event in a GUID's sequence and demo a single-GUID prediction.

## Typical Inputs
- A user-uploaded CSV (semicolon-separated) with columns such as `GUID`, `week_number`, `weekday`, `time_period`, `generalized_event`.

## Typical Outputs
- Interactive Streamlit views: tables, charts, heatmaps, classification report text, confusion matrix, and training metrics printed to the UI.
- No files are saved by the script — trained models remain in memory during the session.

## Key Features / Flow
- `load_data(path)`: cached data loader for the uploaded CSV (uses `sep=';'`).
- Data overview: shows head of dataset and GUID distribution summary plus a bar chart.
- Distribution plots: counts by `week_number`, `weekday`, `time_period`, and `generalized_event`.
- Heatmaps: `week_number × time_period` and `weekday × generalized_event` pivot heatmaps.
- LightGBM training: encodes categorical fields, builds cyclical time features (`sin`/`cos`), trains `LGBMClassifier`, shows `classification_report` and confusion matrix for a test split.
- LSTM training: builds per-GUID sequences, filters GUIDs with enough events, creates sequence windows (default SEQ_LEN=10), trains a small Keras `Sequential` model and reports train/validation accuracy.
- Prediction demo: select a GUID and (if models exist and sequence length sufficient) predict the next event using the trained LSTM.

## Functions & Blocks (detailed)
- `load_data(path)`
  - Purpose: Read uploaded CSV into a `pd.DataFrame` and cache results for the session.
  - Notes: Expects `;` separator.

- LightGBM block (triggered by `st.button("Treniraj LightGBM model")`)
  - Encodes `GUID`, `time_period`, and `generalized_event` with `LabelEncoder`.
  - Adds cyclical features for `week_number` and `weekday` (`sin`/`cos`).
  - Trains `LGBMClassifier` with typical hyperparameters and evaluates on a stratified 80/20 split.
  - Outputs classification report and confusion matrix to the UI.

- LSTM block (triggered by `st.button("Treniraj LSTM model")`)
  - Sorts events by `GUID` and temporal order, builds sequences per GUID, and filters short sequences.
  - Creates sliding windows of length `SEQ_LEN` (10) to predict the next event.
  - Trains a small Embedding+LSTM model using Keras and reports accuracy.
  - Note: Training runs in the Streamlit process and can be slow for large datasets.

- Prediction demo
  - Lets user pick a GUID and attempts to predict the next event using the trained LSTM model and `le_event` decoder.
  - Requires at least 10 prior events for the selected GUID and an in-memory trained `lstm_model`.

## Dependencies
- `streamlit`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `lightgbm`, `tensorflow`/`keras`.

## Usage
Run the dashboard locally and upload the CSV:

```bash
streamlit run dashboard.py
```

Then upload `anonymized_events.csv` in the UI and use the buttons to train models and preview predictions.

## Notes & Recommendations
- For large datasets, avoid training the LSTM in the UI; provide a separate training script or persisted model.
- Add error handling for missing columns and non-numeric `week_number` values.
- Persist trained models to disk or use `st.cache_resource` to reuse across interactions.
- Set random seeds for reproducible splits/training if deterministic behavior is needed.
