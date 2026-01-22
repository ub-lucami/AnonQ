# lstm_next_event.py — Script Description

## Summary
Training script for an LSTM sequence model that predicts the next `generalized_event` for a GUID based on a fixed-length window of prior events.

## Purpose
- Build and train a small sequence model (Embedding + LSTM) to forecast the next event in a user's event sequence.

## Inputs
- `anonymized_events.csv` (semicolon-separated) with columns at least: `GUID`, `week_number`, `weekday`, `time_period`, `generalized_event`.

## Outputs
- Trains an in-memory Keras `Sequential` model and prints training progress and a short example prediction to stdout. No files are saved by default.

## Main Steps / Flow
- Read CSV and label-encode `generalized_event` and `GUID` using `LabelEncoder`.
- Sort events by `GUID` and temporal order (`week_number`, `weekday`, `time_period`).
- Group events per GUID into integer sequences and filter sequences shorter than 20 events.
- Produce sliding-window training examples with `SEQ_LEN=10`: input windows of 10 event IDs and targets as the next event ID.
- Convert targets to one-hot with `to_categorical` and build an Embedding→LSTM→Dense softmax model.
- Compile with `categorical_crossentropy` and `adam`, train for 15 epochs with `batch_size=64` and `validation_split=0.2`.
- Print a sample inference decoded back to event label using the `LabelEncoder`.

## Key Parameters
- `SEQ_LEN` (default 10): input window length.
- Minimum sequence length filter: 20 events per GUID (hard-coded).
- Model: `Embedding(input_dim=num_events, output_dim=32)`, `LSTM(64)`, `Dense(num_events)`.
- Training: `epochs=15`, `batch_size=64`, `validation_split=0.2`.

## Dependencies
- pandas, numpy, scikit-learn (`LabelEncoder`), and `tensorflow` / `keras`.

## Usage
Run directly:

```bash
python lstm_next_event.py
```

Considerations:
- Large datasets can make training slow and memory-intensive; consider subsampling or using a GPU.
- Convert the script into functions and persist the trained model with `model.save()` for reuse in production or downstream demos.
- Add input validation for missing columns and non-numeric `week_number` values.
