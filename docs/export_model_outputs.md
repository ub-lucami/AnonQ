# export_model_outputs.py — Script Description

## Summary
A utility script that loads trained model outputs and exports evaluation and prediction artifacts for downstream analysis and reporting. It orchestrates reading model results, formatting outputs, computing summary metrics, and saving CSV/plot/table outputs used by the project's evaluation and reporting pipeline.

## Purpose
- Centralize exporting of model predictions, per-event and per-period evaluation metrics, and summary tables.
- Produce files compatible with the repository's `eval_outputs/`, `figures/`, and `tables/` directories for use in reports and further analysis.

## Main Responsibilities
- Load model predictions and ground-truth labels (from model output files or in-memory objects).
- Compute evaluation metrics (e.g., per-event F1, per-period F1, aggregated summary metrics).
- Save generated CSVs/tables to `eval_outputs/` and other output folders.
- Optionally generate figures or formatted LaTeX tables for reporting.

## Typical Inputs
- Model prediction files (CSV/serialized outputs) or model objects.
- Validation/test labels and any required metadata (e.g., GUIDs, timestamps, classes).
- Config or CLI options controlling output paths and which metrics to compute.

## Typical Outputs
- CSV summary files (e.g., `per_event_f1.csv`, `per_period_f1.csv`, `summary_metrics.csv`) placed in `eval_outputs/`.
- Tables and figures for reports (saved under `tables/`, `figures/`, or `reports/`).
- Optional logs or diagnostic files.

## Key Functions / Flow (high-level)
- `load_predictions()`: Read model outputs and align them with ground truth.
- `compute_metrics()`: Calculate per-event/per-period metrics and aggregate summaries.
- `format_outputs()`: Convert metrics into CSV/LaTeX-friendly tables and dataframes.
- `save_outputs()`: Write CSVs, LaTeX files, and figures into project folders.

## Dependencies
- Typical data science stack: `pandas`, `numpy`, (possibly) `scikit-learn` for metrics, and plotting libraries (`matplotlib`/`seaborn`) if figures are produced.
- Project-specific modules and helper functions from the repository (for consistent formatting and reporting).

## Usage
- Run as a script to regenerate the evaluation outputs after model training or when updated predictions are available.
- Can be integrated into evaluation pipelines or called from other scripts (e.g., `evaluate_utility_extended.py`) that produce model outputs.

## Notes & Recommendations
- Ensure consistent input schema (columns like `guid`, `timestamp`, `true_label`, `pred_label`, `prob_*`) for correct alignment.
- Confirm output directory paths exist (`eval_outputs/`, `tables/`, `figures/`) or allow the script to create them.
- If large datasets are used, consider streaming or chunked processing to limit memory use.

Would you like this expanded into a function-level breakdown based on the actual code, or saved as-is? 

## Function-level breakdown (based on code)

- `load_master()`:
	- Purpose: Read the master anonymized events CSV (`MASTER_CSV`) using `SEP`, coerce key columns to expected dtypes, and return a DataFrame.
	- Inputs: None (uses global `MASTER_CSV` and `SEP`).
	- Outputs: `pd.DataFrame` with columns like `weekday`, `time_period`, `generalized_event`, `GUID`, `week_number`.

- `build_datasets(df)`:
	- Purpose: Create three dataset variants used by experiments: `season` (rows with week_number), `no_season` (drop week_number), and `hierarchical` (week filled with 'unknown' + missing flag).
	- Inputs: master DataFrame.
	- Outputs: tuple `(season, no_season, hier)` DataFrames.

- `prepare_xy(df, mode)`:
	- Purpose: Encode categorical features and labels for model training. Uses `LabelEncoder` for `time_period`, `GUID`, `generalized_event`, and `week_number` (as needed).
	- Inputs: DataFrame and `mode` in `['season','no_season','hierarchical']`.
	- Outputs: `(X, y, df)` where `X` is feature DataFrame, `y` is encoded target array, and `df` is the (possibly modified) DataFrame. Also sets global `EVENTS` to decoded class names.
	- Notes: Encoders are local to each call; encodings may differ across separate calls/modes.

- `save_shap_summary(model, X_te, mode, use_kernel_fallback=False)`:
	- Purpose: Compute and save a SHAP summary dot-plot PDF. Prefers `TreeExplainer` with `model_output='raw'`, falls back to `KernelExplainer` when required. Adjusts SHAP arrays to match feature count.
	- Inputs: trained `model`, test-features `X_te`, `mode` string, and fallback flag.
	- Outputs: Writes `figures/shap_summary_{mode}.pdf`. Returns `True` on success, `False` if SHAP unavailable or failed.
	- Notes: Requires optional `shap` package (`HAS_SHAP` flag). Handles multiclass outputs and pads/trims arrays to match `X_te` columns.

- `train_and_export(df, mode, seed=42)`:
	- Purpose: Full training and export pipeline for a single `mode`:
		- Prepare features/labels, train `LGBMClassifier`, evaluate on test split.
		- Export classification report (`reports/{mode}_classification_report.tex`), confusion matrix PDF (`figures/confmat_{mode}.pdf`), feature importance table (`tables/feature_importance_{mode}.tex`), per-GUID CSV (`tables/per_guid_{mode}.csv`), and SHAP summaries.
		- Compute per-period × per-class F1 scores (returned as `per_rows`).
	- Inputs: dataset DataFrame, `mode`, optional `seed`.
	- Outputs: `per_rows` list for later aggregation; multiple side-effect files written to `reports/`, `figures/`, and `tables/`.
	- Notes: Uses `sklearn` metrics and `LGBMClassifier`. Writes LaTeX via `DataFrame.to_latex()` and saves plots headlessly.

- `build_per_period_table(per_rows)`:
	- Purpose: Aggregate a list of per-period × per-class F1 rows into a DataFrame and write `tables/per_period_per_class_f1.tex`.
	- Inputs: `per_rows` as returned by multiple calls to `train_and_export()`.
	- Outputs: LaTeX table file under `tables/`.

- `build_guid_tables()`:
	- Purpose: Read existing `per_guid_{mode}.csv` files, combine modes, compute average GUID macro-F1, and write top/bottom 10 GUID tables (`per_guid_top10.tex` / `per_guid_bottom10.tex`).
	- Inputs: none (reads CSVs under `tables/`).
	- Outputs: two LaTeX table files under `tables/`.

- `main()`:
	- Purpose: Orchestrates the full export pipeline: load master CSV, build dataset variants, run `train_and_export()` for `season`, `hierarchical`, and `no_season`, then build combined tables.
	- Side effects: creates directories `figures/`, `tables/`, `reports/` (if missing); writes multiple output files and prints completion message.

### Globals & Config

- `MASTER_CSV`, `SEP`: path and separator for the input CSV (note: `MASTER_CSV` is a hard-coded absolute path in the script).
- `OUT_FIG`, `OUT_TAB`, `OUT_REP`: output directories, created at import time.
- `TIME_PERIODS`: order used for per-period evaluation.
- `EVENTS`: global populated by `prepare_xy()` and used in report labeling.
- `HAS_SHAP`: runtime flag whether `shap` is installed.

### Key caveats

- The script uses a hard-coded `MASTER_CSV` path; update before running on a different system.
- Label encoders are fit per call of `prepare_xy()` which may produce differing integer encodings across modes.
- SHAP plotting is optional and will be skipped if `shap` is not installed.

