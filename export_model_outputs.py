
#!/usr/bin/env python3
"""
export_model_outputs.py
Exports all original model artefacts needed for the LaTeX paper.

Outputs:
- reports/*.tex                      classification reports
- figures/confmat_*.pdf             confusion matrices
- figures/shap_summary_*.pdf        SHAP dotplots (TreeExplainer or Kernel fallback)
- tables/per_period_per_class_f1.tex
- tables/feature_importance_*.tex
- tables/per_guid_top10.tex, tables/per_guid_bottom10.tex

Requires: anonymized_events.csv in same folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # force headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

from lightgbm import LGBMClassifier

# Optional SHAP
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    print("[WARNING] SHAP not installed. SHAP plots will be skipped.")

# Output dirs
OUT_FIG = "figures"
OUT_TAB = "tables"
OUT_REP = "reports"
os.makedirs(OUT_FIG, exist_ok=True)
os.makedirs(OUT_TAB, exist_ok=True)
os.makedirs(OUT_REP, exist_ok=True)

MASTER_CSV = "G:/TS_2025/anonymized_events.csv"
SEP = ";"

TIME_PERIODS = ["night", "morning", "daytime", "afternoon"]
EVENTS = None  # Will fill after first encoding


# ---------------------------------------------------------------------
# DATA LOADING & DATASET CONSTRUCTION
# ---------------------------------------------------------------------
def load_master():
    df = pd.read_csv(MASTER_CSV, sep=SEP)
    df["weekday"] = df["weekday"].astype(int)
    df["time_period"] = df["time_period"].astype(str)
    df["generalized_event"] = df["generalized_event"].astype(str)
    df["GUID"] = df["GUID"].astype(str)
    df["week_number"] = df["week_number"].replace("", np.nan)
    return df


def build_datasets(df):
    season = df[df["week_number"].notna()].copy()

    no_season = df.copy()
    no_season = no_season.drop(columns=["week_number"], errors="ignore")

    hier = df.copy()
    hier["week_number_filled"] = hier["week_number"].fillna("unknown")
    hier["wk_missing_flag"] = hier["week_number"].isna().astype(int)

    return season, no_season, hier


# ---------------------------------------------------------------------
# FEATURE ENCODING
# ---------------------------------------------------------------------
def prepare_xy(df, mode):
    global EVENTS

    df = df.copy()
    le_event = LabelEncoder()
    le_tp    = LabelEncoder()
    le_guid  = LabelEncoder()
    le_week  = LabelEncoder()

    df["time_period_enc"] = le_tp.fit_transform(df["time_period"])
    df["GUID_enc"] = le_guid.fit_transform(df["GUID"])
    y = le_event.fit_transform(df["generalized_event"])
    EVENTS = list(le_event.classes_)

    if mode == "season":
        df["week_number_enc"] = le_week.fit_transform(df["week_number"].astype(str))
        X = df[["weekday", "time_period_enc", "week_number_enc", "GUID_enc"]]

    elif mode == "no_season":
        X = df[["weekday", "time_period_enc", "GUID_enc"]]

    else:  # hierarchical
        df["week_number_enc"] = le_week.fit_transform(df["week_number_filled"].astype(str))
        X = df[["weekday", "time_period_enc", "week_number_enc", "wk_missing_flag", "GUID_enc"]]

    return X, y, df


# ---------------------------------------------------------------------
# ROBUST SHAP (TreeExplainer + Kernel fallback + dimension fixes)
# ---------------------------------------------------------------------
def save_shap_summary(model, X_te, mode, use_kernel_fallback=False):
    """
    Computes and saves a SHAP summary dot-plot PDF for the given mode.
    - Uses TreeExplainer with model_output='raw' (required for tree_path_dependent).
    - Falls back to KernelExplainer for 'no_season' if TreeExplainer fails.
    - Ensures SHAP array width matches X_te feature count (trim/pad).
    """
    if not HAS_SHAP:
        return False

    feature_names = list(X_te.columns)
    X_np = X_te.to_numpy()
    n_features = X_np.shape[1]

    try:
        if not use_kernel_fallback:
            # TreeExplainer: raw output is required with tree_path_dependent
            explainer = shap.TreeExplainer(
                model,
                model_output="raw",
                feature_perturbation="tree_path_dependent"
            )
            shap_vals = explainer.shap_values(X_np, check_additivity=False)
        else:
            # Kernel fallback: robust but slower (use small background)
            bg = X_np[:min(200, len(X_np))]
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_vals = explainer.shap_values(X_np)

        # Ensure SHAP array has same width as X_te
        if isinstance(shap_vals, list):  # multiclass
            fixed = []
            for arr in shap_vals:
                arr = np.abs(arr)
                if arr.shape[1] > n_features:
                    arr = arr[:, :n_features]
                elif arr.shape[1] < n_features:
                    arr = np.pad(arr, ((0,0),(0, n_features - arr.shape[1])), mode="constant")
                fixed.append(arr)
            shap_array = np.mean(np.stack(fixed, axis=0), axis=0)
        else:
            arr = np.abs(shap_vals)
            if arr.shape[1] > n_features:
                arr = arr[:, :n_features]
            elif arr.shape[1] < n_features:
                arr = np.pad(arr, ((0,0),(0, n_features - arr.shape[1])), mode="constant")
            shap_array = arr

        shap.summary_plot(
            shap_array,
            features=X_np,
            feature_names=feature_names,
            show=False,
            plot_type="dot"
        )
        outpath = os.path.join(OUT_FIG, f"shap_summary_{mode}.pdf")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        print(f"[SHAP] Saved: {outpath}")
        return True

    except Exception as e:
        print(f"[SHAP WARNING] TreeExplainer failed for '{mode}': {e}")
        if not use_kernel_fallback and mode == "no_season":
            print("[SHAP] Retrying via KernelExplainer...")
            return save_shap_summary(model, X_te, mode, use_kernel_fallback=True)
        return False


# ---------------------------------------------------------------------
# TRAINING, EXPORTING ALL OUTPUTS
# ---------------------------------------------------------------------
def train_and_export(df, mode, seed=42):
    X, y, dfx = prepare_xy(df, mode)
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, dfx,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ---- classification report
    rep = classification_report(
        y_te, y_pred,
        target_names=EVENTS,
        output_dict=True,
        zero_division=0
    )
    rep_df = pd.DataFrame(rep).T
    out_rep = os.path.join(OUT_REP, f"{mode}_classification_report.tex")
    rep_df.to_latex(out_rep, na_rep="-", float_format=lambda x: f"{x:.4f}")
    print(f"[REPORT] {out_rep}")

    # ---- confusion matrix
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(EVENTS))))
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=EVENTS, yticklabels=EVENTS,
        cbar=False
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({mode})")
    plt.tight_layout()
    cm_path = os.path.join(OUT_FIG, f"confmat_{mode}.pdf")
    plt.savefig(cm_path)
    plt.close()
    print(f"[FIG] {cm_path}")

    # ---- per-period × per-class F1
    per_rows = []
    # Align df_te index & lengths safely
    df_te_local = df_te.copy().reset_index(drop=True)
    y_te_series = pd.Series(y_te).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)

    for tp in TIME_PERIODS:
        mask = (df_te_local["time_period"] == tp)
        if not mask.any():
            per_rows.append((mode, tp, *([np.nan] * len(EVENTS))))
            continue

        f1s = []
        for cid, _ in enumerate(EVENTS):
            y_true_bin = (y_te_series[mask] == cid).astype(int)
            y_pred_bin = (y_pred_series[mask] == cid).astype(int)
            f1c = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            f1s.append(f1c)
        per_rows.append((mode, tp, *f1s))

    # ---- feature importances
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        out_fi = os.path.join(OUT_TAB, f"feature_importance_{mode}.tex")
        fi_df.to_latex(out_fi, index=False, float_format=lambda x: f"{x:.0f}")
        print(f"[TABLE] {out_fi}")

    # ---- SHAP summaries
    if HAS_SHAP:
        save_shap_summary(model, X_te, mode)

    # ---- per-GUID macro-F1 (on test split ONLY; lengths aligned)
    df_guid = df_te_local.copy()
    df_guid["y_true"] = y_te_series
    df_guid["y_pred"] = y_pred_series

    guid_scores = []
    for gid, sub in df_guid.groupby("GUID"):
        guid_f1 = f1_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0)
        guid_scores.append((mode, gid, guid_f1))

    guid_df = pd.DataFrame(guid_scores, columns=["mode", "GUID", "macro_f1"])
    guid_path = os.path.join(OUT_TAB, f"per_guid_{mode}.csv")
    guid_df.to_csv(guid_path, index=False)
    print(f"[TABLE] {guid_path}")

    return per_rows


# ---------------------------------------------------------------------
# MERGE PER-PERIOD × PER-CLASS F1 AND GUID TABLES
# ---------------------------------------------------------------------
def build_per_period_table(per_rows):
    if not per_rows:
        return
    cols = ["mode", "time_period"] + [f"class_{i}" for i in range(len(EVENTS))]
    df = pd.DataFrame(per_rows, columns=cols)
    out_pp = os.path.join(OUT_TAB, "per_period_per_class_f1.tex")
    df.to_latex(out_pp, index=False, na_rep="-", float_format=lambda x: f"{x:.4f}")
    print(f"[TABLE] {out_pp}")


def build_guid_tables():
    dfs = []
    for mode in ["season", "hierarchical", "no_season"]:
        fp = os.path.join(OUT_TAB, f"per_guid_{mode}.csv")
        if os.path.exists(fp):
            t = pd.read_csv(fp)
            t["mode"] = mode
            dfs.append(t)
    if not dfs:
        return

    df = pd.concat(dfs, ignore_index=True)

    # average GUID performance across modes
    avg = df.groupby("GUID", as_index=False)["macro_f1"].mean()

    top10 = avg.sort_values("macro_f1", ascending=False).head(10)
    bottom10 = avg.sort_values("macro_f1", ascending=True).head(10)

    top_path = os.path.join(OUT_TAB, "per_guid_top10.tex")
    bot_path = os.path.join(OUT_TAB, "per_guid_bottom10.tex")

    top10.to_latex(top_path, index=False, float_format=lambda x: f"{x:.4f}")
    bottom10.to_latex(bot_path, index=False, float_format=lambda x: f"{x:.4f}")

    print(f"[TABLE] {top_path}")
    print(f"[TABLE] {bot_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    df = load_master()
    season, no_season, hier = build_datasets(df)

    per_all = []
    per_all += train_and_export(season, "season")
    per_all += train_and_export(hier, "hierarchical")
    per_all += train_and_export(no_season, "no_season")

    build_per_period_table(per_all)
    build_guid_tables()
    print("\nExport complete. Check directories: reports/, figures/, tables/.\n")


if __name__ == "__main__":
    main()
