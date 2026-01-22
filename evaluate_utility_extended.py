
# evaluate_utility_extended_v4.py
# Adds bootstrap distribution plots, CI bars, CSV exports, and effect sizes.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    top_k_accuracy_score,
    classification_report
)
from lightgbm import LGBMClassifier

from scipy.stats import ttest_rel, wilcoxon
from tqdm import tqdm
import os

# =========================================================
# CONFIG
# =========================================================
MASTER_CSV = "G:/TS_2025/anonymized_events.csv"
SEP = ";"
BOOTSTRAP_ITER = 200        # increase to 1000 for publication-quality CIs
TIME_PERIODS = ["night", "morning", "daytime", "afternoon"]
OUT_DIR = "eval_outputs"    # where CSVs and figures will be saved

os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# LOAD MASTER
# =========================================================
def load_master():
    df = pd.read_csv(MASTER_CSV, sep=SEP)

    df["weekday"] = df["weekday"].astype(int)
    df["time_period"] = df["time_period"].astype(str)
    df["generalized_event"] = df["generalized_event"].astype(str)
    df["GUID"] = df["GUID"].astype(str)

    if "week_number" in df.columns:
        df["week_number"] = df["week_number"].replace("", np.nan)

    return df

# =========================================================
# BUILD DATASETS (season / no_season / hierarchical)
# =========================================================
def build_datasets(df):
    # season dataset = only rows with valid week_number
    df_season = df[df["week_number"].notna()].copy()

    # no_season = drop week_number entirely
    df_no_season = df.copy()
    df_no_season = df_no_season.drop(columns=["week_number"], errors="ignore")

    # hierarchical = keep rows, fill missing week_number as "unknown"
    df_hier = df.copy()
    df_hier["week_number_filled"] = df_hier["week_number"].fillna("unknown")
    df_hier["wk_missing_flag"] = df_hier["week_number"].isna().astype(int)

    return df_season, df_no_season, df_hier

# =========================================================
# FEATURE PREPARATION (DataFrame-based → NO warnings)
# =========================================================
def prepare_xy(df, mode):
    df = df.copy()

    le_event = LabelEncoder()
    le_tp = LabelEncoder()
    le_guid = LabelEncoder()
    le_week = LabelEncoder()

    df["time_period_enc"] = le_tp.fit_transform(df["time_period"])
    df["GUID_enc"] = le_guid.fit_transform(df["GUID"])
    y = le_event.fit_transform(df["generalized_event"])

    if mode == "season":
        df["week_number_enc"] = le_week.fit_transform(df["week_number"].astype(str))
        X = df[["weekday", "time_period_enc", "week_number_enc", "GUID_enc"]]

    elif mode == "no_season":
        X = df[["weekday", "time_period_enc", "GUID_enc"]]

    elif mode == "hierarchical":
        df["week_number_enc"] = le_week.fit_transform(df["week_number_filled"].astype(str))
        X = df[["weekday", "time_period_enc", "week_number_enc", "wk_missing_flag", "GUID_enc"]]

    else:
        raise ValueError("invalid mode")

    full_labels = np.arange(len(le_event.classes_))
    return X, y, df, le_event, full_labels

# =========================================================
# TRAIN & EVALUATE
# =========================================================
def train_eval(df, mode, random_state=42):
    X, y, dfp, le_event, full_labels = prepare_xy(df, mode)

    # Stratified split
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, dfp, test_size=0.20, random_state=random_state, stratify=y
    )

    # Ensure DataFrame shape is preserved after split
    X_tr = pd.DataFrame(X_tr, columns=X.columns)
    X_te = pd.DataFrame(X_te, columns=X.columns)

    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)

    # Main metrics
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")
    top3 = top_k_accuracy_score(y_te, y_prob, k=3, labels=full_labels)

    # Stable classification report
    report = classification_report(
        y_te, y_pred,
        labels=full_labels,
        target_names=le_event.classes_,
        output_dict=True,
        zero_division=0
    )

    # Safe per-class F1 (no KeyErrors)
    per_event_f1 = {}
    for i in full_labels:
        cls_name = le_event.classes_[i]
        key = str(i)
        per_event_f1[cls_name] = report[key]["f1-score"] if key in report else np.nan

    # Per-time-period F1
    period_scores = {}
    for period in TIME_PERIODS:
        mask = (df_te["time_period"] == period)
        if mask.any():
            period_scores[period] = f1_score(
                y_te[mask], y_pred[mask], average="macro", zero_division=0
            )
        else:
            period_scores[period] = np.nan

    return {
        "mode": mode,
        "n": len(df),
        "acc": acc,
        "f1": f1,
        "top3": top3,
        "per_event_f1": per_event_f1,
        "period_f1": period_scores,
        "labels": full_labels
    }

# =========================================================
# BOOTSTRAP CONFIDENCE INTERVALS (robust)
# =========================================================
def bootstrap_ci(df, mode):
    scores = []
    for _ in tqdm(range(BOOTSTRAP_ITER), desc=f"Bootstrap {mode}"):
        df_boot = df.sample(frac=1, replace=True)
        # skip degenerate samples
        if df_boot["generalized_event"].nunique() < 2:
            continue
        try:
            res = train_eval(df_boot, mode)
            scores.append(res["f1"])
        except Exception:
            continue
    return np.array(scores)

# =========================================================
# EFFECT SIZES
# =========================================================
def cohen_d_paired(a, b):
    """Cohen's d for paired samples (Dz: mean of differences / SD of differences)."""
    diff = a - b
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

def cliffs_delta(a, b):
    """Cliff's delta for two samples (unpaired effect size)."""
    a = np.asarray(a)
    b = np.asarray(b)
    # O(n^2) but fine for n~200
    diff = a[:, None] - b[None, :]
    n_pos = (diff > 0).sum()
    n_neg = (diff < 0).sum()
    return (n_pos - n_neg) / (a.size * b.size)

# =========================================================
# CSV EXPORT HELPERS
# =========================================================
def export_csvs(results, ci_stats):
    # 1) Summary metrics (single evaluation)
    df_sum = pd.DataFrame([
        {"mode": k, "n": v["n"], "accuracy": v["acc"], "f1": v["f1"], "top3": v["top3"]}
        for k, v in results.items()
    ])
    df_sum.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False, sep=";")

    # 2) Per-period F1
    df_period = pd.DataFrame(results["season"]["period_f1"], index=["season"])
    df_period = pd.concat([df_period, pd.DataFrame(results["hierarchical"]["period_f1"], index=["hierarchical"])])
    df_period = pd.concat([df_period, pd.DataFrame(results["no_season"]["period_f1"], index=["no_season"])])
    df_period.to_csv(os.path.join(OUT_DIR, "per_period_f1.csv"), sep=";")

    # 3) Per-event F1
    # Union of all event names across modes
    all_events = sorted(set().union(*[set(v["per_event_f1"].keys()) for v in results.values()]))
    rows = []
    for mode, res in results.items():
        row = {"mode": mode}
        row.update({ev: res["per_event_f1"].get(ev, np.nan) for ev in all_events})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "per_event_f1.csv"), index=False, sep=";")

    # 4) Bootstrap summary stats
    pd.DataFrame(ci_stats).to_csv(os.path.join(OUT_DIR, "bootstrap_summary.csv"), index=False, sep=";")

# =========================================================
# PLOTS
# =========================================================
def plot_comparison(results):
    modes = ["season", "hierarchical", "no_season"]

    # Main metrics plot (single-run point estimates)
    df_m = pd.DataFrame({
        "mode": modes,
        "F1": [results[m]["f1"] for m in modes],
        "Accuracy": [results[m]["acc"] for m in modes],
        "Top3": [results[m]["top3"] for m in modes]
    }).set_index("mode")

    ax = df_m.plot(kind="bar", figsize=(8, 5))
    plt.title("Model Performance Comparison (Point Estimates)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_metrics_v4.png"), dpi=200)
    plt.show()

    # Per-period plot
    df_p = pd.DataFrame({
        "season": results["season"]["period_f1"],
        "hierarchical": results["hierarchical"]["period_f1"],
        "no_season": results["no_season"]["period_f1"]
    }).T

    ax = df_p.plot(kind="bar", figsize=(9, 5))
    plt.title("Per-Time-Period Macro-F1 (Point Estimates)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_period_v4.png"), dpi=200)
    plt.show()

def plot_bootstrap_distributions(ci_season, ci_hier, ci_noseas):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(ci_season, label="Season", fill=True)
    sns.kdeplot(ci_hier, label="Hierarchical", fill=True)
    sns.kdeplot(ci_noseas, label="No Season", fill=True)
    plt.title("Bootstrap Distributions of Macro F1")
    plt.xlabel("Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bootstrap_distributions_v4.png"), dpi=200)
    plt.show()

def plot_f1_with_ci(ci_dict):
    """
    ci_dict: {mode: {"mean":..., "low":..., "high":...}}
    Draw bar chart of bootstrap means with 95% CI error bars.
    """
    modes = list(ci_dict.keys())
    means = [ci_dict[m]["mean"] for m in modes]
    lows  = [ci_dict[m]["low"]  for m in modes]
    highs = [ci_dict[m]["high"] for m in modes]
    yerr  = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]

    plt.figure(figsize=(8, 5))
    plt.bar(modes, means, yerr=yerr, capsize=6, color=["#66c2a5", "#8da0cb", "#fc8d62"])
    plt.ylabel("Macro F1")
    plt.title("Bootstrap Means with 95% CI")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "bootstrap_means_ci_v4.png"), dpi=200)
    plt.show()

# =========================================================
# MAIN
# =========================================================
def main():
    df = load_master()
    df_season, df_no_season, df_hier = build_datasets(df)

    print("Dataset sizes:")
    print(" Season:      ", len(df_season))
    print(" Hierarchical:", len(df_hier))
    print(" No_Season:   ", len(df_no_season))

    # Single evaluations (point estimates)
    res_season = train_eval(df_season, "season")
    res_hier   = train_eval(df_hier, "hierarchical")
    res_noseas = train_eval(df_no_season, "no_season")

    results = {"season": res_season, "hierarchical": res_hier, "no_season": res_noseas}

    # Print point estimates
    for k, v in results.items():
        print("\n====================================")
        print(k.upper(), v)

    # Bootstrap CIs
    ci_season = bootstrap_ci(df_season, "season")
    ci_hier   = bootstrap_ci(df_hier, "hierarchical")
    ci_noseas = bootstrap_ci(df_no_season, "no_season")

    def ci_stats(arr, name):
        return {
            "mode": name,
            "n_boot": len(arr),
            "mean": np.mean(arr),
            "std": np.std(arr, ddof=1),
            "low": np.percentile(arr, 2.5),
            "high": np.percentile(arr, 97.5)
        }

    ci_summary = {
        "season": ci_stats(ci_season, "season"),
        "hierarchical": ci_stats(ci_hier, "hierarchical"),
        "no_season": ci_stats(ci_noseas, "no_season")
    }

    print("\n=== 95% CI (macro F1) ===")
    for name in ["season", "hierarchical", "no_season"]:
        s = ci_summary[name]
        print(f"{name:12s}: [{s['low']:.4f}, {s['high']:.4f}]   mean={s['mean']:.4f}")

    # Statistical tests (paired, length-aligned)
    n = min(len(ci_season), len(ci_hier), len(ci_noseas))
    a = ci_season[:n]
    b = ci_hier[:n]
    c = ci_noseas[:n]

    print("\n=== Paired Tests (macro F1) ===")
    t1, p1 = ttest_rel(a, b)
    w1, pw1 = wilcoxon(a, b)
    print(f"Season vs Hierarchical:   t={t1:.3f}, p={p1:.4f} | Wilcoxon p={pw1:.4f}")

    t2, p2 = ttest_rel(b, c)
    w2, pw2 = wilcoxon(b, c)
    print(f"Hierarchical vs NoSeason: t={t2:.3f}, p={p2:.4f} | Wilcoxon p={pw2:.4f}")

    # Effect sizes
    # Cohen's d for paired samples (Dz) — use aligned vectors
    dz_se_hier = cohen_d_paired(a, b)
    dz_hier_ns = cohen_d_paired(b, c)

    # Cliff's delta (unpaired) — use full arrays (not aligned needed)
    cd_se_hier = cliffs_delta(ci_season, ci_hier)
    cd_hier_ns = cliffs_delta(ci_hier, ci_noseas)

    print("\n=== Effect Sizes ===")
    print(f"Cohen's d (paired) Season vs Hierarchical:   d_z={dz_se_hier:.3f}")
    print(f"Cohen's d (paired) Hierarchical vs NoSeason: d_z={dz_hier_ns:.3f}")
    print(f"Cliff's delta Season vs Hierarchical:        δ={cd_se_hier:.3f}")
    print(f"Cliff's delta Hierarchical vs NoSeason:      δ={cd_hier_ns:.3f}")

    # CSV exports
    export_csvs(results, list(ci_summary.values()))

    # Plots
    plot_comparison(results)
    plot_bootstrap_distributions(ci_season, ci_hier, ci_noseas)
    plot_f1_with_ci(ci_summary)

if __name__ == "__main__":
    main()
