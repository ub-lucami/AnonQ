
# evaluate_utility_extended_v2.py
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


# =========================================================
# CONFIG
# =========================================================
MASTER_CSV = "G:/TS_2025/anonymized_events.csv"
SEP = ";"
BOOTSTRAP_ITER = 200      # increase for publication (e.g., 1000)

TIME_PERIODS = ["night", "morning", "daytime", "afternoon"]


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
# BUILD DATASETS
# =========================================================
def build_datasets(df):
    df_season = df[df["week_number"].notna()].copy()

    df_no_season = df.copy()
    df_no_season = df_no_season.drop(columns=["week_number"], errors="ignore")

    df_hier = df.copy()
    df_hier["week_number_filled"] = df_hier["week_number"].fillna("unknown")
    df_hier["wk_missing_flag"] = df_hier["week_number"].isna().astype(int)

    return df_season, df_no_season, df_hier


# =========================================================
# FEATURE PREPARATION
# =========================================================
def prepare_xy(df, mode):
    df = df.copy()

    le_event = LabelEncoder()
    le_tp    = LabelEncoder()
    le_guid  = LabelEncoder()
    le_week  = LabelEncoder()

    # Encode fundamental categorical features
    df["time_period_enc"] = le_tp.fit_transform(df["time_period"])
    df["GUID_enc"]        = le_guid.fit_transform(df["GUID"])
    y = le_event.fit_transform(df["generalized_event"])

    # Feature selection
    if mode == "season":
        df["week_number_enc"] = le_week.fit_transform(df["week_number"].astype(str))
        X = df[["weekday", "time_period_enc", "week_number_enc", "GUID_enc"]].values

    elif mode == "no_season":
        X = df[["weekday", "time_period_enc", "GUID_enc"]].values

    elif mode == "hierarchical":
        df["week_number_enc"] = le_week.fit_transform(df["week_number_filled"].astype(str))
        X = df[["weekday", "time_period_enc", "week_number_enc",
                "wk_missing_flag", "GUID_enc"]].values

    else:
        raise ValueError("invalid mode")

    # Return full label list for stability
    full_label_ids = np.arange(len(le_event.classes_))

    return X, y, df, le_event, full_label_ids


# =========================================================
# TRAIN & EVALUATE
# =========================================================

def train_eval(df, mode, random_state=42):
    X, y, dfp, le_event, full_labels = prepare_xy(df, mode)

    # Train/test split
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, dfp,
        test_size=0.20,
        random_state=random_state,
        stratify=y
    )

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
    f1  = f1_score(y_te, y_pred, average="macro")

    top3 = top_k_accuracy_score(
        y_te, y_prob,
        k=3,
        labels=full_labels
    )

    # Full-class report, even for missing ones
    report = classification_report(
        y_te,
        y_pred,
        labels=full_labels,
        target_names=le_event.classes_,
        output_dict=True,
        zero_division=0
    )

    # Robust per-event F1
    per_event_f1 = {}
    for i in full_labels:
        cls_name = le_event.classes_[i]
        key = str(i)
        if key in report:
            per_event_f1[cls_name] = report[key]["f1-score"]
        else:
            per_event_f1[cls_name] = np.nan

    # Per-time-period F1 (night/morning/daytime/afternoon)
    period_scores = {}
    for period in ["night", "morning", "daytime", "afternoon"]:
        mask = df_te["time_period"] == period
        if mask.any():
            period_scores[period] = f1_score(
                y_te[mask],
                y_pred[mask],
                average="macro",
                zero_division=0
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
        "labels": full_labels,
    }


# =========================================================
# BOOTSTRAP CONFIDENCE INTERVALS (PATCHED)
# =========================================================
def bootstrap_ci(df, mode):
    results = []

    for _ in tqdm(range(BOOTSTRAP_ITER), desc=f"Bootstrap {mode}"):

        # Take bootstrap sample
        df_boot = df.sample(frac=1, replace=True)

        # skip degenerate samples (fewer than 2 classes)
        if df_boot["generalized_event"].nunique() < 2:
            continue

        try:
            res = train_eval(df_boot, mode)
            results.append(res["f1"])
        except:
            # Skip rare failed model fits
            continue

    return np.array(results)


# =========================================================
# PLOTS
# =========================================================
def plot_comparison(results):
    modes = ["season", "hierarchical", "no_season"]

    # Metrics chart
    plt.figure(figsize=(8, 5))
    df_m = pd.DataFrame({
        "mode": modes,
        "F1": [results[m]["f1"] for m in modes],
        "Accuracy": [results[m]["acc"] for m in modes],
        "Top3": [results[m]["top3"] for m in modes]
    })
    df_m.set_index("mode").plot(kind="bar")
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.savefig("comparison_metrics.png", dpi=200)
    plt.show()

    # Per-period F1
    df_p = pd.DataFrame({
        "season": results["season"]["period_f1"],
        "hierarchical": results["hierarchical"]["period_f1"],
        "no_season": results["no_season"]["period_f1"],
    }).T

    plt.figure(figsize=(8, 5))
    df_p.plot(kind="bar")
    plt.title("Per-Time-Period Macro F1")
    plt.tight_layout()
    plt.savefig("comparison_period.png", dpi=200)
    plt.show()


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_master()
    df_season, df_no_season, df_hier = build_datasets(df)

    print("Dataset sizes:")
    print(" Season:      ", len(df_season))
    print(" No_Season:   ", len(df_no_season))
    print(" Hierarchical:", len(df_hier))

    # Single evaluations
    res_season = train_eval(df_season, "season")
    res_hier   = train_eval(df_hier, "hierarchical")
    res_noseas = train_eval(df_no_season, "no_season")

    results = {
        "season": res_season,
        "hierarchical": res_hier,
        "no_season": res_noseas
    }

    for k, v in results.items():
        print("\n======================================")
        print(k.upper(), v)

    # ====================================
    # Bootstrap confidence intervals
    # ====================================
    ci_season = bootstrap_ci(df_season, "season")
    ci_hier   = bootstrap_ci(df_hier, "hierarchical")
    ci_noseas = bootstrap_ci(df_no_season, "no_season")

    print("\n=== 95% CI (macro F1) ===")
    for name, arr in [
        ("season", ci_season),
        ("hierarchical", ci_hier),
        ("no_season", ci_noseas)
    ]:
        print(f"{name:12s}: [{np.percentile(arr, 2.5):.4f}, "
              f"{np.percentile(arr, 97.5):.4f}]   mean={np.mean(arr):.4f}")

    # ====================================
    # Statistical Tests
    # ====================================
    print("\n=== Paired Tests (macro F1) ===")

    # Align lengths
    n = min(len(ci_season), len(ci_hier), len(ci_noseas))
    a = ci_season[:n]
    b = ci_hier[:n]
    c = ci_noseas[:n]

    # season vs hier
    t1, p1 = ttest_rel(a, b)
    w1, p1w = wilcoxon(a, b)

    # hier vs no_season
    t2, p2 = ttest_rel(b, c)
    w2, p2w = wilcoxon(b, c)

    print(f"Season vs Hierarchical:   t={t1:.3f}, p={p1:.4f} | Wilcoxon p={p1w:.4f}")
    print(f"Hierarchical vs NoSeason: t={t2:.3f}, p={p2:.4f} | Wilcoxon p={p2w:.4f}")

    # ====================================
    # PLOTS
    # ====================================
    plot_comparison(results)


if __name__ == "__main__":
    main()
