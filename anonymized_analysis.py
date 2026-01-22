import argparse
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ---------------------------------------------
# CONFIG
# ---------------------------------------------
INPUT_FILE = "G:/TS_2025/anonymized_events.csv"   # prilagodi ime datoteke
SEP = ";"  # delimiter
SAVE_FIGS = True  # shrani grafe v PNG
OUTPUT_DIR = "figures"  # kam shraniti grafe

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Analyze anonymized events CSV and create plots.")
    p.add_argument("-i", "--input", help="Path to anonymized events CSV", default=INPUT_FILE)
    p.add_argument("--sample", help="Generate a small synthetic sample dataset if input missing", action="store_true")
    return p.parse_args()


args = parse_args()
INPUT_FILE = args.input

if not os.path.exists(INPUT_FILE):
    msg = f"Input file not found: {INPUT_FILE}"
    if args.sample:
        warnings.warn(msg + ". Generating small synthetic sample because --sample was passed.")
        # create a tiny synthetic dataframe with expected columns
        df = pd.DataFrame({
            "GUID": ["G1", "G2", "G1", "G3"],
            "week_number": [1, 1, 2, 2],
            "weekday": [0, 1, 0, 6],
            "time_period": ["morning", "daytime", "night", "afternoon"],
            "generalized_event": ["measurement_event", "power_event", "measurement_event", "system_event"]
        })
    else:
        print(msg)
        sys.exit(1)
else:
    df = pd.read_csv(INPUT_FILE, sep=SEP)

# Pretvori categorical stolpce zaradi lepših grafov
# Use nullable integer dtype to allow NA values instead of failing
if "week_number" in df.columns:
    df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")
else:
    df["week_number"] = pd.Series(dtype="Int64")

if "weekday" in df.columns:
    df["weekday"] = pd.to_numeric(df["weekday"], errors="coerce").astype("Int64")
else:
    df["weekday"] = pd.Series(dtype="Int64")

if "time_period" in df.columns:
    df["time_period"] = df["time_period"].astype("category")
else:
    df["time_period"] = pd.Series(dtype="category")

if "generalized_event" in df.columns:
    df["generalized_event"] = df["generalized_event"].astype("category")
else:
    df["generalized_event"] = pd.Series(dtype="category")

print("\n### BASIC SUMMARY ###")
print(df.head())
print("\nShape:", df.shape)
print("\nUnique GUIDs:", df["GUID"].nunique() if "GUID" in df.columns else "N/A")
print("Unique events:", df["generalized_event"].nunique() if "generalized_event" in df.columns else "N/A")

# ---------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    if SAVE_FIGS:
        plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

# Create plot-safe columns to avoid pandas/seaborn grouping FutureWarnings
def _to_plot_str(s):
    return s.apply(lambda x: str(int(x)) if pd.notna(x) and str(x) != '<NA>' else 'NA')

if "week_number" in df.columns:
    df["_week_number_plot"] = _to_plot_str(df["week_number"])
else:
    df["_week_number_plot"] = pd.Series([], dtype=str)

if "weekday" in df.columns:
    df["_weekday_plot"] = _to_plot_str(df["weekday"])
else:
    df["_weekday_plot"] = pd.Series([], dtype=str)

# ---------------------------------------------
# DISTRIBUTION: events per week
# ---------------------------------------------
# Aggregate counts by week and event, then plot grouped bars to avoid seaborn grouping internals
plt.figure(figsize=(10,5))
week_counts = df.groupby(["_week_number_plot", "generalized_event"], observed=False).size().unstack(fill_value=0)
week_counts.plot(kind="bar", figsize=(10,5))
plt.title("Events per ISO Week Number")
plt.xticks(rotation=45)
savefig("events_per_week")

# ---------------------------------------------
# DISTRIBUTION: events per weekday
# ---------------------------------------------
plt.figure(figsize=(8,5))
weekday_counts = df["_weekday_plot"].value_counts().sort_index()
weekday_counts.plot(kind="bar", figsize=(8,5))
plt.title("Events per Weekday (0=Mon ... 6=Sun)")
savefig("events_per_weekday")

# ---------------------------------------------
# DISTRIBUTION: events per time period
# ---------------------------------------------
plt.figure(figsize=(8,5))
time_order = ["night","morning","daytime","afternoon"]
time_counts = df["time_period"].value_counts().reindex(time_order).fillna(0)
time_counts.plot(kind="bar", figsize=(8,5))
plt.title("Events per Time Period")
savefig("events_per_time_period")

# ---------------------------------------------
# HEATMAP: week_number × time_period
# ---------------------------------------------
pivot = df.pivot_table(index="week_number",
                       columns="time_period",
                       values="GUID",
                       aggfunc="count",
                       observed=False).fillna(0)

plt.figure(figsize=(8,6))
sns.heatmap(pivot, cmap="Blues", annot=False)
plt.title("Heatmap: Week Number × Time Period")
savefig("heatmap_week_timeperiod")

# ---------------------------------------------
# HEATMAP: weekday × generalized_event
# ---------------------------------------------
pivot2 = df.pivot_table(index="weekday",
                        columns="generalized_event",
                        values="GUID",
                        aggfunc="count",
                        observed=False).fillna(0)

plt.figure(figsize=(10,6))
sns.heatmap(pivot2, cmap="Reds", annot=False)
plt.title("Heatmap: Weekday × Generalized Event")
savefig("heatmap_weekday_event")

# ---------------------------------------------
# TOP EVENTS
# ---------------------------------------------
top_events = df["generalized_event"].value_counts()
print("\n### TOP EVENTS ###")
print(top_events)

top_events.plot(kind="bar", figsize=(8,5))
plt.title("Event frequency")
savefig("top_events")

# ---------------------------------------------
# OPTIONAL: analyze single GUID
# ---------------------------------------------
def analyse_guid(guid):
    subset = df[df["GUID"] == guid]
    if subset.empty:
        print(f"GUID {guid} not found.")
        return

    print(f"\n### ANALYSIS FOR GUID: {guid} ###")
    print("Events:", subset.shape[0])
    print(subset["generalized_event"].value_counts())

    plt.figure(figsize=(8,5))
    tp_counts = subset["time_period"].value_counts()
    tp_counts.plot(kind="bar", figsize=(8,5))
    plt.title(f"Time Period Distribution for GUID {guid}")
    savefig(f"guid_{guid}_time_period")

# Example:
# analyse_guid("some-guid-here")
