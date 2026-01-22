import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("G:/TS_2025/anonymized_events.csv", sep=";")

# ---------------------------------------------
# Feature engineering
# ---------------------------------------------
df["week_sin"] = np.sin(2 * np.pi * df["week_number"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["week_number"] / 52)

df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

# Encode categorical
le_guid = LabelEncoder()
le_tp = LabelEncoder()
le_event = LabelEncoder()

df["GUID_enc"] = le_guid.fit_transform(df["GUID"])
df["time_period_enc"] = le_tp.fit_transform(df["time_period"])
df["event_enc"] = le_event.fit_transform(df["generalized_event"])

# Features & target
X = df[[
    "GUID_enc",
    "week_sin","week_cos",
    "weekday_sin","weekday_cos",
    "time_period_enc"
]]

y = df["event_enc"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Model
# -----------------------------
model = LGBMClassifier(
    n_estimators=500,
    max_depth=-1,
    learning_rate=0.03,
    subsample=0.9
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le_event.classes_))
print(confusion_matrix(y_test, y_pred))

# Example prediction:
sample = X_test.iloc[0:1]
pred_event = le_event.inverse_transform(model.predict(sample))
print("Predicted event:", pred_event)
