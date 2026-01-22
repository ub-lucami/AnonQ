
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical


# ------------------------------------------------------
# SETTINGS
# ------------------------------------------------------
st.set_page_config(page_title="Event Analysis & Prediction Dashboard", layout="wide")
st.title("📊 Event Analysis & Prediction Dashboard (Local, Secure)")


# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, sep=";")
    return df

uploaded_file = st.file_uploader("Naloži anonymized_events.csv datoteko", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📁 Pregled podatkov")
    st.write(df.head())

    # ------------------------------------------------------
    # ANALYSIS SECTION
    # ------------------------------------------------------
    st.header("🔍 Analiza podatkov")

    # 1) Distribution of GUIDs
    counts = df["GUID"].value_counts()

    st.subheader("Porazdelitev GUID-ov")
    st.write(counts.describe())
    st.bar_chart(counts)

    # 2) Distributions
    st.subheader("Distribucije po časovnih značilnostih")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Po tednih")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x="week_number", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.write("### Po dnevih v tednu")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x="weekday", ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)
    with col3:
        st.write("### Po time_period")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x="time_period", ax=ax)
        st.pyplot(fig)

    with col4:
        st.write("### Generalized event frekvence")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x="generalized_event", ax=ax)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # ------------------------------------------------------
    # HEATMAPS
    # ------------------------------------------------------
    st.header("🔥 Heatmape")

    pivot1 = df.pivot_table(index="week_number", columns="time_period", values="GUID", aggfunc="count")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.heatmap(pivot1, cmap="Blues", ax=ax)
    st.write("### Week × Time Period")
    st.pyplot(fig)

    pivot2 = df.pivot_table(index="weekday", columns="generalized_event", values="GUID", aggfunc="count")
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot2, cmap="Reds", ax=ax)
    st.write("### Weekday × Event")
    st.pyplot(fig)


    # ------------------------------------------------------
    # LIGHTGBM MODEL
    # ------------------------------------------------------
    st.header("🤖 LightGBM napovedovanje dogodkov")

    # Encode categorical features
    le_guid = LabelEncoder()
    le_tp = LabelEncoder()
    le_event = LabelEncoder()

    df["GUID_enc"] = le_guid.fit_transform(df["GUID"])
    df["time_period_enc"] = le_tp.fit_transform(df["time_period"])
    df["event_enc"] = le_event.fit_transform(df["generalized_event"])

    # Cyclical features
    df["week_sin"] = np.sin(2 * np.pi * df["week_number"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_number"] / 52)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    X = df[["GUID_enc","week_sin","week_cos","weekday_sin","weekday_cos","time_period_enc"]]
    y = df["event_enc"]

    if st.button("Treniraj LightGBM model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            subsample=0.9
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        st.subheader("📈 Rezultati")
        st.text(classification_report(y_test, y_pred, target_names=le_event.classes_))

        st.write("### Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)


    # ------------------------------------------------------
    # LSTM MODEL
    # ------------------------------------------------------
    st.header("🔮 LSTM napoved naslednjega dogodka")

    if st.button("Treniraj LSTM model"):
        df_sorted = df.sort_values(["GUID_enc", "week_number", "weekday", "time_period_enc"])

        sequences = df_sorted.groupby("GUID_enc")["event_enc"].apply(list)
        sequences = [seq for seq in sequences if len(seq) >= 20]

        SEQ_LEN = 10
        X_seq, y_seq = [], []

        for seq in sequences:
            for i in range(len(seq) - SEQ_LEN):
                X_seq.append(seq[i:i+SEQ_LEN])
                y_seq.append(seq[i+SEQ_LEN])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        y_seq_cat = to_categorical(y_seq, len(le_event.classes_))

        lstm_model = Sequential([
            Embedding(input_dim=len(le_event.classes_), output_dim=32, input_length=SEQ_LEN),
            LSTM(64),
            Dense(len(le_event.classes_), activation="softmax")
        ])

        lstm_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        st.write("Treniranje… lahko traja nekaj minut.")
        history = lstm_model.fit(X_seq, y_seq_cat, epochs=10, batch_size=64, validation_split=0.2)

        st.subheader("📈 LSTM natančnost")
        st.write(f"Train accuracy: {history.history['accuracy'][-1]:.3f}")
        st.write(f"Val accuracy: {history.history['val_accuracy'][-1]:.3f}")


    # ------------------------------------------------------
    # Prediction demo
    # ------------------------------------------------------
    st.header("🎯 Napoved za izbran GUID")

    guid_list = df["GUID"].unique().tolist()
    selected_guid = st.selectbox("Izberi GUID", guid_list)

    if st.button("Napovej naslednji dogodek"):
        events = df[df["GUID"] == selected_guid]["event_enc"].tolist()

        if len(events) < 10:
            st.error("Premalo dogodkov za napoved (min 10).")
        else:
            seq = np.array(events[-10:]).reshape(1, -1)
            pred = lstm_model.predict(seq)
            ev = le_event.inverse_transform([np.argmax(pred)])

            st.success(f"Predviden naslednji dogodek: **{ev[0]}**")
