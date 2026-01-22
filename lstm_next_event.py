import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load data
df = pd.read_csv("anonymized_events.csv", sep=";")

# Encode events as integers
le_event = LabelEncoder()
df["event_id"] = le_event.fit_transform(df["generalized_event"])

# Encode GUIDs
le_guid = LabelEncoder()
df["guid_id"] = le_guid.fit_transform(df["GUID"])

# Sort by GUID + time order
df = df.sort_values(["guid_id", "week_number", "weekday", "time_period"])

# Group events by GUID
sequences = df.groupby("guid_id")["event_id"].apply(list)

# Filter out short sequences
sequences = [seq for seq in sequences if len(seq) >= 20]

# Build (X,y) pairs for next-event prediction
X = []
y = []

SEQ_LEN = 10

for seq in sequences:
    for i in range(len(seq) - SEQ_LEN):
        X.append(seq[i:i+SEQ_LEN])
        y.append(seq[i+SEQ_LEN])

X = np.array(X)
y = np.array(y)

# Convert labels to one-hot
num_events = len(le_event.classes_)
y_cat = to_categorical(y, num_events)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=num_events, output_dim=32, input_length=SEQ_LEN),
    LSTM(64),
    Dense(num_events, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print(model.summary())

# Train
model.fit(X, y_cat, epochs=15, batch_size=64, validation_split=0.2)

# Example inference
test_seq = X[0:1]
pred = model.predict(test_seq)
print("Predicted event:", le_event.inverse_transform([np.argmax(pred)]))