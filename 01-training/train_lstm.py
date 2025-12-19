import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load processed data
X = np.load("../04-data/processed/X.npy")
y = np.load("../04-data/processed/y.npy")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Build model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(174, 40)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(len(set(y)), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(
    X,
    y_categorical,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# Save model
model.save("../02-model/lstm_genre_model.h5")

# Save encoder
with open("../01-training/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Model training completed and saved.")