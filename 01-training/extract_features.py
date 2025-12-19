import os
import numpy as np
import librosa

DATASET_PATH = "../04-data/raw"
OUTPUT_PATH = "../04-data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)

X = []
y = []

def extract_mfcc(file_path, max_len=174):
    y_audio, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=40)
    mfcc = mfcc.T

    if mfcc.shape[0] < max_len:
        mfcc = np.pad(
            mfcc,
            ((0, max_len - mfcc.shape[0]), (0, 0)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:max_len]

    return mfcc


for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    if not os.path.isdir(genre_path):
        continue

    for file in os.listdir(genre_path):
        if file.endswith(".wav"):
            file_path = os.path.join(genre_path, file)
            mfcc = extract_mfcc(file_path)
            X.append(mfcc)
            y.append(genre)

X = np.array(X)
y = np.array(y)

np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)

print("Feature extraction completed.")
print("X shape:", X.shape)
print("y shape:", y.shape)