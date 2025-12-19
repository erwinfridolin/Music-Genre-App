import librosa
import librosa.display
import numpy as np
import io
import soundfile as sf
import matplotlib.pyplot as plt

def extract_mfcc(uploaded_file, n_mfcc=40, max_len=174):
    try:
        # Baca bytes dari Streamlit UploadedFile
        audio_bytes = uploaded_file.read()

        # Convert ke buffer
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio
        y, sr = librosa.load(audio_buffer, sr=None, duration=30)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T

        # Padding / Truncating
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len]

        return mfcc

    except Exception as e:
        print("Audio processing error:", e)
        return None


def plot_spectrogram(uploaded_file):
    try:
        audio_bytes = uploaded_file.read()
        audio_buffer = io.BytesIO(audio_bytes)

        y, sr = librosa.load(audio_buffer, sr=None, duration=30)

        fig, ax = plt.subplots(figsize=(8, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="log",
            ax=ax
        )

        ax.set_title("Spectrogram")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

        return fig

    except Exception as e:
        print("Spectrogram error:", e)
        return None