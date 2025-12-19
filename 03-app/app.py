import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model 
from audio_utils import extract_mfcc, plot_spectrogram

# Load model & encoder
model = load_model("../02-model/lstm_genre_model.h5")

with open("../01-training/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.set_page_config(page_title="Music Genre Classification", layout="centered")

st.title("🎶 Music Genre Classification with LSTM")

uploaded_file = st.file_uploader(
    "Upload audio file (.wav / .mp3)",
    type=["wav", "mp3"]
)

if uploaded_file:

    #Audio player
    st.audio(uploaded_file)

    #Spectrogram
    st.subheader("🎼 Audio Spectrogram")
    uploaded_file.seek(0)
    fig = plot_spectrogram(uploaded_file)
    st.pyplot(fig)

    #Feature extraction
    uploaded_file.seek(0)
    mfcc = extract_mfcc(uploaded_file)
    if mfcc is None:
        st.error("❌ Gagal memproses audio. File mungkin rusak atau format tidak valid.")
        st.stop()

    mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])

    #Prediction
    prediction = model.predict(mfcc)[0]

    predicted_index = np.argmax(prediction)
    predicted_genre = encoder.inverse_transform([predicted_index])[0]

    confidence = prediction[predicted_index] * 100

    #Output
    st.subheader("🧠 Prediction Result")
    st.success(f"Genre: **{predicted_genre}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    #Confidence bar chart
    st.subheader("📊 Prediction Probabilities")
    for i, prob in enumerate(prediction):
        genre = encoder.inverse_transform([i])[0]
        st.progress(float(prob), text=f"{genre}: {prob*100:.1f}%")