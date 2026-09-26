import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import joblib
from sklearn.svm import SVC
from PIL import Image

# Convert audio to spectrogram
def spectrogram_conversion(wav_file):
    # Load audio
    y, sr = librosa.load(wav_file, sr=22050)  # standard sampling rate

    # Compute Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,
                                         hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Save as .png
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_db, sr=sr, hop_length=512, x_axis=None, y_axis=None)
    plt.axis("off")

    out_path = "/Users/tylerdurand/Downloads/School Stuff/math3375/temp_spectrogram.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return mel_db.flatten()  # Flatten the spectrogram for SVC input

# Streamlit app
def main():
    st.title("Drum Sound Classifier")
    st.write("Upload an audio file to classify it.")

    model_path = "/Users/tylerdurand/Downloads/School Stuff/math3375/svc_model_v1.joblib"
    labels = ["Crash", "Hat", "Kick", "Snare"]

    # Check if the model exists, otherwise train and save it
    if not os.path.exists(model_path):
        st.error("SVC model not found. Please train and save the model first.")
        return

    # Load the SVC model
    svc_model = joblib.load(model_path)

    uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

    if uploaded_file is not None:
        # Convert audio to spectrogram and preprocess for SVC
        features = spectrogram_conversion(uploaded_file)

        # Reshape features for SVC input
        features = features.reshape(1, -1)

        # Perform classification
        predicted_label = labels[svc_model.predict(features)[0]]
        st.write(f"Predicted Label: **{predicted_label}**")

if __name__ == "__main__":
    main()
