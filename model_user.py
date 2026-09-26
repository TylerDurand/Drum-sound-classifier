import streamlit as st
import torch
import torch.nn as nn
from torchvision.transforms import v2
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the model
def create_model():
    channels = 16
    ksize = 3
    K3 = (channels * 2) * 56 * 56  # Flattened size after second pooling layer

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=ksize),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(in_channels=channels, out_channels=channels * 2, kernel_size=ksize),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(K3, 128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )
    return model

# Classify the uploaded image
def classify_image(image, model, labels):
    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224)),  # Resize to match model input size
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pretrained models
    ])
    # Perform classification
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, axis=1).item()

    return labels[predicted_class]

def spectrogram_conversion(wav_file):

        # Load audio
        y, sr = librosa.load(wav_file, sr=22050)   # standard sampling rate

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


# Streamlit app
def main():
    st.title("Drum Sound Classifier")
    st.write("Upload an image to classify it.")

    model_path = "/Users/tylerdurand/Downloads/School Stuff/math3375/final_model1.pth"
    labels = ["Crash","Hat","Kick","Snare"]

    uploaded_file = st.file_uploader("Upload Spectrogram", type=["png","wav"])

    if uploaded_file is not None:

        spectrogram_conversion(uploaded_file)
        # Display the uploaded image
        image = Image.open("temp_spectrogram.png").convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Load the model
        model = create_model()
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

        # Classify the image
        predicted_label = classify_image(image, model, labels)
        st.write(f"Predicted Label: **{predicted_label}**")

if __name__ == "__main__":
    main()