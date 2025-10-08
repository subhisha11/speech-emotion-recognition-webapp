# app.py

import streamlit as st
import os
from src.predict import predict_emotion

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎤", layout="centered")
st.title("🎤 Speech Emotion Recognition Web App")

st.write("""
Upload an audio file (WAV format) and the app will predict the emotion of the speaker.
""")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("temp_audio.wav")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_file_path, format="audio/wav")
    
    # Predict emotion
    with st.spinner("Analyzing audio..."):
        label, confidence = predict_emotion(temp_file_path)

    # Show result
    st.success(f"Predicted Emotion: **{label}**")
    st.info(f"Confidence: **{confidence*100:.2f}%**")

    # Optional: Remove temp file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
