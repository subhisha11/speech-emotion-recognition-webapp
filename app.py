import streamlit as st
from src.predict import predict_emotion

st.title("Speech Emotion Recognition 🎤")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    label, conf = predict_emotion("temp.wav")
    st.success(f"Predicted Emotion: {label}, Confidence: {conf:.2f}")
