# src/predict.py

import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Emotion labels (update according to your model)
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ser_model.h5')
model = load_model(MODEL_PATH)

def extract_features(file_path, sr=22050, n_mfcc=40):
    """
    Extract MFCC features from an audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_emotion(file_path):
    """
    Predict emotion from an audio file.
    Returns predicted label and confidence.
    """
    features = extract_features(file_path)
    if features is None:
        return "Error", 0.0

    # Model expects 2D array: (1, num_features)
    features = np.expand_dims(features, axis=0)
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    label = EMOTION_LABELS[predicted_index]
    return label, confidence

# Example usage (for local testing)
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), 'sample_audio.wav')
    label, conf = predict_emotion(file_path)
    print(f"Predicted Emotion: {label}, Confidence: {conf:.2f}")
