import numpy as np
from tensorflow.keras.models import load_model
from .preprocess import extract_features, EMOTION_LABELS


# Load trained model
model = load_model("src/ser_model.h5")

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error", 0.0
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features, verbose=0)
    label_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    return EMOTION_LABELS[label_index], confidence

if __name__ == "__main__":
    # Example usage
    file = r"data/Audio_Song_Actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"
    label, conf = predict_emotion(file)
    print(f"Predicted Emotion: {label}, Confidence: {conf:.2f}")
