import numpy as np
import librosa

EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features(file_path, sr=22050):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
        mel_features = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        return mel_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
