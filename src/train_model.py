import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from preprocess import extract_features, EMOTION_LABELS

DATASET_PATH = "C:\\SER\\data\\Audio_Song_Actors_01-24"

def load_data(dataset_path=DATASET_PATH):
    X, y = [], []
    for actor in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor)
        if not os.path.isdir(actor_path):
            continue
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                file_path = os.path.join(actor_path, file)
                features = extract_features(file_path)
                X.append(features)
                emotion_code = file.split("-")[2]  # e.g., "03" from "03-01-01-01-01-01-01.wav"
                y.append(EMOTION_LABELS.get(emotion_code))
    return np.array(X), np.array(y)

# Load data
X, y = load_data()
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(256, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("C:\\SER\\src\\ser_model.h5")
print("🎉 Training complete. Model saved as ser_model.h5")
