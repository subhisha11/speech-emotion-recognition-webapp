import os
import numpy as np
from .preprocess import extract_features, EMOTION_LABELS

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data/Audio_Song_Actors_01-24"

def load_data():
    X, y = [], []
    for actor in os.listdir(DATA_DIR):
        actor_path = os.path.join(DATA_DIR, actor)
        if not os.path.isdir(actor_path):
            continue
        for file in os.listdir(actor_path):
            file_path = os.path.join(actor_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                emotion_index = int(file.split("-")[2]) - 1  # adjust index according to file naming
                y.append(emotion_index)
    return np.array(X), to_categorical(y, num_classes=len(EMOTION_LABELS))

# Load data
X, y = load_data()
input_shape = X.shape[1]

# Build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(input_shape,)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(EMOTION_LABELS), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

# Save model
model.save("src/ser_model.h5")
print("🎉 Training complete. Model saved as ser_model.h5")
