import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "db_path": os.path.join(BASE_DIR, "data", "gestures.db"),
    "confidence_threshold": 0.5,
    "prediction_cooldown": 15,
    "camera_resolution": (1280, 720),
    "target_fps": 30,
    "train_fps": 20,
    "max_num_hands": 1,
    "min_detection_confidence": 0.5,
    "rf_estimators": 100,  # Para Random Forest
    "lstm_units": 64,      # Unidades por camada LSTM
    "lstm_layers": 2,      # Número de camadas LSTM
    "max_sequence_length": 15,  # Frames por sequência (dinâmicos)
    "min_samples_per_class": 50,  # Mínimo de samples/sequências por gesto
    "gesture_types": ["letter", "word", "movement"],
    "log_file": os.path.join(BASE_DIR, "logs", "Gesture_recognizer.log"),
    "train_data_dir": os.path.join(BASE_DIR, "data", "train")
}