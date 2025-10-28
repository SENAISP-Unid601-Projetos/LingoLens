import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(DATA_DIR, "gestures.db")

CONFIG = {
    "log_file": os.path.join(LOG_DIR, "app.log"),
    "db_path": DB_PATH,
    "camera_resolution": (640, 480),
    "target_fps": 30,
    "train_fps": 15,
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_samples_per_class": 10,
    "movement_threshold": 0.0005,
    "prediction_cooldown": 15,
    "confidence_threshold": 0.75
}

STATIC_MODEL = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
    "n_jobs": 1
}

DYNAMIC_MODEL = {
    "sequence_length": 15,
    "lstm_units": 32,
    "dense_units": 64,
    "dropout": 0.4,
    "epochs": 20,
    "batch_size": 8,
    "validation_split": 0.25
}

def validate_gesture_type(gesture_type):
    valid_types = ["letter", "number", "word"]
    if gesture_type.lower() not in valid_types:
        raise ValueError(f"Tipo de gesto inválido: {gesture_type}. Use: {valid_types}")