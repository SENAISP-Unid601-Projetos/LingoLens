import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

CONFIG = {
    "min_detection_confidence": 0.6,
    "min_tracking_confidence": 0.6,
    "static_confidence_boost": 0.05,
    "smoothing_factor": 0.65,
    "db_path": os.path.join(BASE_DIR, "data", "gestures.db"),
    "confidence_threshold_static": 0.78,
    "confidence_threshold_dynamic": 0.62,
    "prediction_cooldown": 50,
    "camera_resolution": (1280, 720),
    "target_fps": 30,
    "train_fps": 20,
    "max_num_hands": 1,
    "min_detection_confidence": 0.5,
    "rf_estimators": 100,
    "lstm_units": 64,
    "lstm_layers": 2,
    "sequence_length": 30,
    "min_samples_per_class": 120,
    "gesture_types": ["letter"],
    "log_file": os.path.join(BASE_DIR, "logs", "Gesture_recognizer.log"),
    "train_data_dir": os.path.join(BASE_DIR, "data", "train"),
    "use_lstm_for_dynamic": True,
    "dynamic_letters": ["H", "J", "K", "X", "Y", "Z"],  # DINÂMICAS"stable_threshold": 10,
    "motion_threshold": 0.004,
}

if CONFIG["train_fps"] > CONFIG["target_fps"]:
    raise ValueError(f"train_fps não pode ser maior que target_fps")

def validate_gesture_type(gesture_type):
    if gesture_type not in CONFIG["gesture_types"]:
        raise ValueError(f"Tipo inválido: {gesture_type}. Válidos: {CONFIG['gesture_types']}")