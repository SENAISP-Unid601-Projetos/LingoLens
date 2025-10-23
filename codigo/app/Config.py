import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "db_path": os.path.join(BASE_DIR, "data", "gestures.db"),
    "confidence_threshold": 0.7,
    "prediction_cooldown": 45,
    "camera_resolution": (800, 600),
    "target_fps": 30,
    "max_num_hands": 1,
    "min_detection_confidence": 0.5,
    "knn_neighbors": 5,
    "log_file": os.path.join(BASE_DIR, "Logs", "Gesture_recognizer.log"),
    
    "train_fps": 30,
    "gesture_types": ["letter", "word", "movement"],
    "max_sequence_length": 30,
    "train_data_dir": os.path.join(BASE_DIR, "data", "train"),
}