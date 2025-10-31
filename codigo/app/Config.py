import os
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(DATA_DIR, "gestures.db")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

logger = logging.getLogger("GestureApp")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def get_logger(name):
    return logging.getLogger(f"GestureApp.{name}")

CONFIG = {
    "log_file": LOG_FILE,
    "db_path": DB_PATH,
    "camera_resolution": (1080, 720),
    "target_fps": 30,
    "train_fps": 15,
    "max_num_hands": 1,
    "min_detection_confidence": 0.7,
    "min_samples_per_class": 10,
    "max_samples_per_gesture": 100,
    "movement_threshold": 0.0005,
    "prediction_cooldown": 1.5,
    "confidence_threshold": 0.85,
    "min_frames": 20,
    "model_dir": MODEL_DIR
}

def validate_gesture_type(gesture_type):
    valid = ["letter", "number", "word"]
    if gesture_type.lower() not in valid:
        raise ValueError(f"Tipo invalido: {gesture_type}. Use: {valid}")