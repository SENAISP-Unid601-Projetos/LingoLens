import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Criar diretórios necessários
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

CONFIG = {
    "db_path": os.path.join(BASE_DIR, "data", "gestures.db"),
    "confidence_threshold": 0.7,  # Limiar para predições confiáveis
    "prediction_cooldown": 45,  # 1.5 segundos a 30 FPS
    "camera_resolution": (1280, 720),
    "target_fps": 30,
    "train_fps": 20,
    "max_num_hands": 1,
    "min_detection_confidence": 0.5,
    "rf_estimators": 100,
    "lstm_units": 64,
    "lstm_layers": 2,
    "max_sequence_length": 15,
    "min_samples_per_class": 150,  # Aumentado para mais dados por letra
    "gesture_types": ["letter", "word", "movement"],
    "log_file": os.path.join(BASE_DIR, "logs", "Gesture_recognizer.log"),
    "train_data_dir": os.path.join(BASE_DIR, "data", "train")
}

if CONFIG["train_fps"] > CONFIG["target_fps"]:
    raise ValueError(f"train_fps ({CONFIG['train_fps']}) não pode ser maior que target_fps ({CONFIG['target_fps']})")

def validate_gesture_type(gesture_type):
    if gesture_type not in CONFIG["gesture_types"]:
        raise ValueError(f"Tipo de gesto inválido: {gesture_type}. Tipos válidos: {CONFIG['gesture_types']}")