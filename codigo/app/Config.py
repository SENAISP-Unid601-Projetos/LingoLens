import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Criar diretórios necessários
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

CONFIG = {
    "db_path": os.path.join(BASE_DIR, "data", "gestures.db"),
    "data_dir": os.path.join(BASE_DIR, "data"),  # Diretório para modelos e dados
    "log_file": os.path.join(BASE_DIR, "logs", "Gesture_recognizer.log"),  # Log geral
    "dynamic_log_file": os.path.join(BASE_DIR, "logs", "dynamic_gesture.log"),  # Log para gestos dinâmicos
    "confidence_threshold": 0.7,  # Limiar para predições confiáveis
    "prediction_cooldown": 30,  # 1 segundo a 30 FPS
    "camera_resolution": [1280, 720],
    "target_fps": 30,
    "train_fps": 30,  # Igualado a target_fps para consistência
    "max_num_hands": 1,
    "min_detection_confidence": 0.5,
    "rf_estimators": 100,
    "lstm_units": 8,  # Reduzido para acelerar treinamento
    "lstm_layers": 2,
    "sequence_length": 15,  # Comprimento da sequência para gestos dinâmicos
    "min_samples_per_class": 30,  # Para gestos estáticos
    "max_dynamic_samples": 5,  # Limite de sequências por gesto dinâmico
    "movement_threshold": 0.02,  # Sensibilidade geral (estabilidade)
    "dynamic_movement_threshold": 0.05,  # Sensibilidade para gestos dinâmicos
    "lstm_epochs": 3,  # Reduzido para acelerar treinamento
    "gesture_types": ["letter", "word", "movement"],
    "train_data_dir": os.path.join(BASE_DIR, "data", "train")
}

if CONFIG["train_fps"] > CONFIG["target_fps"]:
    raise ValueError(f"train_fps ({CONFIG['train_fps']}) não pode ser maior que target_fps ({CONFIG['target_fps']})")

def validate_gesture_type(gesture_type):
    if gesture_type not in CONFIG["gesture_types"]:
        raise ValueError(f"Tipo de gesto inválido: {gesture_type}. Tipos válidos: {CONFIG['gesture_types']}")