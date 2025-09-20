import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "db_path": os.path.join(BASE_DIR, "data", "gestures.db"),
    "confidence_threshold": 0.5,
    "prediction_cooldown": 15,
    "camera_resolution": (1280, 720),
    "target_fps": 30,
    "max_num_hands": 1,
    "min_detection_confidence": 0.5,
    "knn_neighbors": 5,
    "log_file": os.path.join(BASE_DIR, "logs", "Gesture_recognizer.log"),
    
    # Configs adicionais para treino
    "train_fps": 15,  # FPS reduzido para coleta de dados, evita frames duplicados
    "gesture_types": ["letter", "word", "movement"],  # Tipos de gestos
    "max_sequence_length": 30,  # Para movimentos que precisam de várias frames
    "train_data_dir": os.path.join(BASE_DIR, "data", "train"),  # Onde salvar dados brutos
}
