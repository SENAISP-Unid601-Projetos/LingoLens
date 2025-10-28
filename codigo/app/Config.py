import os
import tkinter as tk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detectar tamanho da tela automaticamente
try:
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)
except:
    window_width, window_height = 1024, 768

# Resoluções disponíveis
RESOLUTION_OPTIONS = [
    (1920, 1080),
    (1280, 720), 
    (1024, 768),
    (800, 600),
]

CONFIG = {
    "db_path": os.path.join(BASE_DIR, "data", "gestures.db"),
    "confidence_threshold": 0.80,
    "prediction_cooldown": 45,
    "camera_resolution": (1280, 720),
    "window_size": (window_width, window_height),
    "target_fps": 30,
    "max_num_hands": 2,
    "min_detection_confidence": 0.8,
    "knn_neighbors": 5,
    "log_file": os.path.join(BASE_DIR, "Logs", "Gesture_recognizer.log"),
    
    "train_fps": 30,
    "gesture_types": ["letter", "word", "movement", "libras"],
    "max_sequence_length": 30,
    "train_data_dir": os.path.join(BASE_DIR, "data", "train"),
    
    "resolution_options": RESOLUTION_OPTIONS,
    "current_resolution_index": 0,
    
    "libras_sequence_length": 10,
    "libras_movement_threshold": 0.1,
    "libras_confidence_threshold": 0.6,
    "libras_two_hands": True,
}