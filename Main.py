import cv2
import mediapipe as mp
import logging
import time
import numpy as np
from config.Config import CONFIG, validate_gesture_type
from src.database.Database_manager import DatabaseManager
from src.models.Model_manager import ModelManager
from src.utils.Ui_manager import UIManager
from src.utils.Extract_landmarks import extract_landmarks
from src.core.GestureApp import GestureApp

if __name__ == "__main__":
    try:
        gesture_type = "letter"  # Pode ser configurado dinamicamente
        validate_gesture_type(gesture_type)
        app = GestureApp(gesture_type=gesture_type)
        app.run()
    except Exception as e:
        logging.error(f"Erro ao executar o aplicativo: {e}")
        print(f"[ERROR] Erro ao executar o aplicativo: {e}")