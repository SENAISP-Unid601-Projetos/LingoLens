import cv2
import mediapipe as mp
import logging
import time
import numpy as np
from Config import CONFIG, validate_gesture_type
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Extract_landmarks import extract_landmarks
from GestureApp import GestureApp

if __name__ == "__main__":
    try:
        gesture_type = "letter"  # Pode ser configurado dinamicamente
        validate_gesture_type(gesture_type)
        app = GestureApp(gesture_type=gesture_type)
        app.run()
    except Exception as e:
        logging.error(f"Erro ao executar o aplicativo: {e}")
        print(f"[ERROR] Erro ao executar o aplicativo: {e}")