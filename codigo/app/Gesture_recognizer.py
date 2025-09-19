import cv2
import mediapipe as mp
import logging
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Utils import extract_landmarks

import os
from Config import CONFIG

# Cria a pasta do log caso não exista
log_dir = os.path.dirname(CONFIG["log_file"])
os.makedirs(log_dir, exist_ok=True)


class GestureRecognizer:
    SPECIAL_KEYS = {
        "BACKSPACE": lambda word: word[:-1],
        "SPACE": lambda word: word + " "
        # adicione outras teclas especiais aqui, se necessário
    }

    def __init__(self):
        # Configuração do logging
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Conexão com DB e modelo
        self.db = DatabaseManager(CONFIG["db_path"])
        self.model_manager = ModelManager(CONFIG["knn_neighbors"])
        self.ui = UIManager()
        self.labels, self.data, self.gesture_names = self.db.load_gestures()
        if self.labels:
            self.model_manager.train(self.data, self.labels)

        # Configuração da câmera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG["target_fps"])

        # Configuração do MediaPipe Hands
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"]
        )

        self.current_word = ""

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Processamento da imagem
            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
    
            # Reconhecimento de gestos
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    landmarks = extract_landmarks(hand)
                    if landmarks is not None and self.labels:
                        pred, prob = self.model_manager.predict(landmarks)
                        if prob >= CONFIG["confidence_threshold"]:
                            # Se for tecla especial
                            if pred in self.SPECIAL_KEYS:
                                self.current_word = self.SPECIAL_KEYS[pred](self.current_word)
                            else:
                                self.current_word += pred
    
            # Atualização da UI
            status = "Modo: Reconhecimento"
            self.ui.draw_ui(image, status, 0, self.current_word)
    
            # Tecla para sair
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
        # Encerramento
        self.cap.release()
        self.db.close()
        cv2.destroyAllWindows()
