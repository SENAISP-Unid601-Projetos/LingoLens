import cv2
import mediapipe as mp
import logging
import os
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Utils import extract_landmarks

class GestureRecognizer:
    def __init__(self):
        print("[INFO] Inicializando GestureRecognizer...")

        # garante que a pasta de logs exista
        os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)

        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        print("[INFO] Logging configurado.")

        self.db = DatabaseManager(CONFIG["db_path"])
        print("[INFO] Conexão com banco de dados criada.")

        self.model_manager = ModelManager(CONFIG["knn_neighbors"])
        self.ui = UIManager()
        print("[INFO] ModelManager e UIManager inicializados.")

        self.labels, self.data, self.gesture_names = self.db.load_gestures()
        if self.labels:
            print(f"[INFO] Treinando modelo com {len(self.labels)} gestos...")
            self.model_manager.train(self.data, self.labels)
        else:
            print("[INFO] Nenhum gesto encontrado para treinar.")

        print("[INFO] Inicializando captura de vídeo...")
        self.cap = None
        for i in range(5):  # tenta abrir câmeras de 0 a 4
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                self.cap = temp_cap
                print(f"[INFO] Captura de vídeo pronta na câmera {i}.")
                break
        if self.cap is None:
            print("[ERROR] Nenhuma câmera disponível.")
            exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG["target_fps"])

        print("[INFO] Captura de vídeo pronta.")

        print("[INFO] Inicializando MediaPipe Hands...")
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"]
        )
        print("[INFO] MediaPipe Hands pronto.")

        self.current_word = ""
        print("[INFO] GestureRecognizer inicializado com sucesso!")

    def run(self):
        print("[INFO] Iniciando loop principal...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Falha ao capturar frame da câmera.")
                break

            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Status antes do processamento
            print("[DEBUG] Processando frame...")
            results = self.hands.process(rgb)
            print("[DEBUG] Frame processado.")

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    landmarks = extract_landmarks(hand)
                    if landmarks is not None and self.labels:
                        pred, prob = self.model_manager.predict(landmarks)
                        if pred is not None and prob >= CONFIG["confidence_threshold"]:
                            print(f"[INFO] Reconhecido gesto: {pred} ({prob:.2f})")
                            self.current_word += pred

            status = "Modo: Reconhecimento"
            self.ui.draw_ui(image, status, 0, self.current_word)

            cv2.imshow("Gesture Recognizer", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Usuário saiu do loop.")
                break

        print("[INFO] Encerrando GestureRecognizer...")
        self.cap.release()
        self.db.close()
        cv2.destroyAllWindows()
        print("[INFO] GestureRecognizer encerrado.")
