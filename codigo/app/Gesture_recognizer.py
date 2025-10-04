import cv2
import mediapipe as mp
import logging
import numpy as np
import os
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Utils import extract_landmarks

class GestureRecognizer:
    def __init__(self, specific_letter=None, gesture_type="letter"):
        print("[INFO] Inicializando GestureRecognizer...")
        # Criar diretório de logs se não existir
        os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("Logging configurado.")
        self.db = DatabaseManager(CONFIG["db_path"])
        print("[INFO] Conexão com banco de dados criada.")
        self.gesture_type = gesture_type
        self.model_manager = ModelManager(gesture_type=gesture_type)
        self.ui = UIManager()
        self.labels, self.data, self.gesture_names = self.db.load_gestures(gesture_type=gesture_type)
        if self.labels:
            self.model_manager.train(self.data, self.labels)
            print(f"[INFO] Modelo carregado com {len(self.labels)} gestos.")
        else:
            print("[INFO] Nenhum gesto treinado ainda.")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG["train_fps"])

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"],
            static_image_mode=False,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.current_word = ""
        self.mode = "treino" if specific_letter else "teste"
        self.new_gesture_name = specific_letter.upper() if specific_letter else ""
        self.new_gesture_data = []
        self.input_text = ""
        self.is_input_active = not specific_letter
        self.sample_count = 0
        self.frame_count = 0
        self.prev_landmarks = None
        self.sequence_buffer = []  # Buffer para sequências dinâmicas

    def is_hand_stable(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return False
        variance = np.var(np.array(landmarks) - np.array(self.prev_landmarks))
        self.prev_landmarks = landmarks
        return variance < 0.01

    def detect_gesture_type(self, sequence_buffer):
        if len(sequence_buffer) < 10:  # Mínimo de frames para variância
            return "letter"
        variance = np.var(np.array(sequence_buffer), axis=0).mean()
        return "word" if variance >= 0.01 else "letter"

    def run(self):
        print("[INFO] Teclas: Q=Sair C:Limpar T:Treino S:Gesto")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Falha ao capturar frame.")
                break

            self.frame_count += 1
            if self.mode == "treino" and self.frame_count % (CONFIG["target_fps"] // CONFIG["train_fps"]) != 0:
                continue

            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)  # Usar array RGB diretamente

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    landmarks = extract_landmarks(hand, image_shape=image.shape)
                    if landmarks is not None and len(landmarks) > 0:
                        if self.mode == "treino" and self.new_gesture_name:
                            if self.gesture_type == "letter":
                                if self.is_hand_stable(landmarks):
                                    self.new_gesture_data.append(landmarks)
                                    self.sample_count += 1
                            else:  # word, movement
                                self.sequence_buffer.append(landmarks)
                                if len(self.sequence_buffer) >= CONFIG["max_sequence_length"]:
                                    self.new_gesture_data.append(self.sequence_buffer[-CONFIG["max_sequence_length"]:])
                                    self.sequence_buffer = []
                                    self.sample_count += 1
                        elif self.mode == "teste":
                            self.sequence_buffer.append(landmarks)
                            if len(self.sequence_buffer) > CONFIG["max_sequence_length"]:
                                self.sequence_buffer.pop(0)  # Manter 30 frames
                            self.gesture_type = self.detect_gesture_type(self.sequence_buffer)
                            self.model_manager = ModelManager(gesture_type=self.gesture_type)
                            if self.labels:
                                if self.gesture_type == "letter" and len(self.sequence_buffer) >= 1:
                                    pred, prob = self.model_manager.predict(self.sequence_buffer[-1])
                                elif self.gesture_type == "word" and len(self.sequence_buffer) == CONFIG["max_sequence_length"]:
                                    pred, prob = self.model_manager.predict(self.sequence_buffer)
                                else:
                                    pred, prob = None, 0.0
                                if pred and prob >= CONFIG["confidence_threshold"]:
                                    self.current_word += pred
                                    logging.info(f"Predição: {pred} | Probabilidade: {prob:.2f}")

            self.ui.draw_ui(
                image, f"Modo: {'Treino' if self.mode=='treino' else 'Teste'} ({self.gesture_type})",
                0, self.current_word, self.sample_count, self.input_text,
                self.is_input_active, self.new_gesture_name
            )
            cv2.imshow("GestureRecognizer", image)
            key = cv2.waitKey(1) & 0xFF

            if self.is_input_active:
                if key == 13:  # Enter
                    self.new_gesture_name = self.input_text.upper()
                    self.is_input_active = False
                    self.input_text = ""
                    self.sample_count = 0
                    if self.new_gesture_name:
                        print(f"[INFO] Modo Treino ativado para '{self.new_gesture_name}' ({self.gesture_type})")
                        self.mode = "treino"
                    else:
                        print("[INFO] Nenhum nome de gesto fornecido.")
                elif key == 8:  # Backspace
                    self.input_text = self.input_text[:-1]
                elif key in [ord(c) for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"]:
                    self.input_text += chr(key).upper()
            else:
                if key == ord("q"):
                    print("[INFO] Saindo do aplicativo...")
                    break
                elif key == ord("c"):
                    self.current_word = ""
                    print("[INFO] Palavra atual limpa.")
                elif key == ord("t") and not self.new_gesture_name:
                    self.mode = "treino"
                    self.is_input_active = True
                    self.input_text = ""
                    self.new_gesture_data = []
                    self.sample_count = 0
                    self.prev_landmarks = None
                    self.sequence_buffer = []
                    print("[INFO] Modo de entrada de texto ativado. Digite na janela.")
                elif key == ord("s"):
                    if self.mode == "treino" and self.new_gesture_name and self.new_gesture_data:
                        if len(self.new_gesture_data) < CONFIG["min_samples_per_class"]:
                            print(f"[WARNING] Coletados poucos samples ({len(self.new_gesture_data)}). Recomenda-se {CONFIG['min_samples_per_class']}.")
                        self.labels += [self.new_gesture_name] * len(self.new_gesture_data)
                        self.data += self.new_gesture_data
                        self.db.save_gestures(self.labels, self.data, [self.gesture_type] * len(self.new_gesture_data))
                        self.model_manager.train(self.data, self.labels)
                        print(f"[INFO] Gestos de '{self.new_gesture_name}' ({self.gesture_type}) salvos e modelo atualizado.")
                    elif self.mode == "treino" and not self.new_gesture_data:
                        print("[WARNING] Nenhum dado de gesto capturado para salvar.")
                    self.mode = "teste"
                    self.new_gesture_name = ""
                    self.new_gesture_data = []
                    self.is_input_active = False
                    self.sample_count = 0
                    self.sequence_buffer = []
                    print("[INFO] Modo Teste ativado.")

        self.cap.release()
        cv2.destroyAllWindows()
        self.db.close()
        print("[INFO] GestureRecognizer encerrado.")