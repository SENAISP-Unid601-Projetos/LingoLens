import cv2
import mediapipe as mp
import logging
import time
import numpy as np
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Extract_landmarks import extract_landmarks

class GestureApp:
    def __init__(self, gesture_type="letter"):
        print("[INFO] Inicializando GestureApp...")
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.db = DatabaseManager(CONFIG["db_path"])
        self.model_manager = ModelManager(gesture_type=gesture_type)
        self.ui = UIManager()
        self.gesture_type = gesture_type.lower()

        # Carregar dados existentes
        self.data_dict = self.db.load_gestures(gesture_type=self.gesture_type)
        self.labels = []
        self.data = []
        for label, landmarks_list in self.data_dict.items():
            self.labels.extend([label] * len(landmarks_list))
            self.data.extend(landmarks_list)
        
        if self.labels:
            self.model_manager.train(self.data, self.labels)
            print(f"[INFO] Modelo carregado com {len(set(self.labels))} gestos: {set(self.labels)}")
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
        self.mode = "teste"
        self.new_gesture_name = ""
        self.new_gesture_data = []
        self.input_text = ""
        self.is_input_active = False
        self.sample_count = 0
        self.frame_count = 0
        self.prev_landmarks = None
        self.last_prediction_time = time.time()
        self.cooldown = 0

    def is_hand_stable(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return False
        variance = np.var(np.array(landmarks) - np.array(self.prev_landmarks))
        self.prev_landmarks = landmarks
        return variance < 0.01

    def run(self):
        print("[INFO] Teclas: Q=Sair C:Limpar T:Treino S:Gesto")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Falha ao capturar frame.")
                break

            self.frame_count += 1
            if self.frame_count % (CONFIG["target_fps"] // CONFIG["train_fps"]) != 0 and self.mode == "treino":
                continue

            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logging.debug(f"Dimensões da imagem: {image.shape}")
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    landmarks = extract_landmarks(hand, image.shape)
                    if landmarks is not None and len(landmarks) > 0:
                        logging.debug(f"Tipo de landmarks: {type(landmarks)}")
                        if self.mode == "treino" and self.new_gesture_name and self.is_hand_stable(landmarks):
                            self.new_gesture_data.append(landmarks)
                            self.sample_count += 1
                        elif self.mode == "teste" and self.labels and time.time() - self.last_prediction_time > self.cooldown:
                            pred, prob = self.model_manager.predict(landmarks)
                            if pred and prob >= CONFIG["confidence_threshold"]:
                                self.current_word += pred
                                cv2.putText(image, f"Predição: {pred} ({prob:.2f})",
                                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                logging.info(f"Predição: {pred} | Probabilidade: {prob:.2f}")
                                self.last_prediction_time = time.time()
                                self.cooldown = CONFIG["prediction_cooldown"] / CONFIG["target_fps"]

            self.ui.draw_ui(
                image, f"Modo: {'Treino' if self.mode=='treino' else 'Teste'} ({self.gesture_type})",
                self.cooldown, self.current_word, self.sample_count, self.input_text,
                self.is_input_active, self.new_gesture_name
            )

            cv2.imshow("GestureApp", image)
            key = cv2.waitKey(1) & 0xFF

            if self.is_input_active:
                if key == 13:  # Enter
                    self.new_gesture_name = self.input_text.upper()
                    self.is_input_active = False
                    self.input_text = ""
                    self.sample_count = 0
                    if self.new_gesture_name:
                        print(f"[INFO] Modo Treino ativado para '{self.new_gesture_name}'")
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
                    self.new_gesture_data = []
                    self.sample_count = 0
                    print("[INFO] Palavra atual e dados de treino limpos.")
                elif key == ord("t"):
                    self.mode = "treino"
                    self.is_input_active = True
                    self.input_text = ""
                    self.new_gesture_data = []
                    self.sample_count = 0
                    self.prev_landmarks = None
                    print("[INFO] Modo de entrada de texto ativado. Digite na janela.")
                elif key == ord("s"):
                    if self.mode == "treino" and self.new_gesture_name and self.new_gesture_data:
                        if len(self.new_gesture_data) < CONFIG["min_samples_per_class"]:
                            print(f"[WARNING] Coletados poucos samples ({len(self.new_gesture_data)}). Recomenda-se {CONFIG['min_samples_per_class']}.")
                        new_labels = [self.new_gesture_name] * len(self.new_gesture_data)
                        new_gesture_types = [self.gesture_type] * len(self.new_gesture_data)
                        self.labels.extend(new_labels)
                        self.data.extend(self.new_gesture_data)
                        self.db.save_gestures(new_labels, self.new_gesture_data, new_gesture_types)
                        self.model_manager.train(self.data, self.labels)
                        print(f"[INFO] Gestos de '{self.new_gesture_name}' salvos e modelo atualizado.")
                    elif self.mode == "treino" and not self.new_gesture_data:
                        print("[WARNING] Nenhum dado de gesto capturado para salvar.")
                    self.mode = "teste"
                    self.new_gesture_name = ""
                    self.new_gesture_data = []
                    self.is_input_active = False
                    self.sample_count = 0
                    print("[INFO] Modo Teste ativado.")

        self.cap.release()
        cv2.destroyAllWindows()
        self.db.close()
        print("[INFO] GestureApp encerrado.")

if __name__ == "__main__":
    app = GestureApp(gesture_type="letter")
    app.run()