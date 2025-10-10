import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from Database_manager import DatabaseManager
from Config import CONFIG
import logging
import os
import time

# Configurar logging
os.makedirs(os.path.dirname(CONFIG['log_file']), exist_ok=True)
logging.basicConfig(filename=CONFIG['log_file'], level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class GestureRecognizer:
    def __init__(self, specific_letter=None, gesture_type='letter'):
        self.db_manager = DatabaseManager(db_path=CONFIG['db_path'])
        self.rf_model = None
        self.lstm_model = None
        self.label_encoder = LabelEncoder()
        self.sequence_length = CONFIG['max_sequence_length']
        self.landmarks_buffer = []
        self.new_gesture_name = specific_letter
        self.gesture_type = gesture_type
        self.training_mode = False
        self.input_active = False
        self.input_text = ""
        self.current_letter = specific_letter if specific_letter else ""
        self.word = ""
        self.cooldown = 0
        self.last_prediction_time = time.time()
        self.cursor_blink_time = 0

    def train_gestures(self, gesture_name, landmarks, gesture_type='letter'):
        try:
            self.new_gesture_name = gesture_name
            self.gesture_type = gesture_type
            self.landmarks_buffer = landmarks

            # Forçar 'letter' para gestos de uma letra
            if len(gesture_name) == 1:
                self.gesture_type = 'letter'

            if len(self.landmarks_buffer) == 0:
                logging.warning("Nenhum dado de gesto para treinar.")
                return False

            # Carregar dados existentes
            existing_data = self.db_manager.load_gestures(self.gesture_type)
            if gesture_name not in existing_data:
                existing_data[gesture_name] = []
            existing_data[gesture_name].extend(self.landmarks_buffer)

            # Salvar no banco (provavelmente onde o problema de apenas A/B ocorre)
            self.db_manager.save_gestures(gesture_name, self.gesture_type, self.landmarks_buffer)

            # Treinar modelo
            if self.gesture_type == 'letter':
                X, y = [], []
                logging.info(f"Classes carregadas para treinamento ({self.gesture_type}): {existing_data.keys()}")
                for name, landmarks_list in existing_data.items():
                    for landmarks in landmarks_list:
                        X.append(np.array(landmarks).flatten())
                        y.append(name)

                if len(set(y)) < 2:
                    logging.warning(f"Apenas uma classe disponível para '{self.gesture_type}': {set(y)}")
                    return False

                X = np.array(X)
                y = np.array(y)
                self.rf_model = RandomForestClassifier(n_estimators=CONFIG['rf_estimators'], random_state=42)
                scores = cross_val_score(self.rf_model, X, y, cv=5)
                logging.info(f"Precisão média CV (Random Forest): {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
                self.rf_model.fit(X, y)
                logging.info(f"Modelo Random Forest treinado para '{gesture_name}' ({self.gesture_type}).")
            else:
                X, y = [], []
                for name, landmarks_list in existing_data.items():
                    for landmarks in landmarks_list:
                        if len(landmarks) >= self.sequence_length:
                            X.append(landmarks[:self.sequence_length])
                            y.append(name)

                if len(set(y)) < 2:
                    logging.warning(f"Apenas uma classe disponível para '{self.gesture_type}': {set(y)}")
                    return False

                X = np.array(X)
                y_encoded = self.label_encoder.fit_transform(y)

                self.lstm_model = Sequential([
                    LSTM(CONFIG['lstm_units'], input_shape=(self.sequence_length, X.shape[2]), return_sequences=True),
                    LSTM(CONFIG['lstm_units'] // 2),
                    Dense(CONFIG['lstm_units'] // 2, activation='relu'),
                    Dense(len(set(y_encoded)), activation='softmax')
                ])
                self.lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.lstm_model.fit(X, y_encoded, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
                logging.info(f"Modelo LSTM treinado para '{gesture_name}' ({self.gesture_type}).")

            return True
        except Exception as e:
            logging.error(f"Erro ao treinar gestos: {e}")
            return False

    def detect_gesture_type(self, landmarks):
        try:
            recent_landmarks = landmarks[-20:] if len(landmarks) >= 20 else landmarks
            recent_variance = np.var(recent_landmarks, axis=0).mean() if recent_landmarks else 0
            return 'letter' if recent_variance < CONFIG['confidence_threshold'] or len(landmarks) < 20 else 'word'
        except Exception as e:
            logging.error(f"Erro ao detectar tipo de gesto: {e}")
            return 'letter'

    def predict(self, landmarks):
        try:
            gesture_type = self.detect_gesture_type(landmarks)
            if gesture_type == 'letter':
                if self.rf_model is None:
                    return None, 0.0
                X = np.array(landmarks[-1]).flatten().reshape(1, -1)
                prediction = self.rf_model.predict(X)
                probability = self.rf_model.predict_proba(X)[0].max()
                return prediction[0], probability
            else:
                if self.lstm_model is None or len(landmarks) < self.sequence_length:
                    return None, 0.0
                X = np.array([landmarks[-self.sequence_length:]])
                prediction = self.lstm_model.predict(X, verbose=0)
                predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction[0])])[0]
                probability = np.max(prediction[0])
                return predicted_label, probability
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return None, 0.0

    def run(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=CONFIG['max_num_hands'],
            min_detection_confidence=CONFIG['min_detection_confidence']
        )
        mp_draw = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_resolution'][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_resolution'][1])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            status = f"Treino ({self.gesture_type})" if self.training_mode else f"Teste ({self.gesture_type})"
            sample_count = len(self.landmarks_buffer)

            # Desenhar interface
            height, width = frame.shape[:2]
            text_color = (255, 255, 255)
            outline_color = (0, 0, 0)

            # Fundo semi-transparente para status
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 80), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Status com contador de samples e tipo de gesto
            status_text = f"Modo: {'Treino' if self.training_mode else 'Teste'} | Tipo: {self.gesture_type} | Samples: {sample_count}/{CONFIG['min_samples_per_class']}" if self.training_mode else f"Modo: Teste | Tipo: {self.gesture_type} | Palavra: {self.word}"
            cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, outline_color, 3, cv2.LINE_AA)
            cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

            # Gesto atual
            if self.current_letter:
                cv2.putText(frame, f"Gesto: {self.current_letter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

            # Barra de progresso para gestos dinâmicos
            if self.gesture_type in ["word", "movement"]:
                progress = len(self.landmarks_buffer) / CONFIG['max_sequence_length'] * 100
                cv2.rectangle(frame, (10, 100), (10 + int(progress * 2), 120), (0, 255, 0), -1)
                cv2.putText(frame, f"Frames: {len(self.landmarks_buffer)}/{CONFIG['max_sequence_length']}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Caixa de texto
            if self.input_active:
                box_top_left = (10, 130)
                box_bottom_right = (300, 170)
                cv2.rectangle(frame, box_top_left, box_bottom_right, (0, 0, 255), 2)
                cursor = "_" if (time.time() - self.cursor_blink_time) % 1 < 0.5 else ""
                display_text = f"Nome do gesto: {self.input_text}{cursor}"
                cv2.putText(frame, display_text, (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                cv2.putText(frame, "Letras A-Z, Enter=Confirmar, Backspace=Apagar", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            elif self.training_mode:
                instruction = {
                    "letter": "Mostre a letra na câmera e pressione 's' para salvar",
                    "word": "Grave a sequência de movimento e pressione 's' para salvar",
                    "movement": "Grave a sequência de movimento e pressionar 's' para salvar"
                }.get(self.gesture_type, "Mostre o gesto na câmera")
                cv2.putText(frame, instruction, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

            # Instruções
            cv2.putText(frame, "Q:Sair C:Limpar T:Treino S:Gesto", 
                        (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

            # Erro
            if self.cooldown > 0:
                cv2.putText(frame, "Aguarde para nova predição...", 
                            (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2, cv2.LINE_AA)

            # Processar landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    self.landmarks_buffer.append(landmarks)

                    if not self.training_mode and time.time() - self.last_prediction_time > self.cooldown:
                        prediction, probability = self.predict(self.landmarks_buffer)
                        if prediction:
                            self.word = prediction if self.gesture_type == 'letter' else self.word + prediction
                            cv2.putText(frame, f"Predição: {prediction} ({probability:.2f})", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            logging.info(f"Predição: {prediction} | Probabilidade: {probability:.2f}")
                            self.last_prediction_time = time.time()
                            self.cooldown = CONFIG['prediction_cooldown'] / CONFIG['target_fps']

            cv2.imshow("LingoLens", frame)
            key = cv2.waitKey(1000 // CONFIG['target_fps']) & 0xFF

            if key == ord('t'):
                self.training_mode = True
                self.input_active = True
                self.input_text = ""
                self.current_letter = ""
                self.landmarks_buffer = []
            elif key == ord('c'):
                self.word = ""
                self.landmarks_buffer = []
            elif key == ord('s') and self.training_mode and not self.input_active:
                if sample_count >= CONFIG['min_samples_per_class']:
                    self.train_gestures(self.current_letter, self.landmarks_buffer, self.gesture_type)
                    self.training_mode = False
                    self.current_letter = ""
                    self.landmarks_buffer = []
                else:
                    logging.warning(f"Amostras insuficientes: {sample_count}/{CONFIG['min_samples_per_class']}")
            elif self.input_active and key >= 32 and key <= 126:
                self.input_text += chr(key)
            elif self.input_active and key == 8:
                self.input_text = self.input_text[:-1]
            elif self.input_active and key == 13:
                if self.input_text:
                    self.current_letter = self.input_text
                    self.input_active = False
                    self.input_text = ""
                    self.landmarks_buffer = []
            elif key == ord('q'):
                break

            self.cooldown = max(0, self.cooldown - (time.time() - self.last_prediction_time))
            self.last_prediction_time = time.time()

        cap.release()
        cv2.destroyAllWindows()