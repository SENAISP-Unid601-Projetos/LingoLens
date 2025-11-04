import cv2
import numpy as np
from Config import CONFIG, get_logger
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
import os
import time
import pickle

logger = get_logger("Dynamic")

class DynamicGestureRecognizer:
    def __init__(self, config):
        self.config = config
        self.db = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.model_trained = False
        self.recording = False
        self.current_gesture_name = ""
        self.dynamic_sequence = []
        self.status_message = ""
        self.classes_ = []
        self.recognition_sequence = []
        self.max_sequence_length = 63

        # DETECÇÃO DE FIM 
        self.last_landmarks = None
        self.stable_frames = 0
        self.required_stable = 3          
        self.movement_threshold = 0.007
        self.last_prediction_time = 0
        self.prediction_cooldown = 3.0    

        # NOVO: MOSTRAR POR 3s
        self.current_prediction = None     
        self.prediction_end_time = 0

    def set_db(self, db):
        self.db = db

    def start_recording(self, name):
        self.current_gesture_name = name.upper()
        self.dynamic_sequence = []
        self.recording = True
        logger.info(f"Iniciando gravacao: {self.current_gesture_name}")

    def process_frame(self, image, landmarks):
        if self.recording:
            if len(self.dynamic_sequence) < CONFIG["max_samples_per_gesture"]:
                self.dynamic_sequence.append(landmarks.copy())
                cv2.putText(image, f"GRAVANDO: {len(self.dynamic_sequence)}/{CONFIG['max_samples_per_gesture']}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # PARA AUTOMATICAMENTE
                self.stop_recording()
                cv2.putText(image, f"LIMITE DE {CONFIG['max_samples_per_gesture']} ATINGIDO!", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return image

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        seq_len = len(self.dynamic_sequence)
        if seq_len >= CONFIG["min_frames"]:
            if self.db:
                self.db.save_dynamic_gesture(self.current_gesture_name, self.dynamic_sequence.copy())
            logger.info(f"'{self.current_gesture_name}' salvo! ({seq_len} frames)")
            self.status_message = f"{self.current_gesture_name} salvo! ({seq_len})"
        else:
            logger.warning(f"Muito curto: {seq_len} frames")
            self.status_message = "Muito curto!"

        # SE ATINGIU O LIMITE
        if seq_len >= CONFIG["max_samples_per_gesture"]:
            self.status_message = f"LIMITE DE {CONFIG['max_samples_per_gesture']} ATINGIDO!"

        self.current_gesture_name = ""
        self.dynamic_sequence = []

    def train_and_save_model_lstm(self):
        logger.info("Carregando gestos dinamicos...")
        try:
            gestures = self.db.load_all_dynamic_gestures()
            sequences, labels = [], []
            for name, seq_list in gestures.items():
                for seq in seq_list:
                    if len(seq) >= CONFIG["min_frames"]:
                        arr = np.array(seq, dtype=np.float32)
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        if arr.shape[1] == 63:
                            sequences.append(arr)
                            labels.append(name)
            if len(sequences) < 4 or len(set(labels)) < 2:
                logger.error("Dados insuficientes")
                return
            max_len = max(s.shape[0] for s in sequences)
            X = np.zeros((len(sequences), max_len, 63))
            for i, s in enumerate(sequences):
                X[i, :s.shape[0], :] = s
            y = to_categorical(self.label_encoder.fit_transform(labels))
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(max_len, 63), 
                     kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                LSTM(32, kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
                Dropout(0.3),
                Dense(len(self.label_encoder.classes_), activation='softmax')
            ])
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            X, y = shuffle(X, y, random_state=42)

            early_stop = EarlyStopping(monitor='val_loss', patience=15, Restaurar_best_weights=True)

            model.fit(X, y, epochs=100, batch_size=8, validation_split=0.3, shuffle=True, callbacks=[early_stop], verbose=0)
            os.makedirs(CONFIG["model_dir"], exist_ok=True)
            model.save(os.path.join(CONFIG["model_dir"], "lstm_dynamic_model.h5"))
            with open(os.path.join(CONFIG["model_dir"], "lstm_classes.pkl"), 'wb') as f:
                pickle.dump(self.label_encoder.classes_, f)
            self.model = model
            self.model_trained = True
            self.classes_ = self.label_encoder.classes_
            self.max_sequence_length = max_len
            logger.info(f"LSTM treinado: {self.classes_}")
        except Exception as e:
            logger.error(f"Erro no treino: {e}")

    def load_model_lstm(self):
        path = os.path.join(CONFIG["model_dir"], "lstm_dynamic_model.h5")
        cls_path = os.path.join(CONFIG["model_dir"], "lstm_classes.pkl")
        if os.path.exists(path) and os.path.exists(cls_path):
            from tensorflow.keras.models import load_model
            self.model = load_model(path)
            with open(cls_path, 'rb') as f:
                self.label_encoder.classes_ = pickle.load(f)
            self.classes_ = self.label_encoder.classes_
            self.model_trained = True
            self.max_sequence_length = self.model.input_shape[1]
            logger.info(f"Modelo LSTM carregado: {len(self.classes_)} classes")
            return True
        return False

    def process_frame_recognition(self, image, landmarks, get_stable_func=None):
        if not self.model_trained or not landmarks:
            return image

        current_time = time.time()

        # MOSTRA POR 3s
        if self.current_prediction and current_time < self.prediction_end_time:
            label, conf = self.current_prediction
            cv2.putText(image, f"PALAVRA: {label} ({conf:.2f})", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return image

        self.recognition_sequence.append(landmarks.copy())
        if len(self.recognition_sequence) > self.max_sequence_length:
            self.recognition_sequence.pop(0)

        # === USA A FUNÇÃO ESTÁTICA DE ESTABILIDADE ===
        if get_stable_func:
            is_stable = get_stable_func(landmarks)
            if is_stable:
                self.stable_frames += 1
            else:
                self.stable_frames = 0
        else:
            # fallback antigo
            if self.last_landmarks is not None:
                diff = np.mean(np.abs(np.array(landmarks) - np.array(self.last_landmarks)))
                if diff < self.movement_threshold:
                    self.stable_frames += 1
                else:
                    self.stable_frames = 0
            self.last_landmarks = landmarks.copy()

        # === PREDIÇÃO QUANDO GESTO TERMINAR ===
        if len(self.recognition_sequence) >= 20 and self.stable_frames >= self.required_stable:
            seq = np.array(self.recognition_sequence, dtype=np.float32)

            # RESPEITA O max_len DO MODELO TREINADO
            target_len = self.max_sequence_length  # vem do treino!

            if seq.shape[0] > target_len:
                seq = seq[-target_len:]  # últimos N frames
            elif seq.shape[0] < target_len:
                pad_width = ((0, target_len - seq.shape[0]), (0, 0))
                seq = np.pad(seq, pad_width, mode='constant')

            padded = np.zeros((1, target_len, 63))
            padded[0, :, :] = seq

            prob = self.model.predict(padded, verbose=0)[0]
            top2 = np.argsort(prob)[-2:][::-1]
            logger.info(f"TOP2: {self.label_encoder.classes_[top2[0]]} ({prob[top2[0]]:.2f}), "
                f"{self.label_encoder.classes_[top2[1]]} ({prob[top2[1]]:.2f})")
            idx = np.argmax(prob)
            confidence = prob[idx]

            if confidence > 0.7:  # AUMENTOU PARA EVITAR FALSOS
                label = self.label_encoder.classes_[idx]
                logger.info(f"RECONHECIDO: {label} ({confidence:.2f})")

                self.current_prediction = (label, confidence)
                self.prediction_end_time = current_time + 3.0

                cv2.putText(image, f"PALAVRA: {label} ({confidence:.2f})", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return image