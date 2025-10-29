import cv2
import numpy as np
from Config import CONFIG, get_logger
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import os
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

    def set_db(self, db):
        self.db = db

    def start_recording(self, name):
        self.current_gesture_name = name.upper()
        self.dynamic_sequence = []
        self.recording = True
        logger.info(f"Iniciando gravacao: {self.current_gesture_name}")

    def process_frame(self, image, landmarks):
        if self.recording:
            self.dynamic_sequence.append(landmarks.copy())
            cv2.putText(image, f"GRAVANDO: {len(self.dynamic_sequence)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
            self.status_message = f"{self.current_gesture_name} salvo!"
        else:
            logger.warning(f"Muito curto: {seq_len} frames")
            self.status_message = "Muito curto!"
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
                LSTM(64, return_sequences=True, input_shape=(max_len, 63)),
                LSTM(32),
                Dense(32, activation='relu'),
                Dense(len(self.label_encoder.classes_), activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X, y, epochs=50, batch_size=8, validation_split=0.2, verbose=0)
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

    def process_frame_recognition(self, image, landmarks):
        if not self.model_trained or not landmarks:
            return image
        self.recognition_sequence.append(landmarks.copy())
        if len(self.recognition_sequence) > self.max_sequence_length:
            self.recognition_sequence.pop(0)
        if len(self.recognition_sequence) >= CONFIG["min_frames"]:
            seq = np.array(self.recognition_sequence)
            padded = np.zeros((1, self.max_sequence_length, 63))
            padded[0, :len(seq), :] = seq
            prob = self.model.predict(padded, verbose=0)[0]
            idx = np.argmax(prob)
            if prob[idx] > 0.8:
                label = self.label_encoder.classes_[idx]
                cv2.putText(image, f"{label} ({prob[idx]:.2f})", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                self.recognition_sequence = []
        return image