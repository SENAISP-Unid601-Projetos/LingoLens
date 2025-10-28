import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import logging
import threading
import cv2
import os
import pickle
from Database_manager import DatabaseManager
from Config import DYNAMIC_MODEL, CONFIG  # ADICIONADO CONFIG

class DynamicGestureRecognizer:
    def __init__(self, config):
        self.config = config
        self.db = DatabaseManager(CONFIG["db_path"])  # CORRIGIDO: com caminho
        self.seq_len = DYNAMIC_MODEL["sequence_length"]
        self.sequence = []
        self.dynamic_sequence = self.sequence
        self.recording = False
        self.current_name = ""
        self.status_message = ""
        self.model = None
        self.classes = []
        self.model_trained = False
        logging.basicConfig(filename=os.path.join(CONFIG["log_file"].replace("app.log", "dynamic.log")), level=logging.INFO)

    def start_recording(self, name):
        self.current_name = name
        self.sequence = []
        self.dynamic_sequence = self.sequence
        self.recording = True
        self.status_message = f"GRAVANDO: {name}"

    def stop_recording(self):
        if len(self.sequence) >= 5:
            self.db.save_dynamic_gestures([self.current_name], [self.sequence])
            self.status_message = f"SALVO: {self.current_name}"
        else:
            self.status_message = "Curto!"
        self.recording = False

    def process_frame(self, image, landmarks):
        if self.recording:
            self.sequence.append(landmarks[:63])
            if len(self.sequence) > self.seq_len:
                self.sequence.pop(0)
            cv2.putText(image, f"REC: {len(self.sequence)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        elif self.model_trained and len(self.sequence) == self.seq_len:
            X = np.array([self.sequence])
            pred = self.model.predict(X, verbose=0)[0]
            conf = np.max(pred)
            if conf > 0.75:
                idx = np.argmax(pred)
                gesture = self.classes[idx]
                self.status_message = f"{gesture} ({conf:.2f})"
                cv2.putText(image, self.status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        return image

    def train_and_save_model_lstm(self):
        def train():
            try:
                X, y = self.db.load_dynamic_gestures()
                if len(X) < 10 or len(set(y)) < 2:
                    self.status_message = "Poucos dados!"
                    return

                valid_X = [x for x in X if len(x) == self.seq_len]
                valid_y = [y[i] for i, x in enumerate(X) if len(x) == self.seq_len]
                if len(valid_X) < 10:
                    self.status_message = "Sequências curtas!"
                    return

                X_arr = np.array(valid_X)
                le = LabelEncoder()
                y_enc = le.fit_transform(valid_y)
                self.classes = le.classes_

                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.seq_len, 63)),
                    tf.keras.layers.LSTM(DYNAMIC_MODEL["lstm_units"], return_sequences=True),
                    tf.keras.layers.LSTM(DYNAMIC_MODEL["lstm_units"]),
                    tf.keras.layers.Dense(DYNAMIC_MODEL["dense_units"], activation='relu'),
                    tf.keras.layers.Dropout(DYNAMIC_MODEL["dropout"]),
                    tf.keras.layers.Dense(len(self.classes), activation='softmax')  # DINÂMICO
                ])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

                history = model.fit(
                    X_arr, y_enc,
                    epochs=DYNAMIC_MODEL["epochs"],
                    batch_size=DYNAMIC_MODEL["batch_size"],
                    validation_split=DYNAMIC_MODEL["validation_split"],
                    verbose=0
                )

                val_acc = history.history['val_accuracy'][-1]
                self.db.save_lstm_model(model)
                self.model = model
                self.model_trained = True

                log_msg = f"DINÂMICO: LSTM | {len(valid_X)} seqs | Val: {val_acc:.3f}"
                logging.info(log_msg)
                print(f"[SUCESSO] {log_msg}")
                self.status_message = f"LSTM: {val_acc:.1%}"

            except Exception as e:
                logging.error(f"Erro LSTM: {e}")
                self.status_message = "Erro"
        threading.Thread(target=train, daemon=True).start()

    def load_model_lstm(self):
        try:
            # Primeiro carrega classes do banco
            self.cursor = self.db.conn.cursor()
            self.cursor.execute("SELECT classes FROM model_lstm ORDER BY id DESC LIMIT 1")
            row = self.cursor.fetchone()
            if row:
                self.classes = pickle.loads(row[0])
            else:
                return

            # Agora cria modelo com número correto de classes
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.seq_len, 63)),
                tf.keras.layers.LSTM(DYNAMIC_MODEL["lstm_units"], return_sequences=True),
                tf.keras.layers.LSTM(DYNAMIC_MODEL["lstm_units"]),
                tf.keras.layers.Dense(DYNAMIC_MODEL["dense_units"], activation='relu'),
                tf.keras.layers.Dropout(DYNAMIC_MODEL["dropout"]),
                tf.keras.layers.Dense(len(self.classes), activation='softmax')  # DINÂMICO
            ])

            classes = self.db.load_lstm_model(model)
            if classes is not None:
                self.model = model
                self.model_trained = True
                logging.info(f"LSTM carregado do banco com {len(self.classes)} classes")
                print(f"[INFO] Modelo dinâmico carregado com {len(self.classes)} classes")
        except Exception as e:
            logging.error(f"Erro ao carregar LSTM: {e}")