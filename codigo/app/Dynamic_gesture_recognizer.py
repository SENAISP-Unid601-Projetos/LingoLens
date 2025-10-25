import cv2
import numpy as np
import logging
import tensorflow as tf
import os
import joblib
import threading
from Database_manager import DatabaseManager
from Preprocess_landmarks import preprocess_landmarks

class DynamicGestureRecognizer:
    def __init__(self, config):
        """Inicializa o reconhecedor de gestos dinâmicos."""
        self.config = config
        self.training_mode = False
        self.dynamic_mode = False
        self.current_gesture_name = ""
        self.dynamic_sequence = []
        self.status_message = ""
        self.model_trained = False
        self.prev_landmarks = None
        self.landmark_history = []
        logging.basicConfig(
            filename=self.config.get("dynamic_log_file", "logs/dynamic_gesture.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        try:
            self.db = DatabaseManager(self.config["db_path"])
            self.data_dynamic, self.labels_dynamic = self.db.load_dynamic_gestures()
            logging.info("Tabela de gestos dinâmicos inicializada")
            self.sequence_length = self.config.get("sequence_length", 15)
            self._init_model()
        except Exception as e:
            logging.error(f"Erro na inicialização do DynamicGestureRecognizer: {e}")
            raise

    def _init_model(self):
        """Inicializa o modelo LSTM para gestos dinâmicos."""
        try:
            if len(self.data_dynamic) > 0 and len(set(self.labels_dynamic)) > 1:
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.sequence_length, 63)),
                    tf.keras.layers.LSTM(self.config["lstm_units"], return_sequences=True),
                    tf.keras.layers.LSTM(self.config["lstm_units"]),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(len(set(self.labels_dynamic)), activation='softmax')
                ])
                self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.model_trained = True
                logging.info(f"Modelo LSTM inicializado com {len(set(self.labels_dynamic))} classes")
            else:
                self.model = None
                self.model_trained = False
                logging.info("Nenhum dado dinâmico suficiente para inicializar o modelo")
        except Exception as e:
            logging.error(f"Erro ao inicializar o modelo LSTM: {e}")
            self.model = None
            self.model_trained = False

    def is_hand_moving(self, landmarks):
        """Verifica se a mão está se movendo significativamente para gestos dinâmicos."""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            self.landmark_history = [landmarks]
            return False
        try:
            self.landmark_history.append(landmarks)
            if len(self.landmark_history) > 3:
                self.landmark_history.pop(0)
            smoothed_landmarks = np.mean(self.landmark_history, axis=0)
            key_indices = [0, 5, 9, 13, 17]
            key_landmark_indices = [i * 3 for i in key_indices] + [i * 3 + 1 for i in key_indices] + [i * 3 + 2 for i in key_indices]
            key_landmarks = [smoothed_landmarks[i] for i in key_landmark_indices]
            key_prev_landmarks = [self.prev_landmarks[i] for i in key_landmark_indices]
            variance = np.var(np.array(key_landmarks) - np.array(key_prev_landmarks))
            self.prev_landmarks = smoothed_landmarks
            return variance >= self.config.get("dynamic_movement_threshold", 0.05)
        except Exception as e:
            logging.error(f"Erro ao verificar movimento da mão: {e}")
            return False

    def process_frame(self, image, landmarks):
        """Processa um frame para gestos dinâmicos."""
        try:
            if not landmarks or len(landmarks) < 63:
                self.status_message = "Nenhum landmark detectado"
                return image

            if self.training_mode:
                self.dynamic_sequence.append(landmarks)
                if len(self.dynamic_sequence) > self.sequence_length:
                    self.dynamic_sequence.pop(0)
                self.status_message = f"Capturando gesto dinâmico: {self.current_gesture_name} ({len(self.dynamic_sequence)} frames)"
            elif self.model_trained and self.is_hand_moving(landmarks):
                self.dynamic_sequence.append(landmarks)
                if len(self.dynamic_sequence) > self.sequence_length:
                    self.dynamic_sequence.pop(0)
                if len(self.dynamic_sequence) == self.sequence_length:
                    sequence = np.array([self.dynamic_sequence])
                    prediction = self.model.predict(sequence, verbose=0)
                    class_idx = np.argmax(prediction, axis=1)[0]
                    probability = np.max(prediction)
                    if probability >= self.config.get("confidence_threshold", 0.7):
                        gesture = self.classes[class_idx]
                        self.status_message = f"Dinâmico: {gesture} ({probability:.2f})"
                        cv2.putText(image, self.status_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    else:
                        self.status_message = "Nenhum gesto dinâmico reconhecido"
                else:
                    self.status_message = f"Coletando frames: {len(self.dynamic_sequence)}/{self.sequence_length}"
            else:
                self.status_message = "Mão parada ou modelo não treinado"
                self.dynamic_sequence = []  # Resetar sequência se a mão está parada

            return image
        except Exception as e:
            logging.error(f"Erro ao processar frame: {e}")
            self.status_message = f"Erro: {str(e)}"
            return image

    def save_gesture_data(self):
        """Salva os dados do gesto dinâmico no banco de dados."""
        try:
            if self.current_gesture_name and self.dynamic_sequence:
                self.status_message = "Salvando gesto dinâmico..."
                # Limitar o número de sequências salvas
                self.db.save_dynamic_gestures([self.current_gesture_name], [self.dynamic_sequence])
                logging.info(f"Gesto dinâmico '{self.current_gesture_name}' salvo com {len(self.dynamic_sequence)} frames")
                self.status_message = f"Gesto '{self.current_gesture_name}' salvo!"
                self.dynamic_sequence = []
            else:
                logging.warning("Nenhum dado de gesto dinâmico para salvar")
                self.status_message = "Nenhum dado para salvar"
        except Exception as e:
            logging.error(f"Erro ao salvar gesto dinâmico: {e}")
            self.status_message = f"Erro ao salvar: {str(e)}"

    def train_and_save_model_lstm(self):
        """Treina e salva o modelo LSTM para gestos dinâmicos em uma thread separada."""
        def train():
            try:
                self.status_message = "Treinando modelo dinâmico..."
                self.data_dynamic, self.labels_dynamic = self.db.load_dynamic_gestures()
                if len(self.data_dynamic) < 2:
                    logging.warning("Menos de 2 gestos dinâmicos para treinamento")
                    self.status_message = "Menos de 2 gestos para treinar"
                    self.model_trained = False
                    return

                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                encoded_labels = label_encoder.fit_transform(self.labels_dynamic)
                self.classes = label_encoder.classes_
                joblib.dump(label_encoder, os.path.join(self.config["data_dir"], "dynamic_label_encoder.pkl"))

                # Limitar o número de amostras por classe
                max_samples = self.config.get("max_dynamic_samples", 5)
                X, y = [], []
                for label in set(self.labels_dynamic):
                    indices = [i for i, l in enumerate(self.labels_dynamic) if l == label]
                    indices = indices[:max_samples]  # Limitar amostras
                    X.extend([self.data_dynamic[i] for i in indices])
                    y.extend([encoded_labels[i] for i in indices])

                X = np.array(X)
                y = np.array(y)
                self.model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.sequence_length, 63)),
                    tf.keras.layers.LSTM(self.config["lstm_units"], return_sequences=True),
                    tf.keras.layers.LSTM(self.config["lstm_units"]),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(len(self.classes), activation='softmax')
                ])
                self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.model.fit(X, y, epochs=self.config.get("lstm_epochs", 3), batch_size=32, validation_split=0.2, verbose=0)
                self.model.save(os.path.join(self.config["data_dir"], "gesture_lstm_model.h5"))
                joblib.dump(self.classes, os.path.join(self.config["data_dir"], "gesture_lstm_classes.pkl"))
                self.model_trained = True
                logging.info(f"Modelo LSTM treinado e salvo com {len(self.classes)} classes")
                self.status_message = "Modelo dinâmico treinado e salvo!"
            except Exception as e:
                logging.error(f"Erro ao treinar modelo LSTM: {e}")
                self.status_message = f"Erro ao treinar: {str(e)}"
                self.model_trained = False

        # Executar treinamento em uma thread separada
        threading.Thread(target=train, daemon=True).start()

    def load_model_lstm(self):
        """Carrega o modelo LSTM salvo."""
        try:
            model_path = os.path.join(self.config["data_dir"], "gesture_lstm_model.h5")
            classes_path = os.path.join(self.config["data_dir"], "gesture_lstm_classes.pkl")
            if os.path.exists(model_path) and os.path.exists(classes_path):
                self.model = tf.keras.models.load_model(model_path)
                self.classes = joblib.load(classes_path)
                self.model_trained = True
                logging.info("Modelo LSTM carregado com sucesso")
            else:
                logging.warning("Arquivos gesture_lstm_model.h5 ou gesture_lstm_classes.pkl não encontrados")
                self.model = None
                self.model_trained = False
                self.status_message = "Modelo LSTM não encontrado"
        except Exception as e:
            logging.error(f"Erro ao carregar modelo LSTM: {e}")
            self.model = None
            self.model_trained = False
            self.status_message = f"Erro ao carregar modelo: {str(e)}"

    def __del__(self):
        """Libera recursos."""
        try:
            self.db.close()
            logging.info("Recursos de gestos dinâmicos liberados")
        except Exception as e:
            logging.error(f"Erro ao liberar recursos: {e}")