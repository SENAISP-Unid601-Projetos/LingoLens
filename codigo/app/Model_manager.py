import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from Config import CONFIG

class ModelManager:
    def __init__(self, gesture_type="letter"):
        from Config import validate_gesture_type
        validate_gesture_type(gesture_type)
        self.gesture_type = gesture_type
        self.trained = False
        self.labels = []  # Inicializar labels
        self.label_map = {}
        if gesture_type == "letter":
            self.model = RandomForestClassifier(
                n_estimators=CONFIG["rf_estimators"], 
                max_depth=15,  # Aumentado para mais flexibilidade
                min_samples_split=5,  # Mitigar overfitting
                random_state=42
            )
        else:
            self.model = self._build_lstm_model()

    def _build_lstm_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                CONFIG["lstm_units"],
                return_sequences=True,
                input_shape=(CONFIG["max_sequence_length"], 65)
            ),
            tf.keras.layers.LSTM(CONFIG["lstm_units"]),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, data, labels):
        if not data or not labels or len(data) != len(labels):
            logging.error("Dados ou labels inválidos para treinamento")
            return False

        unique_labels = set(labels)
        if len(unique_labels) < 1:
            logging.error("Nenhum dado disponível para treino")
            return False

        if len(unique_labels) == 1:
            logging.warning(f"Treinando com apenas 1 classe ({list(unique_labels)[0]}). Predições não serão discriminativas.")

        try:
            if self.gesture_type == "letter":
                data = np.array(data)
                if data.shape[1] != 65:
                    logging.error(f"Formato de dados inválido. Esperado (N, 65), recebido (N, {data.shape[1]})")
                    return False
                if len(unique_labels) >= 2:
                    scores = cross_val_score(self.model, data, labels, cv=5, scoring='accuracy')
                    logging.info(f"Precisão média CV (Random Forest): {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
                self.model.fit(data, labels)
            else:
                data = np.array(data)
                if data.shape[1:] != (CONFIG["max_sequence_length"], 65):
                    logging.error(f"Formato de dados inválido. Esperado (N, {CONFIG['max_sequence_length']}, 65), recebido {data.shape}")
                    return False
                self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
                numeric_labels = [self.label_map[label] for label in labels]
                if self.model.layers[-1].units != len(unique_labels):
                    self.model.layers.pop()
                    self.model.add(tf.keras.layers.Dense(len(unique_labels), activation='softmax'))
                    self.model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                self.model.fit(data, np.array(numeric_labels), epochs=10, batch_size=32, validation_split=0.2, verbose=0)
                logging.info(f"Modelo LSTM treinado com {len(data)} sequências e {len(unique_labels)} classes")

            self.trained = True
            self.labels = labels  # Armazenar labels para uso em predict
            logging.info(f"Modelo treinado com {len(data)} amostras e {len(unique_labels)} classes: {unique_labels}")
            return True
        except Exception as e:
            logging.error(f"Erro ao treinar modelo: {e}")
            return False

    def predict(self, data):
        if not self.trained:
            logging.warning("Modelo ainda não treinado. Retornando None")
            return None, 0.0

        unique_labels = set(self.labels) if self.labels else set()
        if len(unique_labels) < 2:
            logging.warning(f"Predição com {len(unique_labels)} classe(s). Resultado não discriminativo.")

        try:
            if self.gesture_type == "letter":
                data = np.array(data)
                if data.shape != (65,):
                    logging.error(f"Formato de dados inválido para predição. Esperado (65,), recebido {data.shape}")
                    return None, 0.0
                prediction = self.model.predict([data])[0]
                probability = self.model.predict_proba([data]).max()
            else:
                data = np.array([data])
                if data.shape != (1, CONFIG["max_sequence_length"], 65):
                    logging.error(f"Formato de dados inválido para predição. Esperado (1, {CONFIG['max_sequence_length']}, 65), recebido {data.shape}")
                    return None, 0.0
                probabilities = self.model.predict(data, verbose=0)[0]
                prediction_idx = np.argmax(probabilities)
                probability = probabilities[prediction_idx]
                prediction = list(self.label_map.keys())[list(self.label_map.values()).index(prediction_idx)]
            logging.debug(f"Predição: {prediction} | Probabilidade: {probability:.2f}")
            return prediction, probability
        except NotFittedError:
            logging.error("Erro: modelo não está ajustado corretamente")
            return None, 0.0
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return None, 0.0