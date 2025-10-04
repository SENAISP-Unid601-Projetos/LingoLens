import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suprimir aviso do TensorFlow
import tensorflow as tf
from Config import CONFIG

class ModelManager:
    def __init__(self, gesture_type="letter"):
        self.gesture_type = gesture_type
        self.trained = False
        self.label_map = {}  # Para mapear labels para índices (LSTM)
        if gesture_type == "letter":
            self.model = RandomForestClassifier(
                n_estimators=CONFIG["rf_estimators"], random_state=42
            )
        else:  # word, movement
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
            tf.keras.layers.Dense(32, activation='softmax')  # Ajustado dinamicamente no treino
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train(self, data, labels):
        unique_labels = set(labels)
        if len(unique_labels) < 1:
            logging.warning("Nenhum dado disponível para treino.")
            return False

        if len(unique_labels) == 1:
            logging.warning(
                f"Apenas uma classe ({list(unique_labels)[0]}) disponível. Previsões podem não ser confiáveis."
            )

        if self.gesture_type == "letter":
            scores = cross_val_score(self.model, data, labels, cv=5, scoring='accuracy')
            logging.info(f"Precisão média CV (Random Forest): {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
            self.model.fit(data, labels)
        else:
            # Converter labels para índices numéricos
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = [self.label_map[label] for label in labels]
            data = np.array(data)  # Shape: [n_sequências, max_sequence_length, 65]
            self.model.fit(data, np.array(numeric_labels), epochs=10, batch_size=32, validation_split=0.2)
            logging.info(f"Modelo LSTM treinado com {len(data)} sequências e {len(unique_labels)} classes.")

        self.trained = True
        logging.info(f"Modelo treinado com {len(data)} amostras e {len(unique_labels)} classes.")
        return True

    def predict(self, data):
        if not self.trained:
            logging.warning("Modelo ainda não treinado. Retornando None.")
            return None, 0.0

        try:
            if self.gesture_type == "letter":
                prediction = self.model.predict([data])[0]
                probability = self.model.predict_proba([data]).max()
            else:
                data = np.array([data])  # Shape: [1, max_sequence_length, 65]
                probabilities = self.model.predict(data, verbose=0)[0]
                prediction_idx = np.argmax(probabilities)
                probability = probabilities[prediction_idx]
                prediction = list(self.label_map.keys())[list(self.label_map.values()).index(prediction_idx)]
            logging.debug(f"Predição: {prediction} | Probabilidade: {probability:.2f}")
            return prediction, probability
        except NotFittedError:
            logging.error("Erro: modelo não está ajustado corretamente.")
            return None, 0.0
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return None, 0.0