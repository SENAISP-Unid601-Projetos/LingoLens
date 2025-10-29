import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from Config import CONFIG

class ModelManager:
    def __init__(self, gesture_type="letter"):
        self.gesture_type = gesture_type
        self.trained = False
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        self.lstm_model = None
        self.dynamic_classes = set()
        self.label_map = {}
        self.is_hybrid = CONFIG["use_lstm_for_dynamic"]

    def _build_lstm(self, n_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(CONFIG["sequence_length"], 67)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, static_data, static_labels, dynamic_data, dynamic_labels):
        try:
            if static_labels:
                X = np.array(static_data)
                if X.shape[1] != 67:
                    print(f"[ERRO] Estáticas: {X.shape[1]} features, esperado 67")
                    return False
                self.rf_model.fit(X, static_labels)
                logging.info(f"RF treinado: {X.shape}")

            if dynamic_labels and self.is_hybrid:
                if len(dynamic_labels) >= CONFIG["min_samples_per_class"]:
                    X = np.array(dynamic_data)
                    if X.shape[2] != 67:
                        print(f"[ERRO] Dinâmicas: {X.shape[2]} features, esperado 67")
                        return False
                    unique = list(set(dynamic_labels))
                    self.label_map = {label: i for i, label in enumerate(unique)}
                    y = [self.label_map[l] for l in dynamic_labels]
                    if self.lstm_model is None:
                        self.lstm_model = self._build_lstm(len(unique))
                    self.lstm_model.fit(X, np.array(y), epochs=15, batch_size=16, verbose=0)
                    logging.info(f"LSTM treinado: {X.shape}")
            self.trained = True
            return True
        except Exception as e:
            logging.error(f"Erro no treino: {e}")
            return False

    def predict(self, data):
        if not self.trained:
            return None, 0.0

        is_seq = isinstance(data[0], list) and len(data) == CONFIG["sequence_length"]

        try:
            if is_seq and self.is_hybrid and self.dynamic_classes:
                X = np.array([data])
                if X.shape[2] != 67:
                    return None, 0.0
                probs = self.lstm_model.predict(X, verbose=0)[0]
                idx = np.argmax(probs)
                pred = list(self.label_map.keys())[idx]
                prob = probs[idx]
                if prob >= CONFIG["confidence_threshold"]:
                    return pred, prob
            else:
                frame = data[-1] if is_seq else data
                if len(frame) != 67:
                    return None, 0.0
                X = np.array([frame])
                pred = self.rf_model.predict(X)[0]
                prob = self.rf_model.predict_proba(X)[0].max()
                if prob >= CONFIG["confidence_threshold"]:
                    return pred, prob
            return None, 0.0
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return None, 0.0