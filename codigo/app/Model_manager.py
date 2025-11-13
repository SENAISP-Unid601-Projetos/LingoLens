import numpy as np
import tensorflow as tf
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Config import CONFIG


class ModelManager:
    def __init__(self, gesture_type="letter"):
        self.gesture_type = gesture_type
        self.rf_model = None
        self.lstm_model = None
        self.dynamic_labels_list = []  # 🔹 Guarda os nomes das letras dinâmicas
        print("[INFO] ModelManager inicializado (67 features).")

    # ===========================================================
    def _build_lstm(self, n_classes):
        lstm_units = CONFIG.get("lstm_units", 64)
        seq_len = CONFIG.get("sequence_length", 12)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=(seq_len, 67)),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(max(32, lstm_units // 2), activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    # ===========================================================
    def train(self, static_data=None, static_labels=None, dynamic_data=None, dynamic_labels=None):
        """Treina os modelos com 67 features"""
        try:
            # --- RF (estáticos) ---
            if static_data and static_labels:
                X_static = np.array(static_data, dtype=float)
                y_static = np.array(static_labels)
                if X_static.ndim == 2 and X_static.shape[1] == 67:
                    self.rf_model = RandomForestClassifier(
                        n_estimators=CONFIG.get("rf_estimators", 200),
                        random_state=42
                    )
                    self.rf_model.fit(X_static, y_static)
                    print(f"[ModelManager] RF treinado: {X_static.shape}")
                else:
                    print(f"[ERRO] Shape inválido para RF: {X_static.shape}")

            # --- LSTM (dinâmicos) ---
            if dynamic_data and dynamic_labels:
                X_dyn = np.array(dynamic_data, dtype=float)
                y_dyn = np.array(dynamic_labels)
                if X_dyn.ndim == 3 and X_dyn.shape[2] == 67:
                    labels = sorted(list(set(y_dyn)))
                    self.dynamic_labels_list = labels  # 🔹 Salva os rótulos originais
                    label_to_index = {label: i for i, label in enumerate(labels)}
                    y_encoded = np.array([label_to_index[y] for y in y_dyn])

                    n_classes = len(labels)
                    self.lstm_model = self._build_lstm(n_classes)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_dyn, y_encoded, test_size=0.2, random_state=42
                    )
                    self.lstm_model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=CONFIG.get("epochs", 10),
                        batch_size=CONFIG.get("batch_size", 16),
                        verbose=0
                    )
                    print(f"[ModelManager] LSTM treinado: {X_dyn.shape}")
                else:
                    print(f"[ERRO] Shape inválido para LSTM: {X_dyn.shape}")

        except Exception as e:
            logging.error(f"[ModelManager] Erro no treino: {e}")
            print(f"[ERRO] Erro no treino: {e}")

    # ===========================================================
    def predict(self, data):
        """Predição (67 features)."""
        try:
            if isinstance(data, list):
                data = np.array(data, dtype=float)

            # --- dinâmico ---
            if data.ndim == 2 and data.shape[1] == 67 and self.lstm_model:
                seq = np.expand_dims(data, axis=0)
                preds = self.lstm_model.predict(seq, verbose=0)
                label_index = int(np.argmax(preds))
                prob = float(np.max(preds))

                # 🔹 Mapeia índice de volta para letra
                if self.dynamic_labels_list and label_index < len(self.dynamic_labels_list):
                    label = self.dynamic_labels_list[label_index]
                else:
                    label = str(label_index)

                return label, prob

            # --- estático ---
            elif data.ndim == 1 and len(data) == 67 and self.rf_model:
                pred = self.rf_model.predict([data])[0]
                prob = max(self.rf_model.predict_proba([data])[0])
                return pred, prob

            else:
                print(f"[ERRO] Shape inesperado: {data.shape}")
                return None, 0.0

        except Exception as e:
            logging.error(f"[ModelManager] Erro na predição: {e}")
            print(f"[ERRO] Erro na predição: {e}")
            return None, 0.0
