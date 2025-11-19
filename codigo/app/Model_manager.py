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
        self.dynamic_labels_list = []  # Guarda ordem original das letras dinâmicas
        print("[INFO] ModelManager inicializado (69 features).")

    # ===========================================================
    def _build_lstm(self, n_classes):
        lstm_units = CONFIG.get("lstm_units", 64)
        seq_len = 38  # Agora usa o valor real do Config
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=(seq_len, 69)),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dropout(0.3),  # Evita overfitting em dinâmicas
            tf.keras.layers.Dense(max(32, lstm_units // 2), activation="relu"),
            tf.keras.layers.Dense(n_classes, activation="softmax")
        ])
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ===========================================================
    def train(self, static_data=None, static_labels=None, dynamic_data=None, dynamic_labels=None):
        """Treina os modelos com 69 features"""
        try:
            # --- Random Forest (gestos estáticos) ---
            if static_data is not None and static_labels is not None and len(static_data) > 0:
                X_static = np.array(static_data, dtype=np.float32)
                y_static = np.array(static_labels)

                if X_static.ndim == 2 and X_static.shape[1] == 69:
                    self.rf_model = RandomForestClassifier(
                        n_estimators=CONFIG.get("rf_estimators", 200),
                        random_state=42,
                        n_jobs=-1,
                        class_weight="balanced"
                    )
                    self.rf_model.fit(X_static, y_static)
                    print(f"[ModelManager] RF treinado -> {X_static.shape}")
                else:
                    print(f"[ERRO] Shape inválido para RF: {X_static.shape} (esperado: (*, 69))")

            # --- LSTM (gestos dinâmicos) ---
            if dynamic_data is not None and dynamic_labels is not None and len(dynamic_data) > 0:
                X_dyn = np.array(dynamic_data, dtype=np.float32)
                y_dyn = np.array(dynamic_labels)

                if X_dyn.ndim == 3 and X_dyn.shape[2] == 69:
                    labels = sorted(list(set(y_dyn)))
                    self.dynamic_labels_list = labels
                    label_to_index = {label: i for i, label in enumerate(labels)}
                    y_encoded = np.array([label_to_index[y] for y in y_dyn])

                    n_classes = len(labels)
                    self.lstm_model = self._build_lstm(n_classes)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_dyn, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                    )

                    self.lstm_model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=CONFIG.get("epochs", 15),
                        batch_size=CONFIG.get("batch_size", 16),
                        verbose=1
                    )
                    print(f"[ModelManager] LSTM treinado → {X_dyn.shape} | Classes: {labels}")
                else:
                    print(f"[ERRO] Shape inválido para LSTM: {X_dyn.shape} (esperado: (n, seq_len, 69))")

        except Exception as e:
            logging.error(f"[ModelManager] Erro no treino: {e}")
            print(f"[ERRO] Erro no treino: {e}")

    # ===========================================================
    
    def predict(self, data):
        """Predição compatível com 69 features + sequence_length=30"""
        try:
            if isinstance(data, list):
                data = np.array(data, dtype=np.float32)
            seq_len = 38  # ← também aqui, pra predição usar 38
            # --- Gesto dinâmico: sequência completa (30, 69) ---
            if (data.ndim == 2 and data.shape == (seq_len, 69) and self.lstm_model):
                seq = np.expand_dims(data, axis=0)  # (1, 30, 69)
                preds = self.lstm_model.predict(seq, verbose=0)[0]
                label_index = int(np.argmax(preds))
                prob = float(np.max(preds))
                label = self.dynamic_labels_list[label_index] if label_index < len(self.dynamic_labels_list) else "?"
                return label, prob
            # --- Gesto estático: frame único (69,) ---
            elif (data.ndim == 1 and len(data) == 69 and self.rf_model):
                pred = self.rf_model.predict([data])[0]
                prob = float(np.max(self.rf_model.predict_proba([data])[0]))
                return pred, prob
            else:
                print(f"[ERRO] Shape inesperado na predição: {data.shape} (esperado: (30,69) ou (69,))")
                return None, 0.0
        except Exception as e:
            print(f"[ERRO] Erro na predição: {e}")
            return None, 0.0