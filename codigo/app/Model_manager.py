import logging
import numpy as np
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from Config import CONFIG


class ModelManager:
    def __init__(self, gesture_type="letter"):
        self.gesture_type = gesture_type
        self.trained = False
        self.rf_model = RandomForestClassifier(
            n_estimators=CONFIG.get("rf_estimators", 100),
            random_state=42
        )
        self.lstm_model = None
        self.dynamic_classes = set()
        self.label_map = {}
        self.inverse_label_map = {}
        self.is_hybrid = CONFIG.get("use_lstm_for_dynamic", True)

    # ---------------------------------------------------------
    # Construção do modelo LSTM
    # ---------------------------------------------------------
    def _build_lstm(self, n_classes):
        lstm_units = CONFIG.get("lstm_units", 64)
        seq_len = CONFIG.get("sequence_length", 12)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=(seq_len, 67)),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(max(32, lstm_units // 2), activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # ---------------------------------------------------------
    # Treinamento
    # ---------------------------------------------------------
    def train(self, static_data, static_labels, dynamic_data, dynamic_labels):
        try:
            if static_labels:
                X_static = np.array(static_data)
                if X_static.ndim == 2 and X_static.shape[1] == 67:
                    self.rf_model.fit(X_static, static_labels)
                    print(f"[INFO] RF treinado com {len(static_labels)} amostras estáticas")
                else:
                    print(f"[ERRO] Shape inválido para RF: {X_static.shape}")

            if dynamic_labels and self.is_hybrid:
                X_dyn = np.array(dynamic_data)
                if X_dyn.ndim == 3 and X_dyn.shape[2] == 67:
                    unique = list(sorted(set(dynamic_labels)))
                    self.label_map = {label: i for i, label in enumerate(unique)}
                    self.inverse_label_map = {i: label for label, i in self.label_map.items()}
                    self.dynamic_classes = set(unique)
                    y = np.array([self.label_map[l] for l in dynamic_labels])

                    if len(unique) == 1:
                        print("[AVISO] Apenas uma letra dinâmica treinada — o softmax sempre retornará 1.")
                    if self.lstm_model is None or self.lstm_model.output_shape[-1] != len(unique):
                        self.lstm_model = self._build_lstm(len(unique))

                    self.lstm_model.fit(X_dyn, y, epochs=15, batch_size=16, verbose=0)
                    print(f"[INFO] LSTM treinado com {len(dynamic_labels)} sequências dinâmicas")
                else:
                    print(f"[ERRO] Shape inválido para LSTM: {X_dyn.shape}")

            self.trained = True
            return True

        except Exception as e:
            logging.error(f"[ModelManager] Erro no treino: {e}")
            print(f"[ERRO] {e}")
            return False

    # ---------------------------------------------------------
    # Predição híbrida com debug prints
    # ---------------------------------------------------------
    def predict(self, data):
        if not self.trained:
            return None, 0.0

        try:
            is_sequence = (
                isinstance(data, (list, tuple, np.ndarray))
                and len(data) == CONFIG.get("sequence_length", 12)
                and isinstance(data[0], (list, tuple, np.ndarray))
            )

            # --- Heurística de movimento ---
            use_lstm = False
            if is_sequence and self.is_hybrid and self.lstm_model is not None and len(self.dynamic_classes) > 0:
                arr = np.array(data)
                motion_var = np.var(arr[:-1] - arr[1:])
                if motion_var > 0.002:  # movimento significativo
                    use_lstm = True

            # === LSTM (gestos dinâmicos) ===
            if use_lstm:
                X = np.array([data])
                if X.ndim == 3 and X.shape[2] == 67:
                    probs = self.lstm_model.predict(X, verbose=0)[0]
                    idx = int(np.argmax(probs))
                    prob = float(probs[idx])
                    label = self.inverse_label_map.get(idx)
                    if label is not None and prob >= CONFIG.get("confidence_threshold", 0.7):
                        print(f"[DEBUG] usando LSTM ({label}) — prob={prob:.2f}")
                        return label, prob
                    else:
                        print(f"[DEBUG] LSTM sem confiança suficiente ({prob:.2f})")
                return None, 0.0

            # === RF (gestos estáticos) ===
            frame = data[-1] if is_sequence else data
            if isinstance(frame, (list, tuple, np.ndarray)) and len(frame) == 67:
                Xf = np.array([frame])
                pred = self.rf_model.predict(Xf)[0]
                try:
                    prob = float(self.rf_model.predict_proba(Xf)[0].max())
                except Exception:
                    prob = 1.0
                if prob >= CONFIG.get("confidence_threshold", 0.7):
                    print(f"[DEBUG] usando RF ({pred}) — prob={prob:.2f}")
                    return pred, prob
                else:
                    print(f"[DEBUG] RF sem confiança suficiente ({prob:.2f})")
            return None, 0.0

        except Exception as e:
            logging.error(f"[ModelManager] Erro na predição: {e}")
            print(f"[ERRO] {e}")
            return None, 0.0
