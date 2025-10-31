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
        # use valores do CONFIG
        self.rf_model = RandomForestClassifier(n_estimators=CONFIG.get("rf_estimators", 100),
                                               random_state=42)
        self.lstm_model = None
        self.dynamic_classes = set()
        self.label_map = {}             # label -> idx
        self.inverse_label_map = {}     # idx -> label
        self.is_hybrid = CONFIG.get("use_lstm_for_dynamic", True)

    def _build_lstm(self, n_classes):
        # usa parâmetros do CONFIG quando existirem
        lstm_units = CONFIG.get("lstm_units", 64)
        seq_len = CONFIG.get("sequence_length", 12)
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, input_shape=(seq_len, 67)),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(max(32, lstm_units//2), activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, static_data, static_labels, dynamic_data, dynamic_labels):
        """
        static_data: list of frames (n_samples, 67)
        static_labels: list of labels (n_samples,)
        dynamic_data: list of sequences (n_seq, seq_len, 67)
        dynamic_labels: list of labels (n_seq,)
        """
        try:
            # TREINO RF para estáticas (se existirem)
            if static_labels and len(static_labels) > 0:
                X_static = np.array(static_data)
                if X_static.ndim != 2 or X_static.shape[1] != 67:
                    logging.error(f"[ModelManager] Estáticas com shape inválido: {X_static.shape}, esperado (n,67)")
                    print(f"[ERRO] Estáticas: {X_static.shape} features, esperado 67")
                else:
                    self.rf_model.fit(X_static, static_labels)
                    logging.info(f"[ModelManager] RF treinado: {X_static.shape}")
                    print(f"[INFO] RF treinado com {X_static.shape[0]} amostras estáticas")

            # TREINO LSTM para dinâmicas (se for modo híbrido e houver dados suficientes)
            if dynamic_labels and len(dynamic_labels) > 0 and self.is_hybrid:
                # garantir formato numpy (n_seq, seq_len, 67)
                X_dyn = np.array(dynamic_data)
                # valida shapes
                if X_dyn.ndim != 3 or X_dyn.shape[2] != 67:
                    logging.error(f"[ModelManager] Dinâmicas com shape inválido: {X_dyn.shape}, esperado (n,seq_len,67)")
                    print(f"[ERRO] Dinâmicas: {X_dyn.shape} features, esperado (n,{CONFIG['sequence_length']},67)")
                else:
                    # verifica quantidade mínima por classe
                    # dynamic_labels é lista de labels repetidos por amostra
                    unique, counts = np.unique(dynamic_labels, return_counts=True)
                    min_req = CONFIG.get("min_samples_per_class", 120)
                    insufficient = [u for u, c in zip(unique, counts) if c < min_req]
                    if insufficient:
                        logging.info(f"[ModelManager] Algumas classes dinâmicas têm menos que {min_req} amostras: {insufficient}")
                        print(f"[AVISO] Classes dinâmicas com menos que {min_req} amostras: {insufficient} (LSTM não será treinada até atingir mínimo)")
                    else:
                        # preparar label_map
                        unique_list = list(unique)
                        self.label_map = {label: i for i, label in enumerate(unique_list)}
                        self.inverse_label_map = {i: label for label, i in self.label_map.items()}
                        self.dynamic_classes = set(unique_list)

                        y = np.array([self.label_map[l] for l in dynamic_labels])

                        # (re)criar modelo se necessário
                        if self.lstm_model is None or self.lstm_model.output_shape[-1] != len(unique_list):
                            self.lstm_model = self._build_lstm(len(unique_list))

                        # treino LSTM
                        epochs = 15
                        batch_size = 16
                        self.lstm_model.fit(X_dyn, y, epochs=epochs, batch_size=batch_size, verbose=0)
                        logging.info(f"[ModelManager] LSTM treinado: {X_dyn.shape}")
                        print(f"[INFO] LSTM treinado com {X_dyn.shape[0]} sequências dinâmicas")

            # marcar como treinado se ao menos um modelo foi treinado
            self.trained = True
            return True
        except Exception as e:
            logging.error(f"[ModelManager] Erro no treino: {e}")
            return False

            
    def predict(self, data):
        """
        data: ou um único frame (lista/ndarray de 67 features)
              ou uma sequência (lista de frames) com len == sequence_length
        Retorna: (pred_label ou None, prob)
        """
        if not self.trained:
            return None, 0.0

        try:
            is_sequence = isinstance(data, (list, tuple)) and len(data) == CONFIG.get("sequence_length", 12) \
                          and isinstance(data[0], (list, tuple, np.ndarray))

            # === Heurística: decidir se é gesto dinâmico ou estático ===
            use_lstm = False
            if is_sequence and self.is_hybrid and self.lstm_model is not None and len(self.dynamic_classes) > 0:
                # calcular variância média entre frames (movimento da mão)
                arr = np.array(data)
                motion_var = np.var(arr[:-1] - arr[1:])
                # se a mão se moveu significativamente, usar LSTM
                if motion_var > 0.002:  # limiar empírico — ajustável
                    use_lstm = True

            # === Predição dinâmica (LSTM) ===
            if use_lstm:
                X = np.array([data])
                if X.ndim == 3 and X.shape[2] == 67:
                    probs = self.lstm_model.predict(X, verbose=0)[0]
                    idx = int(np.argmax(probs))
                    prob = float(probs[idx])
                    label = self.inverse_label_map.get(idx)
                    if label is not None and prob >= CONFIG.get("confidence_threshold", 0.7):
                        return label, prob
                return None, 0.0

            # === Predição estática (RF) ===
            frame = data[-1] if is_sequence else data
            if not (isinstance(frame, (list, tuple, np.ndarray)) and len(frame) == 67):
                return None, 0.0

            Xf = np.array([frame])
            pred = self.rf_model.predict(Xf)[0]
            try:
                prob = float(self.rf_model.predict_proba(Xf)[0].max())
            except Exception:
                prob = 1.0
            if prob >= CONFIG.get("confidence_threshold", 0.7):
                return pred, prob
            return None, 0.0

        except Exception as e:
            logging.error(f"[ModelManager] Erro na predição: {e}")
            return None, 0.0
