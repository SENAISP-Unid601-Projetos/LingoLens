import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import logging
from Database_manager import DatabaseManager
from Config import CONFIG

class ModelManager:
    def __init__(self, gesture_type="letter"):
        self.gesture_type = gesture_type
        self.db = DatabaseManager(CONFIG["db_path"])
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        )
        self.trained = False
        log_path = CONFIG["log_file"].replace("app.log", "model_static.log")
        logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

    def train(self, X, y):
        try:
            if len(X) < 10:
                logging.warning("Menos de 10 amostras para treino")
                print("[AVISO] Colete pelo menos 10 amostras por letra")
                return False

            X_clean = [x for x in X if len(x) == 63]
            y_clean = [y[i] for i, x in enumerate(X) if len(x) == 63]
            if len(set(y_clean)) < 2:
                print("[AVISO] Precisa de pelo menos 2 letras diferentes")
                return False

            X_clean = np.array(X_clean)
            y_clean = np.array(y_clean)

            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean,
                test_size=0.25,
                random_state=42,
                stratify=y_clean
            )
            self.model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, self.model.predict(X_train))
            val_acc = accuracy_score(y_test, self.model.predict(X_test))
            self.trained = True

            self.db.save_static_model(self.model, val_acc)

            log_msg = f"ESTÁTICO: RF | {len(X_clean)} amostras | {len(set(y_clean))} classes | Treino: {train_acc:.3f} | Val: {val_acc:.3f}"
            logging.info(log_msg)
            print(f"[SUCESSO] {log_msg}")

            if val_acc < 0.7:
                print("[ALERTA] Acurácia baixa! Colete mais amostras variadas.")
            return True
        except Exception as e:
            logging.error(f"Erro treino estático: {e}")
            return False

    def predict(self, landmarks):
        if not self.trained or len(landmarks) != 63:
            return None, 0.0
        try:
            prob = self.model.predict_proba([landmarks])[0]
            conf = np.max(prob)
            if conf < CONFIG["confidence_threshold"]:
                return None, 0.0
            pred = self.model.classes_[np.argmax(prob)]
            return pred, conf
        except:
            return None, 0.0

    def load_model(self):
        try:
            model = self.db.load_static_model()
            if model:
                self.model = model
                self.trained = True
                logging.info("Modelo estático (RF) carregado do banco")
                print(f"[INFO] Modelo estático carregado com {len(model.classes_)} classes")
                return True
            return False
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
            return False