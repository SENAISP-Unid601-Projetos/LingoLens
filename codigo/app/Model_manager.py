import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import os
from Config import CONFIG, get_logger

logger = get_logger("Model")

class ModelManager:
    def __init__(self, gesture_type="letter"):
        self.gesture_type = gesture_type
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1
        )
        self.trained = False
        self.accuracy = 0.0

    def train(self, data, labels):
        if len(data) == 0 or len(labels) == 0:
            logger.warning("Nenhum dado para treinar.")
            return False
        try:
            X = np.array(data)
            y = np.array(labels)
            unique_classes = len(set(y))
            
            # CORREÇÃO AQUI
            if unique_classes < 2:
                logger.info(f"ESTÁTICO: RF | {len(data)} amostras | 1 classe | Sem validação cruzada")
                self.model.fit(X, y)
                self.accuracy = 1.0
            else:
                cv_value = min(5, unique_classes)
                scores = cross_val_score(self.model, X, y, cv=cv_value, scoring='accuracy')
                self.accuracy = scores.mean()
                self.model.fit(X, y)
                logger.info(f"ESTÁTICO: RF | {len(data)} amostras | {unique_classes} classes | Treino: {self.accuracy:.3f}")
            
            self.trained = True
            return True
        except Exception as e:
            logger.error(f"Falha no treino estático: {e}")
            return False

    def predict(self, landmarks):
        if not self.trained:
            return None, 0.0
        try:
            X = np.array(landmarks).reshape(1, -1)
            prob = self.model.predict_proba(X)[0]
            pred_idx = np.argmax(prob)
            confidence = prob[pred_idx]
            prediction = self.model.classes_[pred_idx] if confidence >= CONFIG["confidence_threshold"] else None
            return prediction, confidence
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return None, 0.0

    def save_model(self):
        if not self.trained:
            return False
        try:
            model_path = os.path.join(CONFIG["model_dir"], f"rf_model_{self.gesture_type}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'accuracy': self.accuracy,
                    'classes': self.model.classes_
                }, f)
            logger.info(f"Modelo estático salvo: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            return False

    def load_model(self):
        model_path = os.path.join(CONFIG["model_dir"], f"rf_model_{self.gesture_type}.pkl")
        if not os.path.exists(model_path):
            return False
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.accuracy = data['accuracy']
                self.trained = True
            logger.info(f"Modelo estático carregado: {len(self.model.classes_)} classes")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return False