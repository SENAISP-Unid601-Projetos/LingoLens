import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

class ModelManager:
    def __init__(self, n_estimators):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.trained = False

    def train(self, data, labels):
        unique_labels = set(labels)
        if len(unique_labels) < 1:
            logging.warning("Nenhum dado disponível para treino.")
            return False

        if len(unique_labels) == 1:
            logging.warning(
                f"Apenas uma classe ({list(unique_labels)[0]}) disponível. Modelo treinado, mas previsões podem não ser confiáveis."
            )

        # Validação cruzada
        scores = cross_val_score(self.model, data, labels, cv=5, scoring='accuracy')
        logging.info(f"Precisão média CV: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

        self.model.fit(data, labels)
        self.trained = True
        logging.info(f"Modelo treinado com {len(data)} amostras e {len(unique_labels)} classes.")
        return True

    def predict(self, landmarks):
        if not self.trained:
            logging.warning("Modelo ainda não treinado. Retornando None.")
            return None, 0.0

        try:
            prediction = self.model.predict([landmarks])[0]
            probability = self.model.predict_proba([landmarks]).max()
            logging.debug(f"Predição: {prediction} | Probabilidade: {probability:.2f}")
            return prediction, probability
        except NotFittedError:
            logging.error("Erro: modelo não está ajustado corretamente.")
            return None, 0.0
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return None, 0.0s