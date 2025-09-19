import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class ModelManager:
    def __init__(self, neighbors):
        self.model = KNeighborsClassifier(n_neighbors=neighbors)

    def train(self, data, labels):
        if len(set(labels)) > 1:
            self.model.fit(data, labels)
            logging.info("Modelo treinado")
            return True
        logging.warning("Dados insuficientes para treino")
        return False

    def predict(self, landmarks):
        prediction = self.model.predict([landmarks])[0]
        probability = self.model.predict_proba([landmarks]).max()
        return prediction, probability
