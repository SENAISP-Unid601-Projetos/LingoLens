import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError

class ModelManager:
    def __init__(self, neighbors):
        self.model = KNeighborsClassifier(n_neighbors=neighbors)
        self.trained = False

    def train(self, data, labels):
        unique_labels = set(labels)
        if len(unique_labels) < 1:
            logging.warning("Nenhum dado disponível para treino.")
            return False

        if len(unique_labels) == 1:
            logging.warning(
                f"Apenas uma classe ({list(unique_labels)[0]}) disponível. Previsões podem não ser confiáveis."
            )

        self.model.fit(data, labels)
        self.trained = True
        logging.info(f"Modelo treinado com {len(data)} amostras e {len(unique_labels)} classes.")
        return True

    def predict(self, landmarks):
        if not self.trained:
            logging.warning("Modelo ainda não treinado.")
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
            return None, 0.0
