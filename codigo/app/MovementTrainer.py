import numpy as np
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
from Utils import extract_landmarks


class MovementTrainer:
    def __init__(self, db):
        self.db = db
        self.model = KNeighborsClassifier(n_neighbors=5)
        self.trained = False
        self.labels, self.data, _ = self.db.load_movements()
        if self.labels:
            self.train(self.data, self.labels)

    def train(self, data, labels):
        if len(data) == 0:
            logging.warning("Nenhum dado disponível para treino.")
            return False
            
        unique_labels = set(labels)
        if len(unique_labels) < 1:
            logging.warning("Nenhum dado disponível para treino.")
            return False
        if len(unique_labels) == 1:
            logging.warning(
                f"Apenas uma classe ({list(unique_labels)[0]}) disponível. Modelo treinado, mas previsões podem não ser confiáveis."
            )
        self.model.fit(data, labels)
        self.trained = True
        logging.info(
            f"Modelo de movimento treinado com {len(data)} amostras e {len(unique_labels)} classes."
        )
        return True

    def predict(self, landmarks):
        if not self.trained:
            return None, 0.0
        try:
            pred = self.model.predict([landmarks])[0]
            prob = self.model.predict_proba([landmarks]).max()
            return pred, prob
        except NotFittedError:
            logging.error("Modelo de movimento não está treinado.")
            return None, 0.0
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return None, 0.0

    def save_movement(self, name, data):
        if not data:
            logging.error("Nenhum dado para salvar.")
            return
            
        clean_data = []
        for d in data:
            if d is not None:
                clean_data.append(d.tolist() if hasattr(d, "tolist") else d)
                
        if not clean_data:
            logging.error("Nenhum dado válido para salvar.")
            return
            
        # Carrega dados existentes
        existing_labels, existing_data, _ = self.db.load_movements()
        
        # Adiciona novos dados
        updated_labels = existing_labels + [name] * len(clean_data)
        updated_data = existing_data + clean_data
        
        # Salva no banco
        self.db.save_movements(updated_labels, updated_data)
        
        # Atualiza e retreina o modelo
        self.labels = updated_labels
        self.data = updated_data
        self.train(self.data, self.labels)
        
        logging.info(f"Movimento '{name}' salvo com {len(clean_data)} amostras.")

    def extract_landmarks(self, hand_landmarks):
        return extract_landmarks(hand_landmarks)