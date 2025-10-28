import logging
import cv2
from .BaseCore import BaseCore
from Model_manager import ModelManager
from Config import CONFIG

class GestureCore(BaseCore):
    def __init__(self, db):
        super().__init__(db)
        self.model_manager = ModelManager(CONFIG["knn_neighbors"])
        # CORREÇÃO APLICADA: Filtrar por tipo para evitar mistura de shapes
        self.labels, self.data, _ = self.db.load_gestures(gesture_type="letter")
        if self.labels:
            self.model_manager.train(self.data, self.labels)
        
        # Estados específicos de gestos
        self.new_gesture_name = ""
        self.new_gesture_data = []

    def start_training_mode(self):
        """Inicia o modo de treino"""
        self.mode = "treino"
        self.new_gesture_data = []
        self.samples_count = 0
        self.new_gesture_name = ""

    def cancel_training(self):
        """Cancela o modo de treino"""
        self.mode = "teste"
        self.new_gesture_name = ""
        self.new_gesture_data = []
        self.samples_count = 0

    def predict_gesture(self, landmarks):
        """Faz predição do gesto"""
        if self.mode == "teste" and self.labels:
            pred, prob = self.model_manager.predict(landmarks)
            if pred is not None and prob >= CONFIG["confidence_threshold"]:
                self.current_word += pred
                return f"Reconhecido: {pred} ({prob:.2f})"
        return None

    def add_training_sample(self, landmarks):
        """Adiciona amostra de treino"""
        if self.mode == "treino" and self.new_gesture_name:
            self.new_gesture_data.append(landmarks)
            self.samples_count = len(self.new_gesture_data)
            return True
        return False

    def save_gesture(self):
        """Salva o gesto treinado"""
        if not self.new_gesture_name or not self.new_gesture_data:
            return "Defina um nome e capture gestos primeiro!"

        clean_data = []
        for l in self.new_gesture_data:
            if l is not None:
                clean_data.append(l.tolist() if hasattr(l, "tolist") else l)
        
        if not clean_data:
            return "Nenhum dado válido para salvar!"

        try:
            # Carrega dados existentes e adiciona novos
            existing_labels, existing_data, _ = self.db.load_gestures()
            updated_labels = existing_labels + [self.new_gesture_name] * len(clean_data)
            updated_data = existing_data + clean_data
            
            self.db.save_gestures(updated_labels, updated_data)
            self.labels, self.data, _ = self.db.load_gestures()
            self.model_manager.train(self.data, self.labels)
            
            message = f"Gesto '{self.new_gesture_name}' salvo com {self.samples_count} amostras!"
            
            # Reseta estado de treino
            self.mode = "teste"
            self.new_gesture_name = ""
            self.new_gesture_data = []
            self.samples_count = 0
            
            return message
        except Exception as e:
            return f"Erro ao salvar gesto: {e}"

    def delete_gesture(self, gesture_name):
        """Deleta um gesto"""
        success = self.db.delete_gesture(gesture_name)
        if success:
            self.labels, self.data, _ = self.db.load_gestures()
            if self.labels:
                self.model_manager.train(self.data, self.labels)
            return f"Gesto '{gesture_name}' deletado com sucesso!"
        return f"Erro ao deletar gesto '{gesture_name}'."

    def change_resolution(self, width, height):
        """Altera a resolução da câmera"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verificar se a resolução foi aplicada
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"[INFO] Resolução alterada para: {actual_width}x{actual_height}")
            return True
        except Exception as e:
            print(f"[ERRO] Falha ao alterar resolução: {e}")
            return False