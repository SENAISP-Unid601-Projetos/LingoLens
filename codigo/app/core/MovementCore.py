from .BaseCore import BaseCore
from MovementTrainer import MovementTrainer
from Config import CONFIG
from Model_manager import ModelManager  # CORREÇÃO APLICADA: Import para modelo estático
from collections import deque
import numpy as np  # Para cálculos de displacement

class MovementCore(BaseCore):
    def __init__(self, db):
        super().__init__(db)
        self.trainer = MovementTrainer(db)
        
        # Carregue modelo estático (KNN) - CORREÇÃO APLICADA: Para reconhecimento misto
        self.static_model = ModelManager(CONFIG["knn_neighbors"])
        self.static_labels, self.static_data, _ = self.db.load_gestures(gesture_type="letter")
        if self.static_labels:
            self.static_model.train(self.static_data, self.static_labels)
        
        # Estados específicos para Libras
        self.new_movement_name = ""
        self.is_recording = False
        self.training_samples = 0
        self.samples_count = 0
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        
        # Estados para detecção de movimento - CORREÇÃO APLICADA: Para diferenciar estático/dinâmico
        self.prev_landmarks = None
        self.movement_threshold = CONFIG["libras_movement_threshold"]
        self.sequence_buffer = deque(maxlen=CONFIG["libras_sequence_length"])
        self.is_dynamic = False

    def process_frame(self, frame):
        image, landmarks_list = super().process_frame(frame)
        
        if landmarks_list is None:
            landmarks_list = []
        
        if not landmarks_list:
            self.current_prediction = ""
            self.prediction_confidence = 0.0
            self.current_word = ""  # CORREÇÃO APLICADA: Limpa se nada detectado
            return image, landmarks_list
        
        # Extrai features do frame atual
        current_features = self.trainer.extract_libras_features(landmarks_list)
        if current_features is None:
            return image, landmarks_list
        
        # Detecção de movimento (comparar com frame anterior) - CORREÇÃO APLICADA
        displacement = 0.0
        if self.prev_landmarks is not None:
            displacement = np.sum(np.abs(np.array(current_features) - np.array(self.prev_landmarks)))
            displacement /= len(current_features) if len(current_features) > 0 else 1
        
        self.prev_landmarks = current_features
        
        # Modo treino - coletar amostras
        if self.mode == "treino" and self.is_recording:
            if landmarks_list and len(landmarks_list) > 0:
                success = self.trainer.add_training_sample(landmarks_list, self.new_movement_name)
                if success:
                    self.training_samples += 1
                    self.samples_count = self.training_samples
        
        # Modo teste - fazer predição com lógica mista - CORREÇÃO APLICADA
        elif self.mode == "teste":
            if displacement < self.movement_threshold:
                # Estático: Use modelo KNN
                self.is_dynamic = False
                if self.static_model.trained:
                    pred, prob = self.static_model.predict(np.array(current_features).flatten())  # Ajuste shape
                    if pred and prob > CONFIG["confidence_threshold"]:
                        self.current_prediction = pred
                        self.prediction_confidence = prob
                        self.current_word = pred
                    else:
                        self.current_word = ""
            else:
                # Dinâmico: Colete sequência
                self.is_dynamic = True
                self.sequence_buffer.append(current_features)
                
                if len(self.sequence_buffer) == CONFIG["libras_sequence_length"]:
                    sequence_features = np.concatenate(list(self.sequence_buffer)).flatten()
                    pred, confidence = self.trainer.predict(sequence_features)
                    if pred and confidence > CONFIG["libras_confidence_threshold"]:
                        self.current_prediction = pred
                        self.prediction_confidence = confidence
                        self.current_word = pred
                    self.sequence_buffer.clear()
            # CORREÇÃO APLICADA: Limpeza quando não detecta
            if not self.current_prediction:
                self.current_word = ""

        return image, landmarks_list

    def predict_movement(self, landmarks_list):
        if self.mode == "teste":
            if landmarks_list is None or len(landmarks_list) == 0:
                return None
                
            pred, confidence = self.trainer.predict(landmarks_list)
            if pred and confidence > CONFIG["libras_confidence_threshold"]:
                self.current_word = pred
                return f"👋 Sinal: {pred} ({confidence:.2f})"
        return None

    def delete_movement(self, movement_name):
        """Deleta um movimento do banco de dados"""
        if not movement_name:
            return "❌ Nome do movimento não fornecido"
            
        # CORREÇÃO APLICADA: Mudar para _load_from_database
        self.trainer._load_from_database()
        
        success, message = self.trainer.delete_sign(movement_name)
        return message

    def start_training(self, movement_name):
        self.new_movement_name = movement_name
        self.mode = "treino"
        self.trainer.start_training_session(movement_name)
        self.training_samples = 0
        self.samples_count = 0
        self.is_recording = False

    def start_recording(self):
        if self.mode == "treino" and self.new_movement_name:
            self.is_recording = True
            return True
        return False

    def stop_recording(self):
        self.is_recording = False

    def save_movement(self):
        if self.new_movement_name and self.training_samples > 0:
            success, message = self.trainer.save_training_session()
            
            if success:
                self.mode = "teste"
                self.new_movement_name = ""
                self.training_samples = 0
                self.samples_count = 0
                self.is_recording = False
                self.current_prediction = ""
                self.prediction_confidence = 0.0
            
            return message
        return "❌ Nenhum dado para salvar. Grave amostras primeiro."

    def cancel_training(self):
        self.mode = "teste"
        self.new_movement_name = ""
        self.training_samples = 0
        self.samples_count = 0
        self.is_recording = False
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.trainer.reset_training()

    def get_training_progress(self):
        return self.trainer.get_training_progress()

    def get_available_signs(self):
        return self.trainer.get_available_signs()

    def get_model_info(self):
        return self.trainer.get_model_info()

    def start_training_mode(self):
        self.mode = "treino"
        self.new_movement_name = ""
        self.training_samples = 0
        self.samples_count = 0
        self.is_recording = False