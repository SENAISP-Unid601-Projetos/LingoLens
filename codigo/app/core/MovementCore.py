from .BaseCore import BaseCore
from MovementTrainer import MovementTrainer
from Config import CONFIG
from Model_manager import ModelManager
from collections import deque
import numpy as np
import logging

class MovementCore(BaseCore):
    def __init__(self, db):
        super().__init__(db)
        self.trainer = MovementTrainer(db)
        
        # Carregue modelo estático (KNN)
        self.static_model = ModelManager(CONFIG["knn_neighbors"])
        self.static_labels, self.static_data, _ = self.db.load_gestures(gesture_type="letter")
        if self.static_labels:
            self.static_model.train(self.static_data, self.static_labels)
        
        # Estados para detecção de movimento
        self.prev_features = None
        self.movement_threshold = CONFIG.get("libras_movement_threshold", 0.1)  # Aumentado para evitar falsos dinâmicos
        self.sequence_buffer = deque(maxlen=CONFIG["libras_sequence_length"])
        self.is_dynamic = False
        self.dynamic_cooldown = 0
        
        # Estados específicos para Libras
        self.new_movement_name = ""
        self.is_recording = False
        self.training_samples = 0
        self.samples_count = 0
        self.current_prediction = ""
        self.prediction_confidence = 0.0

    def process_frame(self, frame):
        image, landmarks_list = super().process_frame(frame)
        
        if not landmarks_list:
            self._reset_prediction()
            return image, landmarks_list
        
        # Extrai features do frame atual
        current_features = self.trainer.extract_libras_features(landmarks_list)
        if current_features is None:
            self._reset_prediction()
            return image, landmarks_list
        
        # Detecção de movimento
        displacement = 0.0
        if self.prev_features is not None:
            if len(current_features) == len(self.prev_features):
                diff = np.abs(current_features - self.prev_features)
                displacement = np.mean(diff)
        
        self.prev_features = current_features
        
        # MODO TESTE: Lógica híbrida
        if self.mode == "teste":
            if displacement < self.movement_threshold and self.dynamic_cooldown == 0:
                # ESTÁTICO
                self.is_dynamic = False
                self.sequence_buffer.clear()
                self._predict_static(current_features)
            else:
                # DINÂMICO: Coleta sequência, só prediz quando cheia
                self.is_dynamic = True
                self.dynamic_cooldown = CONFIG["libras_sequence_length"] // 2
                self.sequence_buffer.append(current_features)
                
                if len(self.sequence_buffer) == CONFIG["libras_sequence_length"]:
                    self._predict_dynamic()
                    self.sequence_buffer.clear()  # Reseta após predição
            
            if self.dynamic_cooldown > 0:
                self.dynamic_cooldown -= 1
        
        # MODO TREINO
        elif self.mode == "treino" and self.is_recording:
            success = self.trainer.add_training_sample(landmarks_list, self.new_movement_name)
            if success:
                self.training_samples += 1
                self.samples_count = self.training_samples

        return image, landmarks_list

    def _predict_static(self, features):
        if not self.static_model.trained:
            return
        try:
            features = features[:63]  # Usa apenas primeira mão para estáticos, se necessário
            pred, prob = self.static_model.predict([features])
            if pred and prob >= CONFIG["confidence_threshold"]:
                self.current_prediction = pred
                self.prediction_confidence = prob
                self.current_word = pred
            else:
                self._reset_prediction()
        except Exception as e:
            logging.error(f"❌ Erro em predição estática: {e}")

    def _predict_dynamic(self):
        try:
            seq = np.concatenate(list(self.sequence_buffer)).flatten()
            pred, conf = self.trainer.predict(seq)
            if pred and conf >= CONFIG["libras_confidence_threshold"]:
                self.current_prediction = pred
                self.prediction_confidence = conf
                self.current_word = pred
            else:
                self._reset_prediction()
        except Exception as e:
            logging.error(f"❌ Erro em predição dinâmica: {e}")
            self._reset_prediction()

    def _reset_prediction(self):
        self.current_prediction = ""
        self.prediction_confidence = 0.0
        self.current_word = ""

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
        if not movement_name:
            return "❌ Nome do movimento não fornecido"
            
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