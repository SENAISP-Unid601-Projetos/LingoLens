from .BaseCore import BaseCore
from MovementTrainer import MovementTrainer
from Config import CONFIG

class MovementCore(BaseCore):
    def __init__(self, db):
        super().__init__(db)
        self.trainer = MovementTrainer(db)
        
        # Estados específicos para Libras
        self.new_movement_name = ""
        self.is_recording = False
        self.training_samples = 0
        self.current_prediction = ""
        self.prediction_confidence = 0.0

    def process_frame(self, frame):
        image, landmarks_list = super().process_frame(frame)
        
        if landmarks_list is None:
            landmarks_list = []
        
        # Modo treino - coletar amostras
        if self.mode == "treino" and self.is_recording:
            if landmarks_list and len(landmarks_list) > 0:
                success = self.trainer.add_training_sample(landmarks_list, self.new_movement_name)
                if success:
                    self.training_samples += 1
                    self.samples_count = self.training_samples
        
        # Modo teste - fazer predição
        elif self.mode == "teste":
            if landmarks_list and len(landmarks_list) > 0:
                prediction, confidence = self.trainer.predict(landmarks_list)
                if prediction and confidence > CONFIG["libras_confidence_threshold"]:
                    self.current_prediction = prediction
                    self.prediction_confidence = confidence
                    self.current_word = prediction
                else:
                    self.prediction_timer -= 1
                    if self.prediction_timer <= 0:
                        self.current_word = ""
                        self.current_prediction = ""
                        self.prediction_confidence = 0.0
        
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
            
        # 🔥 CORREÇÃO: Forçar sincronização antes de deletar
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