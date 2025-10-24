from .BaseCore import BaseCore
from MovementTrainer import MovementTrainer

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
        """Processa frame para detecção de sinais de Libras"""
        image, landmarks_list = super().process_frame(frame)
        
        # Modo treino - coletar amostras
        if self.mode == "treino" and self.is_recording and landmarks_list:
            success = self.trainer.add_training_sample(landmarks_list, self.new_movement_name)
            if success:
                self.training_samples += 1
                self.samples_count = self.training_samples
        
        # Modo teste - fazer predição
        elif self.mode == "teste" and landmarks_list:
            prediction, confidence = self.trainer.predict(landmarks_list)
            if prediction and confidence > 0.6:
                self.current_prediction = prediction
                self.prediction_confidence = confidence
                self.current_word = prediction
        
        return image, landmarks_list

    def predict_movement(self, landmarks_list):
        """Faz predição de sinal de Libras"""
        if self.mode == "teste":
            pred, confidence = self.trainer.predict(landmarks_list)
            if pred and confidence > 0.6:
                self.current_word = pred
                return f"Sinal: {pred} ({confidence:.2f})"
        return None

    def start_training(self, movement_name):
        """Inicia treino de um novo sinal"""
        self.new_movement_name = movement_name
        self.mode = "treino"
        self.trainer.start_training_session(movement_name)
        self.training_samples = 0
        self.samples_count = 0
        self.is_recording = False

    def start_recording(self):
        """Inicia a gravação de amostras"""
        if self.mode == "treino" and self.new_movement_name:
            self.is_recording = True
            return True
        return False

    def stop_recording(self):
        """Para a gravação de amostras"""
        self.is_recording = False

    def save_movement(self):
        """Salva o sinal treinado"""
        if self.new_movement_name and self.training_samples > 0:
            success, message = self.trainer.save_training_session()
            
            if success:
                # Resetar estado
                self.mode = "teste"
                self.new_movement_name = ""
                self.training_samples = 0
                self.samples_count = 0
                self.is_recording = False
            
            return message
        return "Nenhum dado para salvar"

    def cancel_training(self):
        """Cancela o treino atual"""
        self.mode = "teste"
        self.new_movement_name = ""
        self.training_samples = 0
        self.samples_count = 0
        self.is_recording = False
        self.trainer.reset_training()

    def get_training_progress(self):
        return self.trainer.get_training_progress()

    def get_available_signs(self):
        return self.trainer.get_available_signs()

    def get_model_info(self):
        return self.trainer.get_model_info()

    def start_training_mode(self):
        """Inicia modo de treino"""
        self.mode = "treino"
        self.new_movement_name = ""
        self.training_samples = 0
        self.samples_count = 0
        self.is_recording = False