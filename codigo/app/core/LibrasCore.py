import logging
import cv2
import numpy as np
import time
from .BaseCore import BaseCore
from Model_manager import ModelManager
from Config import CONFIG
from .MovementDetector import MovementDetector

class LibrasCore(BaseCore):
    def __init__(self, db):
        super().__init__(db)
        
        # 🔥 MÚLTIPLOS MODELOS: Letras, Palavras e Sinais complexos
        self.letter_model = ModelManager(CONFIG["knn_neighbors"])
        self.word_model = ModelManager(CONFIG["knn_neighbors"])
        self.sentence_model = ModelManager(CONFIG["knn_neighbors"])
        
        # Detector de movimento
        self.movement_detector = MovementDetector(
            sequence_length=CONFIG["movement_sequence_length"],
            threshold=CONFIG["movement_detection_threshold"]
        )
        
        # 🔥 ESTADOS PARA TRADUÇÃO COMPLETA
        self.recognition_mode = CONFIG["current_recognition_mode"]  # letras/palavras/frases
        
        # Controle de tempo para palavras/frases
        self.last_detection_time = 0
        self.current_sentence = []
        self.word_buffer = ""
        
        # Estados de detecção
        self.awaiting_movement = False
        self.movement_buffer = []
        self.last_prediction = None
        self.last_confidence = 0.0
        
        # Carregar modelos treinados
        self._load_models()
        
        # Estados de treino
        self.new_gesture_name = ""
        self.new_gesture_data = []
        self.gesture_type = "letter"  # letter/word/sentence

    def _load_models(self):
        """Carrega os diferentes modelos do banco"""
        # Modelo de letras
        letter_labels, letter_data, _ = self.db.load_gestures(gesture_type="letter")
        if letter_labels:
            self.letter_model.train(letter_data, letter_labels)
            logging.info(f"✅ Modelo de letras carregado: {len(letter_labels)} gestos")
        
        # Modelo de palavras
        word_labels, word_data, _ = self.db.load_gestures(gesture_type="word")
        if word_labels:
            self.word_model.train(word_data, word_labels)
            logging.info(f"✅ Modelo de palavras carregado: {len(word_labels)} gestos")
        
        # Modelo de frases (sinais complexos)
        sentence_labels, sentence_data, _ = self.db.load_gestures(gesture_type="sentence")
        if sentence_labels:
            self.sentence_model.train(sentence_data, sentence_labels)
            logging.info(f"✅ Modelo de frases carregado: {len(sentence_labels)} gestos")

    def process_frame(self, frame):
        """Processa frame para tradução de Libras"""
        image, landmarks_list = super().process_frame(frame)
        
        if landmarks_list and self.mode == "teste":
            current_landmarks = landmarks_list[0]
            is_moving = self.movement_detector.add_landmarks(current_landmarks)
            
            # 🔥 LÓGICA POR MODO DE RECONHECIMENTO
            if self.recognition_mode == "letras":
                result = self._process_letters(current_landmarks, is_moving)
            elif self.recognition_mode == "palavras":
                result = self._process_words(current_landmarks, is_moving)
            elif self.recognition_mode == "frases":
                result = self._process_sentences(current_landmarks, is_moving)
            
            if result:
                self._update_display_text(result)
            
            # Desenhar informações específicas do modo
            image = self._draw_mode_info(image, is_moving)
        
        return image, landmarks_list

    def _process_letters(self, landmarks, is_moving):
        """Processa reconhecimento de letras (soletração)"""
        if not is_moving and not self.awaiting_movement:
            # Tentar reconhecer letra estática
            pred, confidence = self.letter_model.predict(landmarks)
            
            if pred and confidence >= CONFIG["confidence_threshold"]:
                self.last_prediction = pred
                self.last_confidence = confidence
                
                if pred in CONFIG["static_letters"]:
                    return {"type": "letter", "value": pred, "confidence": confidence}
                elif pred in CONFIG["dynamic_letters"]:
                    self.awaiting_movement = True
                    return {"type": "prompt", "value": f"Faça movimento para: {pred}"}
        
        elif is_moving and self.awaiting_movement:
            self.movement_buffer.append(landmarks)
            
        elif not is_moving and self.awaiting_movement:
            if len(self.movement_buffer) >= CONFIG["min_movement_frames"]:
                # Confirmar letra dinâmica
                if self.last_prediction:
                    result = {"type": "letter", "value": self.last_prediction, "confidence": self.last_confidence}
                    self._reset_detection()
                    return result
            
        return None

    def _process_words(self, landmarks, is_moving):
        """Processa reconhecimento de palavras completas"""
        current_time = time.time()
        
        if not is_moving:
            # Tentar reconhecer palavra estática
            pred, confidence = self.word_model.predict(landmarks)
            
            if pred and confidence >= CONFIG["min_word_confidence"]:
                # Verificar se é uma nova palavra (evitar repetições)
                if (current_time - self.last_detection_time > CONFIG["word_detection_timeout"] or 
                    pred != self.last_prediction):
                    
                    self.last_detection_time = current_time
                    self.last_prediction = pred
                    
                    return {"type": "word", "value": pred, "confidence": confidence}
        
        return None

    def _process_sentences(self, landmarks, is_moving):
        """Processa reconhecimento de frases/sinais complexos"""
        current_time = time.time()
        
        if is_moving and not self.awaiting_movement:
            # Iniciar captura de sinal complexo
            self.awaiting_movement = True
            self.movement_buffer = [landmarks]
            self.last_detection_time = current_time
            
        elif is_moving and self.awaiting_movement:
            # Continuar captura
            self.movement_buffer.append(landmarks)
            
        elif not is_moving and self.awaiting_movement:
            # Finalizar captura e reconhecer
            if (current_time - self.last_detection_time > 1.0 and 
                len(self.movement_buffer) >= CONFIG["min_movement_frames"]):
                
                # Extrair features da sequência completa
                sequence_features = self._extract_sequence_features(self.movement_buffer)
                pred, confidence = self.sentence_model.predict(sequence_features)
                
                if pred and confidence >= CONFIG["min_word_confidence"]:
                    result = {"type": "sentence", "value": pred, "confidence": confidence}
                    self._reset_detection()
                    return result
        
        return None

    def _extract_sequence_features(self, sequence):
        """Extrai features de uma sequência de movimentos"""
        if not sequence:
            return None
            
        try:
            # Concatenar todos os landmarks da sequência
            features = []
            for landmarks in sequence:
                features.extend(landmarks.flatten() if hasattr(landmarks, 'flatten') else landmarks)
            
            return np.array(features)
        except Exception as e:
            logging.error(f"❌ Erro ao extrair features de sequência: {e}")
            return None

    def _update_display_text(self, result):
        """Atualiza o texto de exibição baseado no resultado"""
        if result["type"] == "letter":
            self.current_word += result["value"]
            
        elif result["type"] == "word":
            # Adicionar palavra com espaço
            if self.current_word and not self.current_word.endswith(" "):
                self.current_word += " "
            self.current_word += result["value"] + " "
            
        elif result["type"] == "sentence":
            # Adicionar frase completa
            if self.current_word:
                self.current_word += " "
            self.current_word += result["value"] + ". "

    def _draw_mode_info(self, image, is_moving):
        """Desenha informações do modo atual"""
        height, width = image.shape[:2]
        scale_factor = width / 640.0
        
        # Modo de reconhecimento
        mode_color = (0, 255, 255) if self.recognition_mode == "letras" else \
                    (255, 255, 0) if self.recognition_mode == "palavras" else \
                    (0, 255, 0)
        
        cv2.putText(
            image,
            f"Modo: {self.recognition_mode.upper()}",
            (int(width - 200 * scale_factor), int(35 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6 * scale_factor,
            mode_color,
            2,
            cv2.LINE_AA,
        )
        
        # Estado de movimento
        movement_color = (0, 255, 0) if not is_moving else (0, 0, 255)
        movement_text = "ESTÁTICO" if not is_moving else "EM MOVIMENTO"
        
        cv2.putText(
            image,
            movement_text,
            (int(width - 200 * scale_factor), int(65 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale_factor,
            movement_color,
            2,
            cv2.LINE_AA,
        )
        
        # Instrução específica do modo
        if self.awaiting_movement:
            instruction = "Continue o movimento..." if is_moving else "Finalizando movimento..."
            cv2.putText(
                image,
                instruction,
                (int(10 * scale_factor), int(height - 100 * scale_factor)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5 * scale_factor,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
        
        return image

    def _reset_detection(self):
        """Reseta o estado de detecção"""
        self.awaiting_movement = False
        self.movement_buffer = []
        self.last_prediction = None

    # 🔥 NOVOS MÉTODOS PARA CONTROLE DE MODO
    def switch_recognition_mode(self, new_mode):
        """Alterna entre modos de reconhecimento"""
        if new_mode in CONFIG["recognition_modes"]:
            self.recognition_mode = new_mode
            self._reset_detection()
            self.last_detection_time = 0
            logging.info(f"🔁 Modo alterado para: {new_mode}")
            return True
        return False

    def clear_text(self):
        """Limpa o texto atual"""
        self.current_word = ""
        self.current_sentence = []
        self.word_buffer = ""

    # 🔥 MÉTODOS DE TREINO ESPECÍFICOS PARA LIBRAS
    def start_training_mode(self, gesture_type="letter"):
        """Inicia modo de treino para tipo específico"""
        self.mode = "treino"
        self.gesture_type = gesture_type
        self.new_gesture_data = []
        self.samples_count = 0
        self.new_gesture_name = ""
        self._reset_detection()

    def add_training_sample(self, landmarks):
        """Adiciona amostra de treino considerando o tipo"""
        if self.mode == "treino" and self.new_gesture_name:
            # Para sinais dinâmicos, capturar sequência
            if self.gesture_type == "sentence" and self.awaiting_movement:
                self.movement_buffer.append(landmarks)
                return True
            else:
                # Para gestos estáticos
                self.new_gesture_data.append(landmarks)
                self.samples_count = len(self.new_gesture_data)
                return True
        return False

    def save_gesture(self):
        """Salva gesto no banco com tipo específico"""
        if not self.new_gesture_name or not self.new_gesture_data:
            return "❌ Nome e dados necessários!"

        try:
            # Para sinais dinâmicos, usar a sequência completa
            if self.gesture_type == "sentence" and self.movement_buffer:
                sequence_features = self._extract_sequence_features(self.movement_buffer)
                if sequence_features is not None:
                    self.new_gesture_data = [sequence_features.tolist()]

            clean_data = []
            for sample in self.new_gesture_data:
                if sample is not None:
                    clean_data.append(sample.tolist() if hasattr(sample, 'tolist') else sample)

            # Salvar no banco com tipo específico
            existing_labels, existing_data, _ = self.db.load_gestures(gesture_type=self.gesture_type)
            
            updated_labels = existing_labels + [self.new_gesture_name] * len(clean_data)
            updated_data = existing_data + clean_data
            
            success = self.db.save_gestures(updated_labels, updated_data, [self.gesture_type] * len(updated_labels))
            
            if success:
                # Recarregar modelo específico
                self._load_models()
                
                message = f"✅ {self.gesture_type.capitalize()} '{self.new_gesture_name}' salvo!"
                self._reset_training()
                return message
            else:
                return "❌ Erro ao salvar no banco"
                
        except Exception as e:
            return f"❌ Erro ao salvar: {e}"

    def _reset_training(self):
        """Reseta estado de treino"""
        self.mode = "teste"
        self.new_gesture_name = ""
        self.new_gesture_data = []
        self.samples_count = 0
        self.gesture_type = "letter"
        self._reset_detection()