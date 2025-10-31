import logging
import cv2
from .BaseCore import BaseCore
from Model_manager import ModelManager
from Config import CONFIG
import numpy

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

        if self.labels:
            print(f"🔍 DEBUG: {len(self.labels)} labels, {len(self.data)} amostras")
    
            # Verificar tamanhos das amostras
            sizes = []
            for i, sample in enumerate(self.data):
                try:
                    sample_array = np.array(sample)
                    sizes.append(sample_array.shape)
                except:
                    sizes.append("ERRO")
    
            print(f"🔍 Tamanhos das amostras: {set(sizes)}")
    
            self.model_manager.train(self.data, self.labels)

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
            return "❌ Defina um nome e capture gestos primeiro!"

        clean_data = []
        for l in self.new_gesture_data:
            if l is not None:
                try:
                    # 🔥 CORREÇÃO: Garantir formato consistente
                    l_array = np.array(l, dtype=np.float64).flatten()
                    if l_array.size > 0:  # Só adicionar se não estiver vazio
                        clean_data.append(l_array.tolist())  # Salvar como lista
                except:
                    continue

        if not clean_data:
            return "❌ Nenhum dado válido para salvar!"

        try:
            # Carrega dados existentes
            existing_labels, existing_data, _ = self.db.load_gestures(gesture_type="letter")

            # 🔥 CORREÇÃO: Validar dados existentes também
            valid_existing_data = []
            valid_existing_labels = []
        
            for sample, label in zip(existing_data, existing_labels):
                try:
                    sample_array = np.array(sample, dtype=np.float64).flatten()
                    if sample_array.size > 0:
                        valid_existing_data.append(sample_array.tolist())
                        valid_existing_labels.append(label)
                except:
                    continue

            # Combinar dados
            updated_labels = valid_existing_labels + [self.new_gesture_name] * len(clean_data)
            updated_data = valid_existing_data + clean_data
        
            # 🔥 VALIDAÇÃO FINAL
            print(f"📊 Salvando: {len(updated_labels)} labels, {len(updated_data)} amostras")
        
            self.db.save_gestures(updated_labels, updated_data)
        
            # Recarregar e treinar
            self.labels, self.data, _ = self.db.load_gestures(gesture_type="letter")
        
            if self.labels and self.data:
                # 🔥 VALIDAR ANTES DE TREINAR
                valid_data, valid_labels = self._validate_and_prepare_data(self.data, self.labels)
                if valid_data and valid_labels:
                    self.model_manager.train(valid_data, valid_labels)
                else:
                    return "❌ Dados inválidos após validação"
        
            message = f"✅ Gesto '{self.new_gesture_name}' salvo com {self.samples_count} amostras!"
        
           # Resetar estado
            self.mode = "teste"
            self.new_gesture_name = ""
            self.new_gesture_data = []
            self.samples_count = 0

            return message
        
        except Exception as e:
            return f"❌ Erro ao salvar gesto: {e}"

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
        
    def _validate_and_prepare_data(self, data, labels):
        """Valida e prepara dados para treinamento"""
        import numpy as np
    
        valid_data = []
        valid_labels = []
    
        for i, (sample, label) in enumerate(zip(data, labels)):
            try:
                # Converter para array numpy
                sample_array = np.array(sample, dtype=np.float64)

                # Verificar dimensões
                if sample_array.ndim == 0 or sample_array.size == 0:
                    continue

                # Achatar para 1D se necessário
                if sample_array.ndim > 1:
                    sample_array = sample_array.flatten()

                valid_data.append(sample_array)
                valid_labels.append(label)

            except Exception as e:
                print(f"⚠️ Amostra {i} inválida: {e}")
                continue
    
        return valid_data, valid_labels