import numpy as np
import logging
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from Config import CONFIG

class MovementTrainer:
    def __init__(self, db):
        self.db = db
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Configurações para Libras
        self.sequence_length = 10
        self.confidence_threshold = 0.6
        
        # Buffer para sequência temporal
        self.sequence_buffer = deque(maxlen=self.sequence_length)
        
        # Estados
        self.training_data = []
        self.training_labels = []
        self.current_sign_name = ""
        
        # Carregar modelo se existir
        self.model_path = os.path.join(CONFIG["train_data_dir"], "libras_model.pkl")
        self._load_model()

    def _load_model(self):
        """Carrega modelo treinado se existir"""
        try:
            if os.path.exists(self.model_path):
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler'] 
                self.label_encoder = model_data['label_encoder']
                self.is_trained = True
                logging.info("Modelo de Libras carregado!")
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")

    def save_model(self):
        """Salva o modelo treinado"""
        try:
            os.makedirs(CONFIG["train_data_dir"], exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }
            joblib.dump(model_data, self.model_path)
            logging.info("Modelo de Libras salvo!")
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar modelo: {e}")
            return False

    def extract_libras_features(self, landmarks_list):
        """Extrai features específicas para sinais de Libras"""
        if not landmarks_list or len(landmarks_list) == 0:
            return None

        features = []
        all_landmarks = []
        
        # Combinar landmarks de todas as mãos
        for landmarks in landmarks_list:
            if landmarks is not None and len(landmarks) == 63:
                all_landmarks.extend(landmarks)
        
        if not all_landmarks:
            return None
            
        landmarks_array = np.array(all_landmarks)
        
        # 1. Estatísticas básicas
        features.extend([
            np.mean(landmarks_array),
            np.std(landmarks_array),
            np.min(landmarks_array),
            np.max(landmarks_array)
        ])
        
        # 2. Features de posição relativa (para 1 ou 2 mãos)
        if len(landmarks_list) >= 1:
            hand1_features = self._extract_single_hand_features(landmarks_list[0], "m1")
            features.extend(hand1_features)
            
        if len(landmarks_list) >= 2:
            hand2_features = self._extract_single_hand_features(landmarks_list[1], "m2")
            features.extend(hand2_features)
            
            # Features de interação entre mãos
            interaction_features = self._extract_interaction_features(
                landmarks_list[0], landmarks_list[1]
            )
            features.extend(interaction_features)
        
        return np.array(features)

    def _extract_single_hand_features(self, landmarks, hand_prefix):
        """Extrai features de uma única mão"""
        features = []
        landmarks = np.array(landmarks).reshape(-1, 3)
        
        # Pontas dos dedos
        finger_tips = [4, 8, 12, 16, 20]
        palm_base = landmarks[0]  # Pulso
        
        # Distâncias das pontas dos dedos ao pulso
        for tip_idx in finger_tips:
            distance = np.linalg.norm(landmarks[tip_idx] - palm_base)
            features.append(distance)
        
        # Distâncias entre dedos adjacentes
        for i in range(len(finger_tips)-1):
            dist = np.linalg.norm(landmarks[finger_tips[i]] - landmarks[finger_tips[i+1]])
            features.append(dist)
        
        # Ângulo da mão (vetor do pulso para o dedo médio)
        middle_mcp = landmarks[9]
        palm_vector = middle_mcp - palm_base
        features.extend(palm_vector)
        
        return features

    def _extract_interaction_features(self, landmarks1, landmarks2):
        """Extrai features de interação entre duas mãos"""
        features = []
        l1 = np.array(landmarks1).reshape(-1, 3)
        l2 = np.array(landmarks2).reshape(-1, 3)
        
        # Distância entre pulsos
        wrist_dist = np.linalg.norm(l1[0] - l2[0])
        features.append(wrist_dist)
        
        # Distância entre palmas
        palm_dist = np.linalg.norm(l1[9] - l2[9])
        features.append(palm_dist)
        
        return features

    def add_training_sample(self, landmarks_list, sign_name):
        """Adiciona amostra de treino"""
        features = self.extract_libras_features(landmarks_list)
        if features is not None:
            self.training_data.append(features)
            self.training_labels.append(sign_name)
            return True
        return False

    def train(self, data=None, labels=None):
        """Treina o modelo"""
        if data is None:
            data = self.training_data
        if labels is None:
            labels = self.training_labels
            
        if not data or not labels:
            return False
        
        try:
            # Converter para numpy arrays
            X = np.array(data)
            y = np.array(labels)
            
            # Codificar labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            # Normalizar
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar
            self.model.fit(X_train_scaled, y_train)
            
            # Avaliar
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            logging.info(f"Modelo treinado - Acurácia: {accuracy:.2f}")
            
            self.save_model()
            return True
            
        except Exception as e:
            logging.error(f"Erro no treinamento: {e}")
            return False

    def predict(self, landmarks_list):
        """Faz predição"""
        if not self.is_trained:
            return None, 0.0
            
        features = self.extract_libras_features(landmarks_list)
        if features is None:
            return None, 0.0
            
        try:
            features_scaled = self.scaler.transform([features])
            prediction_encoded = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
            
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            return prediction, confidence
            
        except Exception as e:
            logging.error(f"Erro na predição: {e}")
            return None, 0.0

    def start_training_session(self, sign_name):
        """Inicia sessão de treino"""
        self.current_sign_name = sign_name
        self.training_data = []
        self.training_labels = []

    def save_training_session(self):
        """Salva sessão de treino"""
        if not self.training_data:
            return False, "Nenhum dado para salvar"
            
        success = self.train()
        if success:
            return True, f"Sinal '{self.current_sign_name}' salvo com {len(self.training_data)} amostras"
        return False, "Erro ao treinar"

    def get_training_progress(self):
        return len(self.training_data)

    def reset_training(self):
        self.training_data = []
        self.training_labels = []
        self.current_sign_name = ""

    def get_available_signs(self):
        if hasattr(self.label_encoder, 'classes_'):
            return list(self.label_encoder.classes_)
        return []

    def get_model_info(self):
        if not self.is_trained:
            return "Modelo não treinado"
        signs = self.get_available_signs()
        return f"Modelo com {len(signs)} sinais"