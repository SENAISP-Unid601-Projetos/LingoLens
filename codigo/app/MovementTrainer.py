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
        self.sequence_length = CONFIG["libras_sequence_length"]
        self.confidence_threshold = CONFIG["libras_confidence_threshold"]
        
        # Estados
        self.training_data = []
        self.training_labels = []
        self.current_sign_name = ""
        
        # Buffer para sequências no treino
        self.training_buffer = deque(maxlen=self.sequence_length)
        
        # Caminho do modelo .pkl
        self.model_path = os.path.join(CONFIG["train_data_dir"], "libras_model.pkl")
        
        # 🔥 CORREÇÃO: Carregar do banco ao inicializar
        self._load_from_database()

    def _load_from_database(self):
        """Carrega dados do banco - FONTE PRIMÁRIA"""
        try:
            labels, data, _ = self.db.load_movements()
            if labels and data:
                self.training_data = data
                self.training_labels = labels
                
                # Se temos dados, treinar o modelo
                if len(data) > 0:
                    success = self._train_from_data()
                    if success:
                        logging.info(f"✅ Modelo carregado do banco: {len(labels)} movimentos")
                    else:
                        logging.error("❌ Erro ao treinar modelo com dados do banco")
                else:
                    self.is_trained = False
            else:
                logging.info("📝 Nenhum movimento encontrado no banco")
                self.is_trained = False
                
        except Exception as e:
            logging.error(f"❌ Erro ao carregar do banco: {e}")
            self.is_trained = False

    def _train_from_data(self):
        """Treina o modelo com os dados atuais"""
        if not self.training_data or not self.training_labels:
            self.is_trained = False
            return False
        
        try:
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # Codificar labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
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
            logging.info(f"🎯 Modelo treinado com {len(X_train)} amostras - Acurácia: {accuracy:.2f}")
            
            # Salvar cache .pkl
            self._save_model_cache()
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro no treinamento: {e}")
            self.is_trained = False
            return False

    def _save_model_cache(self):
        """Salva cache do modelo (opcional)"""
        try:
            os.makedirs(CONFIG["train_data_dir"], exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }
            joblib.dump(model_data, self.model_path)
        except Exception as e:
            logging.error(f"❌ Erro ao salvar cache: {e}")

    def _save_to_database(self):
        """Salva dados no banco de dados"""
        try:
            success = self.db.save_movements(self.training_labels, self.training_data)
            if success:
                logging.info("✅ Dados salvos no banco com sucesso")
            return success
        except Exception as e:
            logging.error(f"❌ Erro ao salvar no banco: {e}")
            return False

    def extract_libras_features(self, landmarks_list):
        try:
            features = []
            for landmarks in landmarks_list:
                flat = np.array(landmarks).flatten()
                features.extend(flat)
            
            # Features de interação se duas mãos
            if len(landmarks_list) == 2:
                interaction = self.extract_interaction_features(landmarks_list[0], landmarks_list[1])
                if interaction:
                    features.extend(interaction)
            
            return np.array(features)
        except Exception as e:
            logging.error(f"❌ Erro ao extrair features: {e}")
            return None

    def extract_interaction_features(self, landmarks1, landmarks2):
        try:
            features = []
            l1 = np.array(landmarks1).reshape(-1, 3)
            l2 = np.array(landmarks2).reshape(-1, 3)
            
            wrist_dist = np.linalg.norm(l1[0] - l2[0])
            features.append(wrist_dist)
            
            palm_dist = np.linalg.norm(l1[9] - l2[9])
            features.append(palm_dist)
            
            return features
        except Exception as e:
            logging.error(f"❌ Erro ao extrair features de interação: {e}")
            return None

    def add_training_sample(self, landmarks_list, sign_name):
        features = self.extract_libras_features(landmarks_list)
        if features is not None:
            self.training_buffer.append(features)
            if len(self.training_buffer) == self.sequence_length:
                seq_features = np.concatenate(list(self.training_buffer)).flatten()
                self.training_data.append(seq_features)
                self.training_labels.append(sign_name)
                self.training_buffer.clear()
                return True
        return False

    def train(self):
        """Treina o modelo e salva no banco"""
        success = self._train_from_data()
        if success:
            # 🔥 CORREÇÃO: SEMPRE salvar no banco após treinar
            db_success = self._save_to_database()
            if not db_success:
                logging.error("❌ Modelo treinado mas não salvo no banco!")
            return db_success
        return False

    def predict(self, landmarks_list):
        if not self.is_trained:
            return None, 0.0
            
        features = self.extract_libras_features(landmarks_list)
        if features is None:
            return None, 0.0
            
        try:
            if len(features) == 0:
                return None, 0.0
                
            features_reshaped = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features_reshaped)
            prediction_encoded = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_reshaped)[0]
            confidence = np.max(probabilities)
            
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            return prediction, confidence
            
        except Exception as e:
            logging.error(f"❌ Erro na predição: {e}")
            return None, 0.0

    def start_training_session(self, sign_name):
        self.current_sign_name = sign_name
        logging.info(f"🎬 Iniciando treino para sinal: {sign_name}")
        self.training_buffer.clear()

    def save_training_session(self):
        if not self.training_data:
            return False, "❌ Nenhum dado para salvar"
            
        success = self.train()
        if success:
            return True, f"✅ Sinal '{self.current_sign_name}' salvo com {len(self.training_data)} amostras"
        return False, "❌ Erro ao treinar modelo"

    def delete_sign(self, sign_name):
        """Remove um sinal do banco de dados"""
        try:
            # 1. Deletar do banco de dados
            success = self.db.delete_movement(sign_name)
            if not success:
                return False, f"❌ Movimento '{sign_name}' não encontrado no banco"

            # 2. Recarregar do banco (fonte primária)
            self._load_from_database()
            
            return True, f"✅ Movimento '{sign_name}' deletado com sucesso"
                
        except Exception as e:
            logging.error(f"❌ Erro ao deletar movimento '{sign_name}': {e}")
            return False, f"❌ Erro ao deletar movimento: {str(e)}"

    def get_training_progress(self):
        return len(self.training_data)

    def reset_training(self):
        self.training_data = []
        self.training_labels = []
        self.current_sign_name = ""
        self.training_buffer.clear()

    def get_available_signs(self):
        if hasattr(self.label_encoder, 'classes_'):
            return list(self.label_encoder.classes_)
        return []

    def get_model_info(self):
        if not self.is_trained:
            return "📝 Modelo não treinado"
        signs = self.get_available_signs()
        if signs:
            return f"🎯 {len(signs)} sinais: {', '.join(signs[:3])}{'...' if len(signs) > 3 else ''}"
        return "🎯 Modelo vazio"