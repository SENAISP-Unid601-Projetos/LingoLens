import logging
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError

class ModelManager:
    def __init__(self, neighbors):
        self.model = KNeighborsClassifier(n_neighbors=neighbors)
        self.trained = False

    def train(self, data, labels):
        """Treina o modelo garantindo formato consistente dos dados"""
        try:
            if not data or not labels:
                logging.warning("❌ Dados ou labels vazios para treinamento")
                return False
            
            # 🔥 CORREÇÃO CRÍTICA: Garantir formato consistente
            processed_data = []
            processed_labels = []
            
            for i, (sample, label) in enumerate(zip(data, labels)):
                if sample is None:
                    continue
                    
                try:
                    # Converter para numpy array
                    sample_array = np.array(sample, dtype=np.float64)
                    
                    # Verificar se o array é válido
                    if sample_array.size == 0:
                        logging.warning(f"❌ Amostra {i} vazia - removida")
                        continue
                    
                    # 🔥 CORREÇÃO: Achatar PARA 1D e verificar tamanho
                    if sample_array.ndim > 1:
                        sample_array = sample_array.flatten()
                    
                    # 🔥 NOVA VERIFICAÇÃO: Garantir que todas as amostras têm o mesmo tamanho
                    if processed_data and len(sample_array) != len(processed_data[0]):
                        logging.warning(f"❌ Amostra {i} tem tamanho {len(sample_array)}, esperado {len(processed_data[0])} - removida")
                        continue
                    
                    processed_data.append(sample_array)
                    processed_labels.append(label)
                    
                except Exception as e:
                    logging.warning(f"❌ Erro ao processar amostra {i}: {e} - removida")
                    continue
            
            if not processed_data:
                logging.warning("❌ Nenhum dado válido após processamento")
                return False
            
            # 🔥 VERIFICAÇÃO FINAL DE CONSISTÊNCIA
            sizes = [len(sample) for sample in processed_data]
            unique_sizes = set(sizes)
            
            if len(unique_sizes) > 1:
                logging.error(f"❌ Dados ainda têm tamanhos inconsistentes: {unique_sizes}")
                # Manter apenas o tamanho mais comum
                from collections import Counter
                most_common_size = Counter(sizes).most_common(1)[0][0]
                filtered_data = []
                filtered_labels = []
                for sample, label in zip(processed_data, processed_labels):
                    if len(sample) == most_common_size:
                        filtered_data.append(sample)
                        filtered_labels.append(label)
                processed_data = filtered_data
                processed_labels = filtered_labels
                
                if not processed_data:
                    logging.warning("❌ Nenhum dado após filtragem por tamanho")
                    return False

            # Converter para array numpy
            X = np.array(processed_data)
            y = np.array(processed_labels)
            
            logging.info(f"📊 Dados processados: {X.shape}, Labels: {y.shape}")
            
            unique_labels = set(processed_labels)
            if len(unique_labels) < 1:
                logging.warning("❌ Nenhuma classe disponível para treino")
                return False

            if len(unique_labels) == 1:
                logging.warning(f"⚠️ Apenas uma classe ({list(unique_labels)[0]}) disponível")

            self.model.fit(X, y)
            self.trained = True
            logging.info(f"✅ Modelo treinado com {len(X)} amostras e {len(unique_labels)} classes")
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro no treinamento: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False

    def predict(self, landmarks):
        if not self.trained:
            logging.warning("❌ Modelo ainda não treinado")
            return None, 0.0

        try:
            # 🔥 CORREÇÃO: Garantir formato consistente na predição também
            if landmarks is None:
                return None, 0.0
                
            landmarks_array = np.array(landmarks, dtype=np.float64).flatten()
            landmarks_reshaped = landmarks_array.reshape(1, -1)
            
            prediction = self.model.predict(landmarks_reshaped)[0]
            probability = self.model.predict_proba(landmarks_reshaped).max()
            
            logging.debug(f"🎯 Predição: {prediction} | Probabilidade: {probability:.2f}")
            return prediction, probability
            
        except Exception as e:
            logging.error(f"❌ Erro na predição: {e}")
            return None, 0.0