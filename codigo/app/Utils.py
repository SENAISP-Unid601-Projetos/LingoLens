import numpy as np
import logging

def extract_landmarks(hand_landmarks):
    """
    Extrai e normaliza os landmarks da mão.
    Retorna None se landmarks inválidos.
    """
    try:
        if not hand_landmarks or not hasattr(hand_landmarks, 'landmark'):
            return None
            
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        
        # 🔥 CORREÇÃO: Verificação mais robusta
        if landmarks.size != 63:
            logging.warning(f"Landmarks com tamanho inválido: {landmarks.size}")
            return None

        # Verificar se há valores NaN ou infinitos
        if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
            logging.warning("Landmarks contém valores inválidos (NaN ou infinito)")
            return None

        # Normalização robusta
        landmarks_reshaped = landmarks.reshape(-1, 3)
        
        # Verifica se há variação nos dados para normalização
        if np.std(landmarks_reshaped, axis=0).any() == 0:
            # Retorna sem normalizar se todos pontos iguais
            return landmarks.flatten()
            
        normalized = (landmarks_reshaped - landmarks_reshaped.mean(axis=0)) / (landmarks_reshaped.std(axis=0) + 1e-8)
        return normalized.flatten()
        
    except Exception as e:
        logging.error(f"Erro ao extrair landmarks: {e}")
        return None