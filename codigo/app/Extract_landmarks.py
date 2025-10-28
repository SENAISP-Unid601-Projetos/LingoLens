import numpy as np
import logging
import os

# Configura log (opcional, mas seguro)
log_path = os.path.join(os.path.dirname(__file__), "logs", "extract.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(filename=log_path, level=logging.ERROR, format="%(asctime)s - %(message)s")

def extract_landmarks(hand_landmarks, image_shape=None):
    """
    Extrai 63 features (21 landmarks × 3 coords) normalizadas.
    Usa normalização Z-score robusta (media e desvio padrão).
    Compatível com RandomForest e LSTM (63 entradas).
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        image_shape: (H, W) – opcional (não usado, pois MediaPipe já normaliza)
    
    Returns:
        list[63 floats] ou None
    """
    try:
        if not hand_landmarks or len(hand_landmarks.landmark) != 21:
            logging.error("Hand landmarks inválido ou incompleto")
            return None

        # Extrai x, y, z (já normalizados entre 0 e 1 pelo MediaPipe)
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(21, 3)

        # === NORMALIZAÇÃO Z-SCORE (robusta) ===
        mean = landmarks.mean(axis=0)
        std = landmarks.std(axis=0)
        std = np.where(std == 0, 1.0, std)  # evita divisão por zero
        normalized = (landmarks - mean) / (std + 1e-8)

        # Achata para 63 features
        result = normalized.flatten().tolist()

        if len(result) != 63:
            logging.error(f"Feature vector com tamanho errado: {len(result)}")
            return None

        return result

    except Exception as e:
        logging.error(f"Erro em extract_landmarks: {e}")
        return None