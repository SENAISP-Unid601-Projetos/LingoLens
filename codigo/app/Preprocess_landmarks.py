import numpy as np
import logging
import os

# Configura log seguro
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "preprocess.log"),
    level=logging.ERROR,
    format="%(asctime)s - %(message)s"
)

def preprocess_landmarks(hand_landmarks, image_shape):
    """
    Extrai 63 coordenadas (21 landmarks × 3) e normaliza com Z-score.
    MediaPipe já entrega x, y normalizados (0-1). z é relativo.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        image_shape: (H, W, C) – usado apenas para validação
    
    Returns:
        list[63 floats] normalizados ou None
    """
    try:
        if not hand_landmarks or len(hand_landmarks.landmark) != 21:
            logging.error("Hand landmarks inválido ou incompleto")
            return None

        height, width, _ = image_shape

        # Extrai x, y (normalizados), z (relativo)
        landmarks = []
        for lm in hand_landmarks.landmark:
            x = lm.x  # já normalizado [0,1]
            y = lm.y  # já normalizado [0,1]
            z = lm.z  # profundidade relativa (não escalar!)
            landmarks.extend([x, y, z])

        landmarks = np.array(landmarks).reshape(21, 3)

        # === Z-SCORE NORMALIZATION (robusta) ===
        mean = landmarks.mean(axis=0)
        std = landmarks.std(axis=0)
        std = np.where(std == 0, 1.0, std)  # evita divisão por zero
        normalized = (landmarks - mean) / (std + 1e-8)

        result = normalized.flatten().tolist()

        if len(result) != 63:
            logging.error(f"Feature vector com tamanho errado: {len(result)}")
            return None

        return result

    except Exception as e:
        logging.error(f"Erro em preprocess_landmarks: {e}")
        return None