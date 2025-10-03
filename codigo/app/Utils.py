import numpy as np
import logging

def extract_landmarks(hand_landmarks, image_shape=None):
    """
    Extrai e normaliza os pontos de referência de uma mão detectada pelo MediaPipe.
    Inclui distâncias entre pontos-chave para melhorar a diferenciação.
    Args:
        hand_landmarks: Objeto de pontos de referência do MediaPipe.
        image_shape: Tupla (altura, largura) da imagem para normalização (opcional).
    Returns:
        Lista achatada com coordenadas normalizadas e distâncias ou None se inválido.
    """
    landmarks = []
    for lm in hand_landmarks.landmark:
        x = lm.x * image_shape[1] if image_shape else lm.x
        y = lm.y * image_shape[0] if image_shape else lm.y
        z = lm.z
        landmarks.append([x, y, z])
        logging.debug(f"Coordenada: x={x}, y={y}, z={z}, image_shape={image_shape}")
    
    landmarks_array = np.array(landmarks)
    if landmarks_array.size != 63:
        logging.error(f"Landmarks inválidos ({landmarks_array.size})")
        return None
    
    landmarks_reshaped = landmarks_array.reshape(-1, 3)
    normalized = (landmarks_reshaped - landmarks_reshaped.mean(axis=0)) / (landmarks_reshaped.std(axis=0) + 1e-8)
    
    distances = []
    key_pairs = [(4, 8), (8, 12)]  # Polegar (4) - Indicador (8), Indicador (8) - Médio (12)
    for p1, p2 in key_pairs:
        dist = np.linalg.norm(landmarks_reshaped[p1] - landmarks_reshaped[p2])
        distances.append(dist)
    
    result = np.concatenate([normalized.flatten(), distances]).tolist()
    logging.debug(f"Tamanho do vetor de features: {len(result)}")
    return result