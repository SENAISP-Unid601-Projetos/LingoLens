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
    if image_shape is not None and (not isinstance(image_shape, tuple) or len(image_shape) < 2):
        logging.error("image_shape deve ser uma tupla com pelo menos 2 elementos (altura, largura)")
        return None

    landmarks = []
    for lm in hand_landmarks.landmark:
        x = lm.x * image_shape[1] if image_shape else lm.x
        y = lm.y * image_shape[0] if image_shape else lm.y
        z = lm.z
        landmarks.append([x, y, z])
        logging.debug(f"Coordenada: x={x}, y={y}, z={z}, image_shape={image_shape}")
    
    landmarks_array = np.array(landmarks)
    if len(landmarks_array) != 21:  # MediaPipe retorna 21 landmarks por mão
        logging.error(f"Esperado 21 landmarks, mas recebido {len(landmarks_array)}")
        return None
    
    landmarks_reshaped = landmarks_array.reshape(-1, 3)
    std = landmarks_reshaped.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # Evitar divisão por zero
    normalized = (landmarks_reshaped - landmarks_reshaped.mean(axis=0)) / (std + 1e-8)
    
    distances = []
    key_pairs = [(4, 8), (8, 12)]  # Polegar (4) - Indicador (8), Indicador (8) - Médio (12)
    for p1, p2 in key_pairs:
        dist = np.linalg.norm(landmarks_reshaped[p1] - landmarks_reshaped[p2])
        distances.append(dist)
    
    result = np.concatenate([normalized.flatten(), distances]).tolist()
    logging.debug(f"Tamanho do vetor de features: {len(result)}")
    return result