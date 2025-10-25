import numpy as np
import logging

def preprocess_landmarks(hand_landmarks, image_shape):
    """Extrai e normaliza landmarks da mão com dimensões explícitas da imagem."""
    try:
        height, width, _ = image_shape
        landmarks = []
        for landmark in hand_landmarks.landmark:
            # Converter coordenadas normalizadas para coordenadas de pixel
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # Escalar z pela largura para consistência
            landmarks.extend([x, y, z])
        
        if len(landmarks) != 63:  # 21 landmarks * 3 coordenadas (x, y, z)
            logging.error(f"Erro: Número de landmarks inválido ({len(landmarks)}). Esperado 63.")
            return None

        return landmarks
    except Exception as e:
        logging.error(f"Erro ao processar landmarks: {e}")
        return None