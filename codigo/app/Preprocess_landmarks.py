import numpy as np

def preprocess_landmarks(hand_landmarks, image_shape):
    """Extrai landmarks 3D normalizados."""
    try:
        h, w, _ = image_shape
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x * w, lm.y * h, lm.z * w])
        return landmarks
    except Exception as e:
        print(f"[ERRO] Preprocess: {e}")
        return None