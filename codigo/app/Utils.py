import numpy as np
import logging

def extract_landmarks(hand_landmarks):
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
    if landmarks.size != 63:
        logging.error(f"Landmarks inválidos ({landmarks.size})")
        return None
    landmarks_reshaped = landmarks.reshape(-1, 3)
    return ((landmarks_reshaped - landmarks_reshaped.mean(axis=0)) /
            (landmarks_reshaped.std(axis=0) + 1e-8)).flatten()
