import numpy as np

def extract_landmarks(hand_landmarks, image_shape=None):
    if image_shape is not None and len(image_shape) < 2:
        return None

    landmarks = []
    for lm in hand_landmarks.landmark:
        x = lm.x * image_shape[1] if image_shape else lm.x
        y = lm.y * image_shape[0] if image_shape else lm.y
        z = lm.z
        landmarks.append([x, y, z])
    
    landmarks_array = np.array(landmarks)
    if len(landmarks_array) != 21:
        return None

    # Origem = pulso
    wrist = landmarks_array[0]
    translated = landmarks_array - wrist

    # Escala
    finger_tips = [4, 8, 12, 16, 20]
    distances = [np.linalg.norm(translated[i]) for i in finger_tips]
    scale = np.mean(distances) if distances else 1.0
    if scale == 0: scale = 1.0
    normalized = translated / scale

    # 4 distâncias extras → 63 + 4 = 67
    key_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    extra = [np.linalg.norm(normalized[p1] - normalized[p2]) for p1, p2 in key_pairs]

    return np.concatenate([normalized.flatten(), extra]).tolist()