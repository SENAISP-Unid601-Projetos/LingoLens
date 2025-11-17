import numpy as np

def extract_landmarks(hand_landmarks, image_shape=None):
    """
    Extrai 69 features dos landmarks da mão (MediaPipe).
    Inclui orientação absoluta dos dedos → resolve H vs K de uma vez por todas.
    """
    if image_shape is not None and len(image_shape) < 2:
        return None

    # === 1. Coleta os 21 pontos em coordenadas absolutas ===
    landmarks = []
    for lm in hand_landmarks.landmark:
        x = lm.x * image_shape[1] if image_shape else lm.x
        y = lm.y * image_shape[0] if image_shape else lm.y
        z = lm.z
        landmarks.append([x, y, z])
    
    landmarks_array = np.array(landmarks, dtype=np.float32)
    if landmarks_array.shape[0] != 21:
        return None

    # === 2. Normalização clássica (pulso = origem + escala) ===
    wrist = landmarks_array[0]
    translated = landmarks_array - wrist                     # 21 × 3
    finger_tips = [4, 8, 12, 16, 20]
    distances = np.linalg.norm(translated[finger_tips], axis=1)
    scale = np.mean(distances)
    if scale < 1e-6:
        scale = 1.0
    normalized = translated / scale                           # 21 × 3 normalizado

    # === 3. 4 distâncias entre falanges (já tinha) ===
    key_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    extra_distances = [
        np.linalg.norm(normalized[p1] - normalized[p2]) for p1, p2 in key_pairs
    ]  # 4 valores

    # === 4. FEATURES DECISIVAS PARA H vs K (orientação absoluta) ===
    # Vetores dos dois dedos esticados (indicador e médio) a partir do pulso
    index_vec  = normalized[8]   # ponta do indicador
    middle_vec = normalized[12]  # ponta do médio

    # Vetor médio dos dois dedos (representa a direção geral dos dedos esticados)
    direction_vec_xy = (index_vec[:2] + middle_vec[:2]) / 2.0

    # Ângulo absoluto da direção dos dedos (em radianos, -π a +π)
    direction_angle = np.arctan2(direction_vec_xy[1], direction_vec_xy[0])

    # Magnitude do vetor médio (útil para diferenciar mãos muito próximas vs distantes)
    direction_magnitude = np.linalg.norm(direction_vec_xy)

    # === 5. Montagem do vetor final (69 features) ===
    # 21 pontos × 3 coords = 63
    # + 4 distâncias extras = 67
    # + direction_angle + direction_magnitude = 69
    feature_vector = np.concatenate([
        normalized.flatten(),           # 63 valores
        extra_distances,                # +4  → 67
        [direction_angle, direction_magnitude]  # +2 → 69
    ])

    return feature_vector.tolist()  # lista de 69 floats