import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Inicializando o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.6)

# Função para calcular a distância euclidiana entre dois pontos
def calculate_distance(p1, p2):
    return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

# Função para determinar se um dedo está levantado
def is_finger_up(landmarks, finger_tip_idx, finger_mcp_idx, finger_pip_idx, is_thumb=False, is_right_hand=True):
    tip = landmarks[finger_tip_idx]
    mcp = landmarks[finger_mcp_idx]
    pip = landmarks[finger_pip_idx]

    if is_thumb:
        thumb_base = landmarks[mp_hands.HandLandmark.THUMB_CMC]
        wrist = landmarks[mp_hands.HandLandmark.WRIST]
        dist_tip_base = calculate_distance(tip, thumb_base)
        dist_pip_base = calculate_distance(pip, thumb_base)
        vector_x = tip.x - thumb_base.x
        vector_y = tip.y - thumb_base.y
        return dist_tip_base > dist_pip_base * 1.2 and abs(vector_x) > abs(vector_y) * 0.5
    else:
        dist_tip_mcp = calculate_distance(tip, mcp)
        dist_pip_mcp = calculate_distance(pip, mcp)
        return dist_tip_mcp > dist_pip_mcp * 1.1 and tip.y < pip.y

# Função para contar dedos levantados e retornar quais estão levantados
def count_raised_fingers(landmarks, is_right_hand):
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,  # Polegar
        mp_hands.HandLandmark.INDEX_FINGER_TIP,  # Indicador
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,  # Médio
        mp_hands.HandLandmark.RING_FINGER_TIP,  # Anelar
        mp_hands.HandLandmark.PINKY_TIP  # Mindinho
    ]
    finger_mcps = [
        mp_hands.HandLandmark.THUMB_CMC,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    raised_fingers = []
    for i, (tip_idx, mcp_idx, pip_idx) in enumerate(zip(finger_tips, finger_mcps, finger_pips)):
        is_thumb = (i == 0)
        if is_finger_up(landmarks, tip_idx, mcp_idx, pip_idx, is_thumb, is_right_hand):
            raised_fingers.append(tip_idx)
    return raised_fingers

# Mapeamento de combinações de dedos para letras
# Cada tupla representa [Polegar, Indicador, Médio, Anelar, Mindinho]
finger_to_letter = {
    (False, True, False, False, False): 'A',  # Apenas indicador
    (False, True, True, False, False): 'B',   # Indicador e médio
    (False, True, True, True, False): 'C',    # Indicador, médio e anelar
    (False, True, True, True, True): 'D',     # Indicador, médio, anelar e mínimo
    (True, False, False, False, False): 'E',  # Apenas polegar
    (True, True, False, False, False): 'F',   # Polegar e indicador
     (True,True,False,False,True): 'I love you',
     (True,False, False, False,True): 'Hang Loose',
    # Adicione mais combinações conforme necessário
}

# Captura de vídeo
cap = cv2.VideoCapture(0)

# Variáveis para estabilizar a detecção e formar palavras
last_combination = None
combination_count = 0
required_frames = 10  # Número de frames para confirmar uma letra
current_word = ""
combination_buffer = deque(maxlen=required_frames)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inverter a imagem horizontalmente (espelhar)
    frame = cv2.flip(frame, 1)

    # Converta a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Processar resultados
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Desenhar landmarks na imagem
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determinar se é mão direita ou esquerda
            is_right_hand = handedness.classification[0].label == "Right"

            # Identificar dedos levantados
            raised_fingers = count_raised_fingers(hand_landmarks.landmark, is_right_hand)

            # Criar tupla de combinação de dedos
            finger_state = [
                mp_hands.HandLandmark.THUMB_TIP in raised_fingers,
                mp_hands.HandLandmark.INDEX_FINGER_TIP in raised_fingers,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP in raised_fingers,
                mp_hands.HandLandmark.RING_FINGER_TIP in raised_fingers,
                mp_hands.HandLandmark.PINKY_TIP in raised_fingers
            ]

            # Adicionar ao buffer para estabilização
            combination_buffer.append(tuple(finger_state))

            # Verificar se a combinação é estável
            if len(combination_buffer) == required_frames and all(c == combination_buffer[0] for c in combination_buffer):
                current_combination = combination_buffer[0]
                if current_combination in finger_to_letter:
                    letter = finger_to_letter[current_combination]
                    if last_combination != current_combination:
                        current_word += letter
                        last_combination = current_combination
                        combination_buffer.clear()  # Limpar buffer após adicionar letra

            # Destacar dedos levantados (depuração visual)
            for finger_tip_idx in raised_fingers:
                tip = hand_landmarks.landmark[finger_tip_idx]
                h, w, _ = image.shape
                cx, cy = int(tip.x * w), int(tip.y * h)
                color = (0, 0, 255) if finger_tip_idx != mp_hands.HandLandmark.THUMB_TIP else (255, 0, 0)
                cv2.circle(image, (cx, cy), 10, color, -1)

            # Exibir informações na imagem
            label = handedness.classification[0].label
            text = f"{label} hand: {len(raised_fingers)} fingers"
            cv2.putText(image, text, (50, 50 if label == 'Left' else 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Letter: {finger_to_letter.get(tuple(finger_state), '')}", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"Word: {current_word}", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Exibir a imagem
    cv2.imshow('Hand Finger Counter', image)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()