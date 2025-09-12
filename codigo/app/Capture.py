import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def capturar_landmarks(frame):
    """Detecta landmarks da mão em um frame."""
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
    return None

def desenhar_landmarks(frame, landmarks):
    """Desenha landmarks no frame (para debug/visualização)."""
    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
