import cv2
import mediapipe as mp
from Config import CONFIG
from Database_manager import DatabaseManager
from Utils import extract_landmarks

class BaseCore:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"],
        )
        self.drawing = mp.solutions.drawing_utils
        
        # Estados compartilhados
        self.current_word = ""
        self.mode = "teste"
        self.samples_count = 0
        self.current_landmarks = None

    def process_frame(self, frame):
        """Processa um frame e retorna landmarks"""
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
                landmarks = extract_landmarks(hand_landmarks)
                if landmarks is not None:
                    landmarks_list.append(landmarks)
                    self.current_landmarks = landmarks

        return image, landmarks_list

    def change_resolution(self, width, height):
        """Altera a resolução da câmera"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verificar se a resolução foi aplicada
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"[INFO] Resolução alterada para: {actual_width}x{actual_height}")
            return True
        except Exception as e:
            print(f"[ERRO] Falha ao alterar resolução: {e}")
            return False

    def cleanup(self):
        """Limpeza dos recursos"""
        self.cap.release()
        cv2.destroyAllWindows()
    def process_frame(self, frame):
        """Processa um frame e retorna landmarks"""
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        landmarks_list = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
                landmarks = extract_landmarks(hand_landmarks)
                # 🔥 CORREÇÃO: Verificação robusta dos landmarks
                if landmarks is not None and hasattr(landmarks, '__len__') and len(landmarks) == 63:
                    landmarks_list.append(landmarks)
                    self.current_landmarks = landmarks

        return image, landmarks_list