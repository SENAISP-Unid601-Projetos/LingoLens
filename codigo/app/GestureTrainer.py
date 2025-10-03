import cv2
import mediapipe as mp
import logging
from Database_manager import DatabaseManager
from Utils import extract_landmarks
from Config import CONFIG
from TrainUIManager import TrainUIManager


class GestureTrainer:
    def __init__(self):
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.db = DatabaseManager(CONFIG["db_path"])
        self.ui = TrainUIManager()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"]
        )

        # Carrega gestos existentes
        self.labels, self.data, _ = self.db.load_gestures()

    def run(self, letter):
        self.ui.set_current_letter(letter)
        print(f"[INFO] Treinando letra: {letter}")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("[WARNING] Falha ao capturar frame.")
                break

            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    landmarks = extract_landmarks(hand)
                    if landmarks and len(landmarks) == 63:  # garante consistência
                        self.current_landmarks = landmarks

            # Desenha interface
            frame = self.ui.draw_train_ui(image)
            cv2.imshow("Treino de Gestos", frame)

            # Captura tecla (uma única vez por loop)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and hasattr(self, "current_landmarks"):
                self.db.add_gesture(letter, self.current_landmarks, g_type="letter")
                self.labels.append(letter)
                self.data.append(self.current_landmarks)
                print(f"[INFO] Gesto '{letter}' salvo no banco!")

            elif key == ord("q"):
                print("[INFO] Saindo do treino.")
                break

        self.cap.release()
        self.db.close()
        cv2.destroyAllWindows()
