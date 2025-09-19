import cv2
import mediapipe as mp
import logging
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Utils import extract_landmarks

class GestureRecognizer:
    def __init__(self):
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.db = DatabaseManager(CONFIG["db_path"])
        self.model_manager = ModelManager(CONFIG["knn_neighbors"])
        self.ui = UIManager()
        self.labels, self.data, self.gesture_names = self.db.load_gestures()
        if self.labels:
            self.model_manager.train(self.data, self.labels)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG["target_fps"])

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"]
        )

        self.current_word = ""

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    landmarks = extract_landmarks(hand)
                    if landmarks is not None and self.labels:
                        pred, prob = self.model_manager.predict(landmarks)
                        if prob >= CONFIG["confidence_threshold"]:
                            self.current_word += pred

            status = "Modo: Reconhecimento"
            self.ui.draw_ui(image, status, 0, self.current_word)

            cv2.imshow("Reconhecimento de Gestos", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        self.db.close()
        cv2.destroyAllWindows()
