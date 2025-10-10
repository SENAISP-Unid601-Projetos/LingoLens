import cv2
import mediapipe as mp
from MovementTrainer import MovementTrainer
from MovementUIManager import MovementUIManager


class MovementApp:
    def __init__(self, db):
        self.db = db
        self.ui = MovementUIManager()
        self.trainer = MovementTrainer(db)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1, min_detection_confidence=0.5
        )
        self.drawing = mp.solutions.drawing_utils
        self.current_word = ""
        self.mode = "teste"
        self.new_movement_name = ""
        self.new_movement_data = []

    def run(self):
        image = self.ui.draw_ui(
            image, f"Treinar Movimentos - Modo: {self.mode}", 0, self.current_word
        )
        print(
            "[INFO] Teclas: Q=Sair C=Limpar T=Treino S=Salvar N=Nome D=Deletar ESC=Cancelar M=Gestos"
        )
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(
                        image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
                    landmarks = self.trainer.extract_landmarks(hand_landmarks)
                    if landmarks is not None:
                        if self.mode == "teste":
                            pred, prob = self.trainer.predict(landmarks)
                            if prob >= 0.5:
                                self.current_word += pred
                        elif self.mode == "treino" and self.new_movement_name:
                            self.new_movement_data.append(landmarks)

            image = self.ui.draw_ui(image, "Treinar Movimentos", 0, self.current_word)
            cv2.imshow("MovementApp", image)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC cancela treino
                self.mode = "teste"
                self.new_movement_name = ""
                self.new_movement_data = []
                self.ui.set_error("Treino cancelado!")
            elif key == ord("m"):
                self.cap.release()
                cv2.destroyAllWindows()
                return "gesture"
            elif key == ord("c"):
                self.current_word = ""
            elif key == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        return None
