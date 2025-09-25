import cv2
import mediapipe as mp
import logging
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Utils import extract_landmarks

class GestureApp:
    def __init__(self):
        print("[INFO] Inicializando GestureApp...")

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
            print(f"[INFO] Modelo carregado com {len(self.labels)} gestos.")
        else:
            print("[INFO] Nenhum gesto treinado ainda.")

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        self.cap.set(cv2.CAP_PROP_FPS, CONFIG["target_fps"])

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"]
        )

        self.current_word = ""
        self.mode = "teste"  # modos: "teste" ou "treino"
        self.new_gesture_name = ""
        self.new_gesture_data = []

    def run(self):
        print("[INFO] Teclas: Q=Sair C=Limpar T=Treino S=Teste")

        while True:
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
                    if landmarks is not None and len(landmarks) > 0:
                        self.new_gesture_data.append(landmarks)
                        if self.mode == "teste" and self.labels:
                            pred, prob = self.model_manager.predict(landmarks)
                            if prob >= CONFIG["confidence_threshold"]:
                                self.current_word += pred
                        elif self.mode == "treino" and self.new_gesture_name:
                            self.new_gesture_data.append(landmarks)

            status = f"Modo: {'Treino' if self.mode=='treino' else 'Teste'}"
            self.ui.draw_ui(image, status, 0, self.current_word)

            cv2.imshow("GestureApp", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("[INFO] Saindo do aplicativo...")
                break
            elif key == ord("c"):
                self.current_word = ""
                print("[INFO] Palavra atual limpa.")
            elif key == ord("t"):
                self.mode = "treino"
                self.new_gesture_name = input("Digite o nome da nova letra/gesto: ").upper()
                self.new_gesture_data = []
                print(f"[INFO] Modo Treino ativado para '{self.new_gesture_name}'")
            elif key == ord("s"):
                if self.mode == "treino" and self.new_gesture_name and self.new_gesture_data:
                    self.labels += [self.new_gesture_name]*len(self.new_gesture_data)
                    self.data += self.new_gesture_data
                    self.db.save_gestures(self.labels, self.data)
                    self.model_manager.train(self.data, self.labels)
                    print(f"[INFO] Gestos de '{self.new_gesture_name}' salvos e modelo atualizado.")
                self.mode = "teste"
                self.new_gesture_name = ""
                self.new_gesture_data = []
                print("[INFO] Modo Teste ativado.")

        self.cap.release()
        cv2.destroyAllWindows()
        self.db.close()
        print("[INFO] GestureApp encerrado.")


if __name__ == "__main__":
    app = GestureApp()
    app.run()
