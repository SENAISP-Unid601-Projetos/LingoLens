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
        self.drawing = mp.solutions.drawing_utils

        self.labels, self.data, _ = self.db.load_gestures()
        self.current_landmarks = None

    def run(self, letter=None):
        if letter:
            self.ui.set_current_letter(letter)
        else:
            self.ui.show_text_input = True
            self.ui.input_prompt = "Digite a letra para treinar:"
            letter = self._get_letter_input()
            if not letter:
                return

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
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(
                        image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
                    landmarks = extract_landmarks(hand_landmarks)
                    if landmarks and len(landmarks) == 63:
                        self.current_landmarks = landmarks

            status = "Pressione S para salvar o gesto"
            image = self.ui.draw_train_ui(image, status, "")
            cv2.imshow("Treino de Gestos", image)

            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("s") and self.current_landmarks is not None:
                success = self.db.add_gesture(letter, self.current_landmarks, g_type="letter")
                if success:
                    self.labels.append(letter)
                    self.data.append(self.current_landmarks)
                    self.ui.set_error(f"Gesto '{letter}' salvo com sucesso!")
                    print(f"[INFO] Gesto '{letter}' salvo no banco!")
                else:
                    self.ui.set_error(f"Erro ao salvar gesto '{letter}'")

            elif key == ord("c"):
                self.current_landmarks = None
                self.ui.set_error("Gesto atual limpo")

            elif key == ord("m"):
                print("[INFO] Voltando para modo movimentos.")
                break

            elif key == ord("q"):
                print("[INFO] Saindo do treino.")
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _get_letter_input(self):
        """Obtém input de letra do usuário - COM ENTER CORRIGIDO"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            image = self.ui.draw_train_ui(image, "Digite a letra e pressione Enter", "")
            cv2.imshow("Treino de Gestos", image)

            key = cv2.waitKey(1) & 0xFF

            # CORREÇÃO DO ENTER
            if key == 13 or key == 10:  # Enter (ambos os códigos)
                letter = self.ui.input_text.upper()
                self.ui.input_text = ""
                self.ui.show_text_input = False
                if letter:
                    return letter
                else:
                    self.ui.set_error("Digite uma letra válida")
                    
            elif key == 8:  # Backspace
                self.ui.input_text = self.ui.input_text[:-1]
                
            elif key == 27:  # ESC
                break
                
            elif key != 255 and key != 0:  # Ignorar teclas especiais
                if 32 <= key <= 126:  # Caracteres ASCII imprimíveis
                    self.ui.input_text += chr(key)

        return None