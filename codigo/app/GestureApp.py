import cv2
import mediapipe as mp
import logging
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from UI_manager import UIManager
from Utils import extract_landmarks
from GestureTrainer import GestureTrainer  # Adicione esta importação


class GestureApp:
    def __init__(self, db: DatabaseManager):
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.db = db
        self.ui = UIManager()
        self.model_manager = ModelManager(CONFIG["knn_neighbors"])
        self.labels, self.data, _ = self.db.load_gestures()
        if self.labels:
            self.model_manager.train(self.data, self.labels)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=CONFIG["max_num_hands"],
            min_detection_confidence=CONFIG["min_detection_confidence"],
        )
        self.drawing = mp.solutions.drawing_utils
        self.current_word = ""
        self.mode = "teste"
        self.new_gesture_name = ""
        self.new_gesture_data = []
        self.gesture_trainer = GestureTrainer()  # Instancia o treinador

    def run(self):
        print(
            "[INFO] Teclas: Q=Sair | C=Limpar | T=Treino | S=Salvar | N=Nome | D=Deletar | M=Movimento | G:Treinar Gesto"
        )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.ui.set_error("Falha ao capturar frame da câmera!")
                break

            image = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            # Detecção e desenho
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(
                        image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
                    landmarks = extract_landmarks(hand_landmarks)
                    if landmarks is not None:
                        if self.mode == "teste" and self.labels:
                            pred, prob = self.model_manager.predict(landmarks)
                            if prob >= CONFIG["confidence_threshold"]:
                                self.current_word += pred
                        elif self.mode == "treino" and self.new_gesture_name:
                            self.new_gesture_data.append(landmarks)

            # Desenha UI com título fixo
            image = self.ui.draw_ui(image, "Treinar Gestos", 0, self.current_word)
            cv2.imshow("GestureApp", image)

            key = cv2.waitKey(1) & 0xFF

            # Se estiver digitando texto (nome/deleção)
            if self.ui.show_text_input:
                if key == 13:  # Enter
                    gesture_name = self.ui.input_text.upper()
                    self.ui.input_text = ""
                    self.ui.show_text_input = False
                    if getattr(self.ui, "input_action", "") == "delete":
                        success = self.db.delete_gesture(gesture_name)
                        if success:
                            self.labels, self.data, _ = self.db.load_gestures()
                            if self.labels:
                                self.model_manager.train(self.data, self.labels)
                            self.ui.set_error(
                                f"Gesto '{gesture_name}' deletado com sucesso!"
                            )
                        else:
                            self.ui.set_error(
                                f"Erro ao deletar gesto '{gesture_name}'."
                            )
                        self.ui.input_action = ""
                    else:
                        self.new_gesture_name = gesture_name
                        self.ui.set_error(
                            f"Gesto '{self.new_gesture_name}' definido. Pressione 'S' para salvar."
                        )
                elif key == 8:
                    self.ui.input_text = self.ui.input_text[:-1]
                elif key != 255:
                    self.ui.input_text += chr(key)

            else:
                # ESC cancela modo treino ou sai
                if key == 27:
                    if self.mode == "treino":
                        self.mode = "teste"
                        self.new_gesture_name = ""
                        self.new_gesture_data = []
                        self.ui.set_error("Modo treino cancelado.")
                    else:
                        break

                elif key == ord("q"):
                    break
                elif key == ord("m"):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return "movement"
                elif key == ord("g"):  # Novo atalho para treinar gesto
                    self.cap.release()
                    cv2.destroyAllWindows()
                    self.gesture_trainer.run()
                    # Recarrega a câmera após o treino
                    self.cap = cv2.VideoCapture(0)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
                    # Recarrega os dados atualizados
                    self.labels, self.data, _ = self.db.load_gestures()
                    if self.labels:
                        self.model_manager.train(self.data, self.labels)
                    continue
                elif key == ord("c"):
                    self.current_word = ""
                elif key == ord("t"):
                    self.mode = "treino"
                    self.new_gesture_data = []
                    self.ui.set_error(
                        "Modo Treino ativado. Pressione 'N' para definir nome do gesto."
                    )
                elif key == ord("n") and self.mode == "treino":
                    self.ui.show_text_input = True
                    self.ui.input_prompt = "Digite o nome do gesto:"
                elif key == ord("s") and self.mode == "treino":
                    if self.new_gesture_name and self.new_gesture_data:
                        clean_data = []
                        for l in self.new_gesture_data:
                            if l is not None:
                                clean_data.append(
                                    l.tolist() if hasattr(l, "tolist") else l
                                )
                        self.labels += [self.new_gesture_name] * len(clean_data)
                        self.data += clean_data
                        try:
                            self.db.save_gestures(self.labels, self.data)
                            self.model_manager.train(self.data, self.labels)
                            self.ui.set_error(f"Gesto '{self.new_gesture_name}' salvo!")
                        except Exception as e:
                            self.ui.set_error(f"Erro ao salvar gesto: {e}")
                    self.mode = "teste"
                    self.new_gesture_name = ""
                    self.new_gesture_data = []
                elif key == ord("d") and self.mode == "teste":
                    self.ui.show_text_input = True
                    self.ui.input_prompt = "Digite o nome do gesto para deletar:"
                    self.ui.input_action = "delete"

        self.cap.release()
        cv2.destroyAllWindows()
        return None