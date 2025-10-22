import cv2
import mediapipe as mp
import logging
from Config import CONFIG
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Utils import extract_landmarks


class GestureApp:
    def __init__(self, db: DatabaseManager):
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.db = db
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
        
        # Interface igual à de movimentos
        self.current_word = ""
        self.mode = "teste"
        self.new_gesture_name = ""
        self.new_gesture_data = []
        self.samples_count = 0  # Contador de amostras
        
        # Estados da UI
        self.show_text_input = False
        self.input_text = ""
        self.input_prompt = ""
        self.error_message = ""
        self.error_message_timer = 0
        self.error_message_duration = 90
        self.input_action = ""

    def draw_ui(self, image, title, cooldown, word):
        """Desenha interface igual à de movimentos"""
        height, width = image.shape[:2]
        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        # Título da tela
        cv2.putText(
            image,
            title,
            (int(10), int(35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            outline_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            title,
            (int(10), int(35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Modo atual e contador
        if self.mode == "treino":
            mode_text = f"Modo: TREINO | Amostras: {self.samples_count}"
        else:
            mode_text = f"Modo: TESTE"
            
        cv2.putText(
            image,
            mode_text,
            (int(10), int(70)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Palavra atual
        cv2.putText(
            image,
            word,
            (width // 2 - 100, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

        # Instruções baseadas no modo
        if self.mode == "teste":
            instructions = "Q:Sair C:Limpar T:Treino D:Deletar M:Movimentos"
        else:
            instructions = "Q:Sair S:Salvar N:Nome ESC:Cancelar M:Movimentos"
        
        cv2.putText(
            image,
            instructions,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # Mensagem de erro
        if self.error_message:
            cv2.putText(
                image,
                self.error_message,
                (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 50, 50),
                2,
                cv2.LINE_AA,
            )

        # Input de texto
        if self.show_text_input:
            # Fundo para input
            overlay = image.copy()
            cv2.rectangle(overlay, (50, 100), (width - 50, 200), (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
            
            # Prompt
            cv2.putText(
                image,
                self.input_prompt,
                (70, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            
            # Texto digitado
            cv2.putText(
                image,
                self.input_text + "_",
                (70, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return image

    def set_error(self, text):
        self.error_message = text
        self.error_message_timer = 0

    def run(self):
        print(
            "[INFO] Teclas: Q=Sair | C=Limpar | T=Treino | S=Salvar | N=Nome | D=Deletar | M:Movimentos"
        )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.set_error("Falha ao capturar frame da câmera!")
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
                            if pred is not None and prob >= CONFIG["confidence_threshold"]:
                                self.current_word += pred
                                self.set_error(f"Reconhecido: {pred} ({prob:.2f})")
                        elif self.mode == "treino" and self.new_gesture_name:
                            self.new_gesture_data.append(landmarks)
                            self.samples_count = len(self.new_gesture_data)  # Atualiza contador

            # Desenha UI
            title = "Reconhecimento de Gestos" if self.mode == "teste" else "Treino de Gestos"
            image = self.draw_ui(image, title, 0, self.current_word)
            cv2.imshow("GestureApp", image)

            key = cv2.waitKey(1) & 0xFF

            # Processar input de texto
            if self.show_text_input:
                if key == 13:  # Enter
                    gesture_name = self.input_text.upper()
                    self.input_text = ""
                    self.show_text_input = False
                    
                    if self.input_action == "delete":
                        success = self.db.delete_gesture(gesture_name)
                        if success:
                            self.labels, self.data, _ = self.db.load_gestures()
                            if self.labels:
                                self.model_manager.train(self.data, self.labels)
                            self.set_error(f"Gesto '{gesture_name}' deletado com sucesso!")
                        else:
                            self.set_error(f"Erro ao deletar gesto '{gesture_name}'.")
                        self.input_action = ""
                    else:
                        self.new_gesture_name = gesture_name
                        self.set_error(f"Gesto '{self.new_gesture_name}' definido. Pressione 'S' para salvar.")
                        
                elif key == 8:  # Backspace
                    self.input_text = self.input_text[:-1]
                elif 32 <= key <= 126:  # Caracteres imprimíveis
                    self.input_text += chr(key)
                elif key == 27:  # ESC cancela input
                    self.show_text_input = False
                    self.input_text = ""
                    self.input_action = ""

            else:
                # Teclas gerais
                if key == 27:  # ESC cancela modo treino
                    if self.mode == "treino":
                        self.mode = "teste"
                        self.new_gesture_name = ""
                        self.new_gesture_data = []
                        self.samples_count = 0
                        self.set_error("Modo treino cancelado.")
                    # Não sai do app com ESC no modo teste

                elif key == ord("q"):
                    break
                    
                elif key == ord("m"):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return "movement"
                    
                elif key == ord("c"):
                    self.current_word = ""
                    self.set_error("Texto limpo!")
                    
                elif key == ord("t"):
                    self.mode = "treino"
                    self.new_gesture_data = []
                    self.samples_count = 0
                    self.set_error("Modo Treino ativado. Pressione 'N' para definir nome do gesto.")
                    
                elif key == ord("n") and self.mode == "treino":
                    self.show_text_input = True
                    self.input_prompt = "Digite o nome do gesto:"
                    self.input_action = ""
                    
                elif key == ord("s") and self.mode == "treino":
                    if self.new_gesture_name and self.new_gesture_data:
                        clean_data = []
                        for l in self.new_gesture_data:
                            if l is not None:
                                clean_data.append(l.tolist() if hasattr(l, "tolist") else l)
                        
                        if clean_data:
                            # Salva apenas os novos dados
                            temp_labels = [self.new_gesture_name] * len(clean_data)
                            temp_data = clean_data
                            
                            # Carrega dados existentes
                            existing_labels, existing_data, _ = self.db.load_gestures()
                            
                            # Combina com os novos
                            updated_labels = existing_labels + temp_labels
                            updated_data = existing_data + temp_data
                            
                            try:
                                self.db.save_gestures(updated_labels, updated_data)
                                self.labels, self.data, _ = self.db.load_gestures()
                                self.model_manager.train(self.data, self.labels)
                                self.set_error(f"Gesto '{self.new_gesture_name}' salvo com {self.samples_count} amostras!")
                            except Exception as e:
                                self.set_error(f"Erro ao salvar gesto: {e}")
                        else:
                            self.set_error("Nenhum dado válido para salvar!")
                    else:
                        self.set_error("Defina um nome e capture gestos primeiro!")
                    
                    self.mode = "teste"
                    self.new_gesture_name = ""
                    self.new_gesture_data = []
                    self.samples_count = 0
                    
                elif key == ord("d") and self.mode == "teste":
                    self.show_text_input = True
                    self.input_prompt = "Digite o nome do gesto para deletar:"
                    self.input_action = "delete"

            # Atualiza timer da mensagem de erro
            if self.error_message_timer > 0:
                self.error_message_timer -= 1
            else:
                self.error_message = ""

        self.cap.release()
        cv2.destroyAllWindows()
        return None