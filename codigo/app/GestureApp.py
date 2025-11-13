import cv2
import mediapipe as mp
import logging
import numpy as np
import time
from Config import CONFIG, validate_gesture_type
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Extract_landmarks import extract_landmarks
from sklearn.utils import shuffle
from collections import deque


class GestureApp:
    def __init__(self, gesture_type="letter"):
        validate_gesture_type(gesture_type)
        print("[INFO] Inicializando sistema de reconhecimento de Libras...")

        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8"
        )

        try:
            self.db = DatabaseManager(CONFIG["db_path"])
            self.model_manager = ModelManager(gesture_type=gesture_type)
            self.gesture_type = gesture_type.lower()
            self.dynamic_letters = set([g.upper() for g in CONFIG.get("dynamic_letters", [])])

            # Carregar dados existentes
            self.static_dict, self.static_labels, self.static_data, self.static_names = self.db.load_gestures(is_dynamic=False)
            self.dyn_dict, self.dyn_labels, self.dyn_data, self.dyn_names = self.db.load_gestures(is_dynamic=True)

            # Treinar modelos se houver dados
            if self.static_labels or self.dyn_labels:
                if self.static_labels:
                    self.static_data, self.static_labels = shuffle(self.static_data, self.static_labels, random_state=42)
                if self.dyn_labels:
                    self.dyn_data, self.dyn_labels = shuffle(self.dyn_data, self.dyn_labels, random_state=42)
                self.model_manager.train(
                    static_data=self.static_data,
                    static_labels=self.static_labels,
                    dynamic_data=self.dyn_data,
                    dynamic_labels=self.dyn_labels
                )

            # Webcam
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])

            # MediaPipe
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                min_detection_confidence=CONFIG.get("min_detection_confidence", 0.5),
                min_tracking_confidence=CONFIG.get("min_tracking_confidence", 0.5)
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

            # Buffers e controle
            self.sequence_buffer = deque(maxlen=CONFIG["sequence_length"])
            self.prev_landmarks = None
            self.motion_frames = 0
            self.hand_still_frames = 0
            self.in_motion = False
            self.motion_threshold = 0.002
            self.stable_threshold = 15

            # Estado geral
            self.current_word = ""
            self.mode = "teste"
            self.new_gesture_name = ""
            self.new_gesture_data = []
            self.sample_count = 0
            self.is_input_active = False

            # Mensagens visuais
            self.message = ""
            self.message_time = 0

            # Letras treinadas (exibição com L)
            self.show_letters = False
            self.show_letters_time = 0
            self.letters_text = ""

            self.commands = [
                "T = Treinar novo gesto",
                "S = Salvar gesto",
                "C = Limpar palavra",
                "L = Listar letras treinadas",
                "D = Deletar letra",
                "ESC = Cancelar ação",
                "Q = Sair"
            ]

        except Exception as e:
            logging.error(f"Erro na inicialização: {e}")
            raise

    # ===========================================================
    def run(self):
        print("\n=== SISTEMA DE RECONHECIMENTO DE LIBRAS ===")
        for cmd in self.commands:
            print("  " + cmd)
        print("==========================================\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = extract_landmarks(hand, frame.shape)
                    if not landmarks:
                        continue

                    self.sequence_buffer.append(landmarks)

                    # Pausa reconhecimento durante treino/delete
                    if self.mode == "treino" and self.new_gesture_name:
                        self._capture_training_sample(landmarks, frame)
                    elif self.mode == "teste":
                        self._handle_motion_state(landmarks, frame)

                self._draw_ui(frame)
                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    break

        except Exception as e:
            logging.error(f"Erro: {e}")
            print(f"[ERRO] {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.db.close()

    # ===========================================================
    def _handle_motion_state(self, landmarks, frame):
        variance = np.var(np.array(landmarks) - np.array(self.prev_landmarks)) if self.prev_landmarks is not None else 0
        self.prev_landmarks = landmarks

        if variance > self.motion_threshold:
            self.motion_frames += 1
            self.hand_still_frames = 0
        else:
            self.hand_still_frames += 1
            self.motion_frames = max(0, self.motion_frames - 1)

        if not self.in_motion and self.motion_frames > 3:
            self.in_motion = True
            self.sequence_buffer.clear()

        if self.in_motion and self.hand_still_frames > 5 and len(self.sequence_buffer) == CONFIG["sequence_length"]:
            seq = list(self.sequence_buffer)
            pred, prob = self.model_manager.predict(seq)
            if pred and prob >= CONFIG.get("confidence_threshold", 0.7):
                self.current_word += pred
                print(f"[DINÂMICO DETECTADO] {pred} ({prob:.2f})")
            self._reset_motion_state()

        if not self.in_motion and self.hand_still_frames > self.stable_threshold:
            if len(self.sequence_buffer) > 0:
                frame_data = self.sequence_buffer[-1]
                pred, prob = self.model_manager.predict(frame_data)
                if pred and prob >= CONFIG.get("confidence_threshold", 0.7):
                    self.current_word += pred
                    print(f"[ESTÁTICO DETECTADO] {pred} ({prob:.2f})")
                    self.hand_still_frames = 0

    def _reset_motion_state(self):
        self.in_motion = False
        self.motion_frames = 0
        self.hand_still_frames = 0
        self.sequence_buffer.clear()

    # ===========================================================
    def _capture_training_sample(self, landmarks, frame):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return

        if self.new_gesture_name in self.dynamic_letters:
            if len(self.sequence_buffer) == CONFIG["sequence_length"]:
                seq = list(self.sequence_buffer)
                self.new_gesture_data.append(seq)
                self.sample_count += 1
                print(f"[DINÂMICO] {self.sample_count} sequências coletadas")
                self.sequence_buffer.clear()
        else:
            variance = np.var(np.array(landmarks) - np.array(self.prev_landmarks))
            self.prev_landmarks = landmarks
            if variance < self.motion_threshold * 2:
                self.new_gesture_data.append(landmarks)
                self.sample_count += 1
                if self.sample_count % 10 == 0:
                    print(f"[ESTÁTICO] {self.sample_count} amostras coletadas")

    # ===========================================================
    def _draw_ui(self, frame):
        cv2.putText(frame, f"Modo: {self.mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Palavra: {self.current_word}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.is_input_active and self.mode == "treino":
            cv2.putText(frame, "Digite a letra (A-Z)", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif self.mode == "treino" and self.new_gesture_name:
            total_min = CONFIG.get("min_samples_per_class", 120)
            cv2.putText(frame,
                        f"Gravando {self.new_gesture_name}... {self.sample_count}/{total_min}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        elif self.is_input_active and self.mode == "delete":
            cv2.putText(frame, "Digite a letra para apagar (A-Z)", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

        # Mostra mensagens temporárias
        if self.message and (time.time() - self.message_time) < 2:
            cv2.putText(frame, self.message, (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Mostra lista de letras treinadas (por 4s)
        if self.show_letters and (time.time() - self.show_letters_time) < 4:
            cv2.putText(frame, self.letters_text, (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            self.show_letters = False

        cv2.imshow("Libras System", frame)

    # ===========================================================
    def _handle_key(self, key):
        # Entrada de letra em modo treino
        if self.is_input_active and self.mode == "treino" and (65 <= key <= 90 or 97 <= key <= 122):
            self.new_gesture_name = chr(key).upper()
            self.is_input_active = False
            self.mode = "treino"
            self.new_gesture_data = []
            self.sample_count = 0
            print(f"[INFO] Coletando gesto: {self.new_gesture_name}")
            return False

        # Entrada de letra em modo delete
       # 🔧 Entrada de letra em modo delete (corrigido: apaga estática e dinâmica)
        if self.is_input_active and self.mode == "delete" and (65 <= key <= 90 or 97 <= key <= 122):
            letra = chr(key).upper()
            self.is_input_active = False
            self.mode = "teste"

            deleted_static = self.db.delete_gesture(letra, is_dynamic=False)
            deleted_dynamic = self.db.delete_gesture(letra, is_dynamic=True)

            if deleted_static or deleted_dynamic:
                self._show_message(f"Letra {letra} deletada!")
                print(f"[SUCESSO] Letra '{letra}' removida (estática/dinâmica).")
            else:
                self._show_message(f"Letra {letra} não encontrada.")
                print(f"[AVISO] Letra '{letra}' não encontrada nas bases.")

            return False


        # ======= COMANDOS =======
        if key == 27:  # ESC
            self.is_input_active = False
            if self.mode in ["treino", "delete"]:
                self.mode = "teste"
                self.new_gesture_data = []
                self.new_gesture_name = ""
                self.sample_count = 0
                self.sequence_buffer.clear()
                self._show_message("Ação cancelada!")
                print("[INFO] Ação cancelada.")
            return False

        if key == ord("q") or key == ord("Q"):
            print("[INFO] Saindo...")
            return True

        elif key == ord("c") or key == ord("C"):
            self.current_word = ""
            print("[INFO] Palavra limpa")

        elif key == ord("t") or key == ord("T"):
            self.mode = "treino"
            self.is_input_active = True
            self.new_gesture_name = ""
            self.new_gesture_data = []
            self.sample_count = 0
            self.sequence_buffer.clear()
            print("[INFO] Digite a letra (A-Z)")

        elif key == ord("l") or key == ord("L"):
            static_gestures = sorted(set(self.static_labels))
            dynamic_gestures = sorted(set(self.dyn_labels))
            static_text = ", ".join(static_gestures) if static_gestures else "Nenhuma"
            dynamic_text = ", ".join(dynamic_gestures) if dynamic_gestures else "Nenhuma"
            self.letters_text = f"Estaticas: {static_text} | Dinamicas: {dynamic_text}"
            self.show_letters = True
            self.show_letters_time = time.time()
            print("\n=== Letras treinadas ===")
            print(f"Estaticas: {static_text}")
            print(f"Dinamicas: {dynamic_text}")
            print("========================\n")

        elif key == ord("d") or key == ord("D"):
            self.mode = "delete"
            self.is_input_active = True
            print("[INFO] Digite a letra que deseja apagar (A-Z)")

        elif key == ord("s") or key == ord("S"):
            if self.new_gesture_name and len(self.new_gesture_data) >= CONFIG.get("min_samples_per_class", 120):
                is_dynamic = self.new_gesture_name in self.dynamic_letters
                success = self.db.save_gestures(
                    labels=[self.new_gesture_name] * len(self.new_gesture_data),
                    data=self.new_gesture_data,
                    gesture_name=self.new_gesture_name,
                    is_dynamic=is_dynamic
                )

                if success:
                    self._show_message(f"{self.new_gesture_name} salvo!")
                    print(f"[SUCESSO] {self.new_gesture_name} salvo com {len(self.new_gesture_data)} amostras")

                    # 🔹 Recarrega todos os dados do banco
                    self.static_dict, self.static_labels, self.static_data, self.static_names = self.db.load_gestures(is_dynamic=False)
                    self.dyn_dict, self.dyn_labels, self.dyn_data, self.dyn_names = self.db.load_gestures(is_dynamic=True)

                    # 🔹 Reentreina com tudo junto
                    self.model_manager.train(
                        static_data=self.static_data,
                        static_labels=self.static_labels,
                        dynamic_data=self.dyn_data,
                        dynamic_labels=self.dyn_labels
                    )

                self.mode = "teste"
                self.sample_count = 0
                self.new_gesture_data = []
                self.new_gesture_name = ""
                self.sequence_buffer.clear()
            else:
                self._show_message("Poucas amostras!")
                print("[ERRO] Poucas amostras ou nenhum gesto definido")


        return False

    # ===========================================================
    def _show_message(self, text):
        self.message = text
        self.message_time = time.time()
