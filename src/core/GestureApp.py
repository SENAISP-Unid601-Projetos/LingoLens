import cv2
import mediapipe as mp
import logging
import numpy as np
import time
from config.Config import CONFIG, validate_gesture_type
from src.database.Database_manager import DatabaseManager
from src.models.Model_manager import ModelManager
from src.utils.Extract_landmarks import extract_landmarks
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

            # Treino automático ao iniciar
            self.model_manager.train(
                static_data=self.static_data,
                static_labels=self.static_labels,
                dynamic_data=self.dyn_data,
                dynamic_labels=self.dyn_labels
            )
            print("[INFO] Modelos re-treinados ao iniciar o aplicativo.")

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
            self.sequence_buffer = deque(maxlen=38)
            self.prev_landmarks = None
            self.motion_frames = 0
            self.hand_still_frames = 0
            self.in_motion = False 
            self.smooth_landmarks = None
            self.smoothing_factor = 0.65
            self.motion_threshold = 0.004
            self.stable_threshold = 10

            # NOVO SISTEMA TOGGLE T (GRAVAÇÃO DINÂMICA)
            self.is_recording_dynamic = False
            self.recording_buffer = []
            self.last_toggle_time = 0
            self.last_displayed_letter = None          
            self.last_letter_change_time = 0           
            self.min_time_between_same_letter = 1.5     
            self.min_time_between_any_letter = 0.8
            self.toggle_debounce = 0.4

            # Estado geral
            self.current_word = ""
            self.last_pred_time = 0
            self.word_pause_threshold = 1.0
            self.mode = "teste"
            self.new_gesture_name = ""
            self.new_gesture_data = []
            self.sample_count = 0
            self.is_input_active = False

            # === BARRA DE PROGRESSO (adiciona aqui) ===
            self.training_in_progress = False
            self.training_progress = 0.0
            self.training_status_text = ""

            # Mensagens
            self.message = ""
            self.message_time = 0
            self.show_letters = False
            self.show_letters_time = 0
            self.letters_text = ""

            # Força modo teste e predição mais rápida na web
            self.mode = "teste"
            self.min_time_between_any_letter = 0.1   # era 1.0 → agora prediz muito mais rápido

        except Exception as e:
            logging.error(f"Erro na inicialização: {e}")
            raise

    # ===========================================================
    def run(self):
        print("\n=== SISTEMA LIBRAS 2025 - TOGGLE T + DELETE 100% FUNCIONAL ===")
        print("  T = Gravar dinâmico (toggle) | S = Salvar | D = Deletar | L = Listar | Q = Sair")
        print("=============================================================\n")

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

                    # Suavização
                    lm = np.array(landmarks, dtype=float)
                    if self.smooth_landmarks is None:
                        self.smooth_landmarks = lm
                    else:
                        self.smooth_landmarks = (
                            self.smoothing_factor * lm +
                            (1 - self.smoothing_factor) * self.smooth_landmarks
                        )
                    landmarks = self.smooth_landmarks.tolist()

                    self.sequence_buffer.append(landmarks)

                    if self.mode == "treino" and self.new_gesture_name:
                        self._capture_training_sample(landmarks, frame)
                    elif self.mode == "teste":
                        self._handle_motion_state(landmarks, frame)

                self._draw_ui(frame)
                #key = cv2.waitKey(1) & 0xFF
                #if self._handle_key(key):
                #    break

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
        self.prev_landmarks = landmarks.copy()

        current_time = time.time()

        # Detecção de movimento
        if variance > self.motion_threshold:
            self.motion_frames += 1
            self.hand_still_frames = 0
        else:
            self.hand_still_frames += 1
            self.motion_frames = max(0, self.motion_frames - 1)

        # Início de gesto dinâmico
        if not self.in_motion and self.motion_frames > 6:
            self.in_motion = True
            self.sequence_buffer.clear()

        # === RECONHECIMENTO DINÂMICO ===
        if self.in_motion and self.hand_still_frames > 10 and len(self.sequence_buffer) >= 38:
            seq = list(self.sequence_buffer)[-38:]
            pred, prob = self.model_manager.predict(seq)
            thresh_dyn = CONFIG.get("confidence_threshold_dynamic", 0.62)
            if pred and prob >= thresh_dyn:
                # Filtro: só adiciona se for diferente da última OU se passou tempo suficiente
                if (self.last_displayed_letter != pred or 
                    current_time - self.last_letter_change_time > self.min_time_between_same_letter):
                    
                    if current_time - self.last_letter_change_time > self.min_time_between_any_letter:
                        self.current_word += pred
                        self.last_displayed_letter = pred
                        self.last_letter_change_time = current_time
                        print(f"[DINÂMICO] {pred} ({prob:.2f})")
            self._reset_motion_state()
            return

        # === RECONHECIMENTO ESTÁTICO ===
        if not self.in_motion and self.hand_still_frames > self.stable_threshold:
            frame_data = self.sequence_buffer[-1]
            pred, prob = self.model_manager.predict(frame_data)
            thresh_static = CONFIG.get("confidence_threshold_static", 0.78)
            if pred and prob >= thresh_static:
                # Filtro inteligente: evita repetição rápida da mesma letra
                if (self.last_displayed_letter != pred or 
                    current_time - self.last_letter_change_time > self.min_time_between_same_letter):
                    
                    if current_time - self.last_letter_change_time > self.min_time_between_any_letter:
                        # Adiciona espaço só se não for a primeira letra após pausa
                        if self.current_word and current_time - self.last_pred_time > self.word_pause_threshold:
                            self.current_word += " "
                        self.current_word += pred
                        self.last_displayed_letter = pred
                        self.last_letter_change_time = current_time
                        self.last_pred_time = current_time
                        print(f"[ESTÁTICO] {pred} ({prob:.2f})")
                        self.hand_still_frames = 0  # evita repetição imediata

    def _reset_motion_state(self):
        self.in_motion = False
        self.motion_frames = 0
        self.hand_still_frames = 0
        self.sequence_buffer.clear()

    # ===========================================================
    def _capture_training_sample(self, landmarks, frame):
        if self.new_gesture_name not in self.dynamic_letters:
            # ESTÁTICO
            variance = np.var(np.array(landmarks) - np.array(self.prev_landmarks)) if self.prev_landmarks is not None else 0
            self.prev_landmarks = landmarks.copy()
            if variance < 0.012:
                self.new_gesture_data.append(landmarks)
                self.sample_count += 1
                if self.sample_count % 10 == 0:
                    print(f"[ESTÁTICO] {self.sample_count} amostras coletadas")
            return

        # DINÂMICO — TOGGLE T
        status_text = "APERTE T PARA GRAVAR"
        color = (0, 255, 255)

        if self.is_recording_dynamic:
            status_text = f"GRAVANDO: {len(self.recording_buffer)}/38"
            color = (0, 0, 255)
            self.recording_buffer.append(landmarks)

            if len(self.recording_buffer) >= 38:
                seq = self.recording_buffer[-38:]
                self.new_gesture_data.append(seq)
                self.sample_count += 1
                print(f"[DINÂMICO] {self.sample_count} sequências coletadas → 38 frames")
                self.recording_buffer.clear()
                self.is_recording_dynamic = False

        cv2.putText(frame, status_text, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        self.prev_landmarks = landmarks.copy()

    # ===========================================================
    def _draw_ui(self, frame):
        cv2.putText(frame, f"Modo: {self.mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Palavra: {self.current_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.is_input_active:
            cv2.putText(frame, "Digite a letra (A-Z)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif self.mode == "treino" and self.new_gesture_name:
            total_min = CONFIG.get("min_samples_per_class", 120)
            cv2.putText(frame, f"Gravando {self.new_gesture_name} → {self.sample_count}/{total_min}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if self.message and (time.time() - self.message_time) < 2:
            cv2.putText(frame, self.message, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self.show_letters and (time.time() - self.show_letters_time) < 4:
            cv2.putText(frame, self.letters_text, (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

       # cv2.imshow("Libras 2025 - TOGGLE T + DELETE OK", frame)

    # ===========================================================
    def _handle_key(self, key):
        # Digitar letra (treino ou delete)
        if self.is_input_active and (65 <= key <= 90 or 97 <= key <= 122):
            letra = chr(key).upper()
            if self.mode == "treino":
                self.new_gesture_name = letra
                self.new_gesture_data = []
                self.sample_count = 0
                self.is_recording_dynamic = False
                self.recording_buffer.clear()
                print(f"[INFO] Coletando gesto: {letra}")
            elif self.mode == "delete":
                # === DELETAR CORRETO (estática + dinâmica) ===
                deleted_static = self.db.delete_gesture(letra, is_dynamic=False)
                deleted_dynamic = self.db.delete_gesture(letra, is_dynamic=True)

                if deleted_static or deleted_dynamic:
                    self._show_message(f"Letra {letra} apagada!")
                    print(f"[SUCESSO] '{letra}' removida do banco")
                else:
                    self._show_message(f"Letra {letra} não encontrada")
                    print(f"[AVISO] '{letra}' não existe")

                # Recarrega tudo e retreina
                self.static_dict, self.static_labels, self.static_data, self.static_names = self.db.load_gestures(is_dynamic=False)
                self.dyn_dict, self.dyn_labels, self.dyn_data, self.dyn_names = self.db.load_gestures(is_dynamic=True)
                self.model_manager.train(
                    static_data=self.static_data, static_labels=self.static_labels,
                    dynamic_data=self.dyn_data, dynamic_labels=self.dyn_labels
                )

                self.mode = "teste"
            self.is_input_active = False
            return False

        if key == 27:  # ESC
            self.is_input_active = False
            self.mode = "teste"
            self.new_gesture_name = ""
            self.new_gesture_data = []
            self.sample_count = 0
            self.is_recording_dynamic = False
            self.recording_buffer.clear()
            self._show_message("Cancelado")
            return False

        if key == ord("q") or key == ord("Q"):
            print("[INFO] Saindo...")
            return True

        if key == ord("c") or key == ord("C"):
            self.current_word = ""
            print("[INFO] Palavra limpa")

        if key == ord("t") or key == ord("T"):
            if self.new_gesture_name and self.new_gesture_name in self.dynamic_letters:
                current_time = time.time()
                if current_time - self.last_toggle_time > self.toggle_debounce:
                    self.last_toggle_time = current_time
                    self.is_recording_dynamic = not self.is_recording_dynamic
                    if self.is_recording_dynamic:
                        self.recording_buffer.clear()
                        print("[GRAVANDO] → Iniciado (T novamente ou 38 frames)")
                    else:
                        print("[GRAVANDO] → Parado")
            elif not self.new_gesture_name:
                self.mode = "treino"
                self.is_input_active = True
                print("[INFO] Digite a letra (A-Z)")

        if key == ord("l") or key == ord("L"):
            static = sorted(set(self.static_labels)) if self.static_labels else []
            dynamic = sorted(set(self.dyn_labels)) if self.dyn_labels else []
            self.letters_text = f"E: {', '.join(static) if static else 'Nenhuma'} | D: {', '.join(dynamic) if dynamic else 'Nenhuma'}"
            self.show_letters = True
            self.show_letters_time = time.time()

        if key == ord("d") or key == ord("D"):
            self.mode = "delete"
            self.is_input_active = True
            print("[INFO] Digite a letra para apagar")

        if key == ord("s") or key == ord("S"):
            if self.new_gesture_name and len(self.new_gesture_data) >= 120:
                is_dyn = self.new_gesture_name in self.dynamic_letters
                success = self.db.save_gestures(
                    labels=[self.new_gesture_name] * len(self.new_gesture_data),
                    data=self.new_gesture_data,
                    gesture_name=self.new_gesture_name,
                    is_dynamic=is_dyn
                )
                if success:
                    self._show_message(f"{self.new_gesture_name} salvo!")
                    print(f"[SUCESSO] {self.new_gesture_name} salvo ({len(self.new_gesture_data)} amostras)")
                    # Recarrega e retreina
                    self.static_dict, self.static_labels, self.static_data, self.static_names = self.db.load_gestures(is_dynamic=False)
                    self.dyn_dict, self.dyn_labels, self.dyn_data, self.dyn_names = self.db.load_gestures(is_dynamic=True)
                    self.model_manager.train(
                        static_data=self.static_data, static_labels=self.static_labels,
                        dynamic_data=self.dyn_data, dynamic_labels=self.dyn_labels
                    )
                self.mode = "teste"
                self.new_gesture_name = ""
                self.new_gesture_data = []
                self.sample_count = 0
                self.is_recording_dynamic = False
            else:
                self._show_message("Poucas amostras!")

        return False

    def _show_message(self, text):
        self.message = text
        self.message_time = time.time()