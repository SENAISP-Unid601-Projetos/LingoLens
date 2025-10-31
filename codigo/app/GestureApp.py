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
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding='utf-8'
        )

        try:
            self.db = DatabaseManager(CONFIG["db_path"])
            self.model_manager = ModelManager(gesture_type=gesture_type)
            self.gesture_type = gesture_type.lower()
            # manter em maiúsculas igual ao Config
            self.dynamic_letters = set([g.upper() for g in CONFIG.get("dynamic_letters", [])])

            # Carrega dados
            self.static_dict, self.static_labels, self.static_data, self.static_names = self.db.load_gestures(is_dynamic=False)
            self.dyn_dict, self.dyn_labels, self.dyn_data, self.dyn_names = self.db.load_gestures(is_dynamic=True)

            # Embaralhar e treinar se possuir dados
            if (self.static_labels and len(self.static_labels) > 0) or (self.dyn_labels and len(self.dyn_labels) > 0):
                if self.static_labels:
                    self.static_data, self.static_labels = shuffle(self.static_data, self.static_labels, random_state=42)
                if self.dyn_labels:
                    self.dyn_data, self.dyn_labels = shuffle(self.dyn_data, self.dyn_labels, random_state=42)

                trained = self.model_manager.train(
                    static_data=self.static_data,
                    static_labels=self.static_labels,
                    dynamic_data=self.dyn_data,
                    dynamic_labels=self.dyn_labels
                )
                if trained:
                    print(f"[INFO] Modelo treinado: {len(set(self.static_labels)) if self.static_labels else 0} estáticas + {len(set(self.dyn_labels)) if self.dyn_labels else 0} dinâmicas")
                else:
                    print("[AVISO] Erro ao treinar modelos com os dados carregados")
            else:
                print("[INFO] Banco vazio — comece a treinar!")

            # Configura webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Webcam não abriu")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
            self.cap.set(cv2.CAP_PROP_FPS, CONFIG["train_fps"])

            # MediaPipe hands
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=CONFIG.get("max_num_hands",1),
                min_detection_confidence=CONFIG.get("min_detection_confidence",0.5),
                static_image_mode=False,
                min_tracking_confidence=CONFIG.get("min_detection_confidence",0.5)
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

            # Buffers e estados
            self.sequence_buffer = deque(maxlen=CONFIG["sequence_length"])
            self.current_word = ""
            self.mode = "teste"
            self.new_gesture_name = ""
            self.new_gesture_data = []   # lista de frames (estáticas) ou lista de sequências (dinâmicas)
            self.input_text = ""
            self.is_input_active = False
            self.sample_count = 0
            self.prev_landmarks = None
            self.last_prediction_time = time.time()
            self.delete_mode = False
            self.gesture_list = []
            self.selected_index = 0

            # COMANDOS
            self.commands = [
                "T = Iniciar Treino",
                "S = Salvar Gesto (120+ amostras)",
                "C = Limpar Palavra",
                "D = Excluir Gesto",
                "Q = Sair",
                "ESC = Cancelar"
            ]

        except Exception as e:
            logging.error(f"Erro na inicialização: {e}")
            raise

    def is_hand_stable(self, landmarks):
        # simples verificação de variação entre frames (para gestos estáticos)
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return False
        variance = np.var(np.array(landmarks) - np.array(self.prev_landmarks))
        self.prev_landmarks = landmarks
        return variance < 0.005

    def run(self):
        print("\n" + "="*55)
        print("         SISTEMA DE RECONHECIMENTO DE LIBRAS")
        print("="*55)
        for cmd in self.commands:
            print(f"   {cmd}")
        print("="*55 + "\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                image = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                # === DETECÇÃO DA MÃO ===
                if results.multi_hand_landmarks and not self.delete_mode:
                    hand = results.multi_hand_landmarks[0]
                    self.mp_drawing.draw_landmarks(
                        image, hand, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
                    )

                    landmarks = extract_landmarks(hand, image.shape)
                    if landmarks:
                        # adiciona frame ao buffer de sequência (para dinâmicas)
                        self.sequence_buffer.append(landmarks)

                        # === TREINO ESTÁTICO ===
                        if self.mode == "treino" and self.new_gesture_name and self.new_gesture_name not in self.dynamic_letters:
                            # só adiciona se mão está estável
                            if self.is_hand_stable(landmarks):
                                self.new_gesture_data.append(landmarks)
                                self.sample_count += 1
                                if self.sample_count % 10 == 0:
                                    print(f"[INFO] {self.sample_count} amostras estáticas coletadas")

                        # === TREINO DINÂMICO ===
                        elif self.mode == "treino" and self.new_gesture_name and self.new_gesture_name in self.dynamic_letters:
                            # quando atingimos a sequence_length, registra uma sequência
                            if len(self.sequence_buffer) == CONFIG["sequence_length"]:
                                seq = list(self.sequence_buffer)
                                self.new_gesture_data.append(seq)
                                self.sample_count += 1
                                if self.sample_count % 5 == 0:
                                    print(f"[INFO] {self.sample_count} sequências dinâmicas coletadas")
                                # limpar buffer para começar nova sequência
                                self.sequence_buffer.clear()

                        # === TESTE ===
                        elif self.mode == "teste" and len(self.sequence_buffer) == CONFIG["sequence_length"]:
                            # aplicar cooldown entre predições (1s)
                            if time.time() - self.last_prediction_time >= 1.0:
                                seq = list(self.sequence_buffer)
                                pred, prob = self.model_manager.predict(seq)
                                if pred and prob >= CONFIG.get("confidence_threshold", 0.7):
                                    self.current_word += pred
                                    cv2.putText(image, f"{pred}", (10, 60),
                                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
                                    self.last_prediction_time = time.time()
                                # manter o buffer rolando - não limpar aqui para permitir overlapped windows
                                # porém para evitar repetir previsões muito rápidas, usamos cooldown
                                # (sequência irá deslizar com novos frames)

                # === UI NA TELA ===
                cv2.putText(image, f"MODO: {'TREINO' if self.mode == 'treino' else 'TESTE'}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(image, f"PALAVRA: {self.current_word}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                color = (0, 255, 0) if self.sample_count >= CONFIG.get("min_samples_per_class", 120) else (0, 165, 255)
                cv2.putText(image, f"AMOSTRAS: {self.sample_count}/{CONFIG.get('min_samples_per_class',120)}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                if self.mode == "treino" and self.sample_count >= CONFIG.get("min_samples_per_class", 120):
                    cv2.putText(image, "PRESSIONE S PARA SALVAR", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # === COMANDOS NA TELA ===
                cmd_y = image.shape[0] - 180
                cv2.putText(image, "COMANDOS:", (10, cmd_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cmd_y += 30
                for cmd in self.commands:
                    cv2.putText(image, cmd, (20, cmd_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                    cmd_y += 25

                # === ENTRADA DE LETRA ===
                if self.is_input_active:
                    overlay = image.copy()
                    cv2.rectangle(overlay, (50, image.shape[0]//2 - 80), (image.shape[1]-50, image.shape[0]//2 + 80), (0,0,0), -1)
                    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                    cv2.putText(image, f"Letra: {self.input_text}_", (70, image.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

                cv2.imshow("Libras System", image)

                # === TECLAS (100% FUNCIONAIS) ===
                key = cv2.waitKey(1) & 0xFF

                # T = TREINO (inicia input)
                if key == ord('t') or key == ord('T'):
                    self.mode = "treino"
                    self.is_input_active = True
                    self.input_text = ""
                    self.new_gesture_data = []
                    self.sample_count = 0
                    self.sequence_buffer.clear()
                    print("[INFO] Digite a letra (A-Z)")

                # S = SALVAR
                elif key == ord('s') or key == ord('S'):
                    if self.mode == "treino" and self.new_gesture_name:
                        total = len(self.new_gesture_data)
                        min_req = CONFIG.get("min_samples_per_class", 120)
                        if total < min_req:
                            print(f"[AVISO] Faltam {min_req - total} amostras!")
                        else:
                            is_dynamic = self.new_gesture_name in self.dynamic_letters
                            # salvar no banco — DatabaseManager aceita o formato serializável (list)
                            success = self.db.save_gestures(
                                labels=[self.new_gesture_name] * total,
                                data=self.new_gesture_data,
                                gesture_name=self.new_gesture_name,
                                is_dynamic=is_dynamic
                            )
                            if success:
                                if is_dynamic:
                                    # estender listas de treino locais
                                    self.dyn_data.extend(self.new_gesture_data)
                                    self.dyn_labels.extend([self.new_gesture_name] * total)
                                    print(f"[SUCESSO] {self.new_gesture_name} (DINÂMICO) salvo!")
                                else:
                                    self.static_data.extend(self.new_gesture_data)
                                    self.static_labels.extend([self.new_gesture_name] * total)
                                    print(f"[SUCESSO] {self.new_gesture_name} (ESTÁTICO) salvo!")

                                # Re-treinar modelos com os novos dados
                                trained = self.model_manager.train(
                                    static_data=self.static_data,
                                    static_labels=self.static_labels,
                                    dynamic_data=self.dyn_data,
                                    dynamic_labels=self.dyn_labels
                                )
                                if trained:
                                    print("[INFO] Modelos atualizados com o novo gesto")
                                else:
                                    print("[ERRO] Falha ao re-treinar modelos")

                                # reset de estados
                                self.mode = "teste"
                                self.new_gesture_name = ""
                                self.new_gesture_data = []
                                self.sample_count = 0
                                self.sequence_buffer.clear()
                            else:
                                print("[ERRO] Falha no banco")
                    else:
                        print("[INFO] Nada para salvar")

                # C = LIMPAR
                elif key == ord('c') or key == ord('C'):
                    self.current_word = ""
                    print("[INFO] Palavra limpa")

                # D = EXCLUIR
                elif key == ord('d') or key == ord('D'):
                    self.delete_mode = True
                    all_gestures = self.db.list_gestures(is_dynamic=False) + self.db.list_gestures(is_dynamic=True)
                    self.gesture_list = sorted(set(all_gestures))
                    self.selected_index = 0
                    print(f"[INFO] Modo exclusão: {len(self.gesture_list)} gestos")

                # Q = SAIR
                elif key == ord('q') or key == ord('Q'):
                    print("[INFO] Saindo...")
                    break

                # === ENTRADA DE LETRA (DURANTE TREINO) ===
                elif self.is_input_active:
                    if key == 13:  # ENTER
                        self.new_gesture_name = self.input_text.upper()
                        if self.new_gesture_name.isalpha() and len(self.new_gesture_name) == 1:
                            self.is_input_active = False
                            self.input_text = ""
                            # reset contadores para nova coleta
                            self.new_gesture_data = []
                            self.sample_count = 0
                            self.sequence_buffer.clear()
                            print(f"[INFO] Gravando: {self.new_gesture_name}")
                        else:
                            print("[ERRO] Digite UMA letra!")
                            self.input_text = ""
                    elif key == 27:  # ESC
                        self.is_input_active = False
                        self.mode = "teste"
                        print("[INFO] Cancelado")
                    elif key == 8:  # BACKSPACE
                        self.input_text = self.input_text[:-1]
                    elif 65 <= key <= 90 or 97 <= key <= 122:
                        if len(self.input_text) < 1:
                            self.input_text += chr(key).upper()

        except Exception as e:
            logging.error(f"Erro: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.db.close()
            print("[INFO] Sistema encerrado")
