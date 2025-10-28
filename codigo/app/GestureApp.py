import cv2
import mediapipe as mp
import logging
import numpy as np
import time
import os
from Config import CONFIG, validate_gesture_type
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Preprocess_landmarks import preprocess_landmarks
from Dynamic_gesture_recognizer import DynamicGestureRecognizer
from sklearn.utils import shuffle

# Suprimir avisos do TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class GestureApp:
    def __init__(self, gesture_type="letter"):
        validate_gesture_type(gesture_type)
        print("[INFO] Inicializando GestureApp...")
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self._check_cpu_compatibility()

        try:
            self.db = DatabaseManager(CONFIG["db_path"])
            self.model_manager = ModelManager(gesture_type=gesture_type)
            self.ui = UIManager()
            self.dynamic_recognizer = DynamicGestureRecognizer(CONFIG)
            self.gesture_type = gesture_type.lower()

            # === CARREGAR MODELOS DO BANCO AO INICIAR ===
            if not self.model_manager.load_model():
                print("[INFO] Nenhum modelo estático treinado ainda.")
            else:
                print(f"[INFO] Modelo estático carregado com {len(self.model_manager.model.classes_)} classes")

            if not self.dynamic_recognizer.load_model_lstm():
                print("[INFO] Modelo LSTM não encontrado. Treine gestos dinâmicos.")
            else:
                print(f"[INFO] Modelo dinâmico carregado com {len(self.dynamic_recognizer.classes)} classes")

            # Carregar dados estáticos do banco
            self.data_dict, self.labels, self.data, self.gesture_names = self.db.load_gestures(gesture_type=self.gesture_type)
            if self.labels and not self.model_manager.trained:
                success = self.model_manager.train(self.data, self.labels)
                if success:
                    num_classes = len(set(self.labels))
                    print(f"[INFO] Modelo estático treinado com {num_classes} gesto(s): {set(self.labels)}")
                else:
                    print("[WARNING] Falha ao treinar modelo estático.")
            elif not self.labels:
                print("[INFO] Nenhum gesto estático treinado ainda.")

            # Webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Não foi possível abrir a webcam")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
            self.cap.set(cv2.CAP_PROP_FPS, CONFIG["train_fps"])

            # MediaPipe
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=CONFIG["max_num_hands"],
                min_detection_confidence=CONFIG["min_detection_confidence"],
                static_image_mode=False,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

            # Estado
            self.current_word = ""
            self.mode = "train_static"
            self.new_gesture_name = ""
            self.new_gesture_data = []
            self.input_text = ""
            self.is_input_active = False
            self.sample_count = 0
            self.frame_count = 0
            self.prev_landmarks = None
            self.last_prediction_time = time.time()
            self.delete_mode = False
            self.gesture_list = []
            self.selected_index = 0
            self.hand_stable = False
            self.landmark_history = []
            self.variance = 0.0
            self.saving_status = ""
            self.recording_key_pressed = False

        except Exception as e:
            logging.error(f"Erro na inicialização: {e}")
            raise

    def _check_cpu_compatibility(self):
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            flags = info.get('flags', [])
            if 'ssse3' in flags or 'pni' in flags:
                flags.append('sse3')
            required = ['sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2', 'avx', 'avx2', 'fma']
            missing = [f for f in required if f not in flags]
            if missing:
                print(f"[WARNING] CPU não suporta: {', '.join(missing)}")
            else:
                print("[INFO] CPU compatível com todas as instruções.")
        except:
            print("[WARNING] Instale 'py-cpuinfo' para verificar CPU.")

    def normalize_landmarks(self, landmarks):
        try:
            wrist = np.array(landmarks[0:3])
            middle_finger_base = np.array(landmarks[27:30])
            scale = np.linalg.norm(wrist - middle_finger_base)
            if scale == 0:
                return landmarks
            return [coord / scale for coord in landmarks]
        except:
            return landmarks

    def is_hand_stable(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            self.landmark_history = [landmarks]
            return False
        try:
            normalized = self.normalize_landmarks(landmarks)
            self.landmark_history.append(normalized)
            if len(self.landmark_history) > 3:
                self.landmark_history.pop(0)
            smoothed = np.mean(self.landmark_history, axis=0)
            key_indices = [0, 5, 9, 13, 17]
            key_landmark_indices = [i*3 for i in key_indices] + [i*3+1 for i in key_indices] + [i*3+2 for i in key_indices]
            key_landmarks = [smoothed[i] for i in key_landmark_indices]
            key_prev = [self.prev_landmarks[i] for i in key_landmark_indices]
            self.variance = np.var(np.array(key_landmarks) - np.array(key_prev))
            self.prev_landmarks = smoothed
            return self.variance < CONFIG["movement_threshold"]
        except:
            return False

    def run(self):
        print("[INFO] Teclas: Q=Sair C=Limpar T=Treino Estático ou Treinar M=Treino Dinâmico R=Reconhecimento S=Salvar (segure para dinâmico) D=Excluir")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                if self.frame_count % (CONFIG["target_fps"] // CONFIG["train_fps"]) != 0 and self.mode in ["train_static", "train_dynamic"]:
                    continue

                image = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                self.hand_stable = False
                self.saving_status = ""

                if results.multi_hand_landmarks and not self.delete_mode:
                    for hand in results.multi_hand_landmarks:
                        if self.mode in ["train_static", "train_dynamic"]:
                            self.mp_drawing.draw_landmarks(
                                image, hand, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                            )
                        landmarks = preprocess_landmarks(hand, image.shape)
                        if landmarks and len(landmarks) > 0:
                            normalized = self.normalize_landmarks(landmarks)

                            # === TREINO ESTÁTICO ===
                            if self.mode == "train_static" and self.new_gesture_name:
                                self.hand_stable = self.is_hand_stable(normalized)
                                if self.hand_stable:
                                    self.new_gesture_data.append(normalized)
                                    self.sample_count += 1

                            # === TREINO DINÂMICO ===
                            elif self.mode == "train_dynamic" and self.new_gesture_name:
                                image = self.dynamic_recognizer.process_frame(image, normalized)

                            # === RECONHECIMENTO ===
                            elif self.mode == "recognize":
                                if self.model_manager.trained and self.is_hand_stable(normalized) and (time.time() - self.last_prediction_time) >= (CONFIG["prediction_cooldown"] / CONFIG["target_fps"]):
                                    pred, prob = self.model_manager.predict(normalized)
                                    if pred and prob >= CONFIG["confidence_threshold"]:
                                        self.current_word += pred
                                        cv2.putText(image, f"{pred} ({prob:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        self.last_prediction_time = time.time()
                                if self.dynamic_recognizer.model_trained:
                                    image = self.dynamic_recognizer.process_frame(image, normalized)

                # === UI ===
                status = f"Modo: {'Treino Estático' if self.mode == 'train_static' else 'Treino Dinâmico' if self.mode == 'train_dynamic' else 'Reconhecimento'}"
                dynamic_status = self.dynamic_recognizer.status_message if self.mode in ["train_dynamic", "recognize"] else ""
                self.ui.draw_ui(
                    image, status, 0, self.current_word, self.sample_count, self.input_text,
                    self.is_input_active, self.new_gesture_name, self.gesture_list if self.delete_mode else None,
                    self.selected_index if self.delete_mode else None, self.hand_stable, self.variance,
                    dynamic_status, self.mode, len(self.dynamic_recognizer.dynamic_sequence), self.saving_status
                )

                cv2.imshow("GestureApp", image)
                key = cv2.waitKey(1) & 0xFF

                # === TECLAS DE JANELA ===
                if key == ord('f'):
                    cv2.namedWindow("GestureApp", cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty("GestureApp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                elif key == ord('w'):
                    cv2.namedWindow("GestureApp", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("GestureApp", 900, 600)

                # === MODO EXCLUSÃO ===
                if self.delete_mode:
                    if key == 27:  # ESC
                        self.delete_mode = False
                        self.gesture_list = []
                    elif key in [ord('n'), 40] and self.gesture_list:
                        self.selected_index = (self.selected_index + 1) % len(self.gesture_list)
                    elif key in [ord('p'), 38] and self.gesture_list:
                        self.selected_index = (self.selected_index - 1) % len(self.gesture_list)
                    elif key == 13 and self.gesture_list:
                        gesture = self.gesture_list[self.selected_index]
                        if self.mode == "train_dynamic":
                            self.db.delete_dynamic_gesture(gesture)
                        else:
                            self.db.delete_gesture(self.gesture_type, gesture)
                            self.data_dict, self.labels, self.data, self.gesture_names = self.db.load_gestures(self.gesture_type)
                            if self.labels:
                                self.model_manager.train(self.data, self.labels)
                        self.gesture_list = self.db.list_gestures(self.gesture_type) if self.mode != "train_dynamic" else self.db.list_dynamic_gestures()
                        self.selected_index = min(self.selected_index, len(self.gesture_list)-1) if self.gesture_list else 0

                # === ENTRADA DE TEXTO ===
                elif self.is_input_active:
                    if key == 13:
                        self.new_gesture_name = self.input_text.upper()
                        self.is_input_active = False
                        self.input_text = ""
                        self.sample_count = 0
                        if self.new_gesture_name:
                            print(f"[INFO] Iniciando treino para '{self.new_gesture_name}'")
                    elif key == 27:
                        self.is_input_active = False
                        self.input_text = ""
                        self.mode = "train_static"
                    elif key == 8:
                        self.input_text = self.input_text[:-1]
                    elif 65 <= key <= 90 or 97 <= key <= 122:
                        self.input_text += chr(key).upper()

                # === TECLAS PRINCIPAIS ===
                else:
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        self.current_word = ""
                        self.new_gesture_data = []
                        self.sample_count = 0
                        self.dynamic_recognizer.dynamic_sequence = []
                        print("[INFO] Dados limpos")
                    elif key == ord('t'):
                        if self.mode == "train_dynamic":
                            self.saving_status = "Treinando modelo dinâmico..."
                            self.dynamic_recognizer.train_and_save_model_lstm()
                        else:
                            self.mode = "train_static"
                            self.is_input_active = True
                            self.input_text = ""
                            self.new_gesture_name = ""
                            self.new_gesture_data = []
                            self.delete_mode = False
                            self.prev_landmarks = None
                            self.landmark_history = []
                            print("[INFO] Modo Treino Estático")
                    elif key == ord('m'):
                        self.mode = "train_dynamic"
                        self.is_input_active = True
                        self.input_text = ""
                        self.new_gesture_name = ""
                        self.new_gesture_data = []
                        self.delete_mode = False
                        print("[INFO] Modo Treino Dinâmico")
                    elif key == ord('r'):
                        self.mode = "recognize"
                        self.is_input_active = False
                        self.new_gesture_name = ""
                        self.new_gesture_data = []
                        self.delete_mode = False
                        print("[INFO] Modo Reconhecimento")
                    elif key == ord('d'):
                        self.delete_mode = True
                        self.gesture_list = self.db.list_gestures(self.gesture_type) if self.mode != "train_dynamic" else self.db.list_dynamic_gestures()
                        self.selected_index = 0
                        print(f"[INFO] Modo Excluir: {len(self.gesture_list)} gestos")

                    # === GRAVAÇÃO DINÂMICA: SEGURE 'S' ===
                    elif key == ord('s'):
                        if self.mode == "train_dynamic" and self.new_gesture_name and not self.dynamic_recognizer.recording:
                            self.dynamic_recognizer.start_recording(self.new_gesture_name)
                            self.saving_status = "GRAVANDO... (solte S)"
                            self.recording_key_pressed = True
                    if self.recording_key_pressed and key != ord('s'):
                        self.dynamic_recognizer.stop_recording()
                        self.saving_status = ""
                        self.recording_key_pressed = False

                    # === SALVAR ESTÁTICO (COM DADOS ANTIGOS DO BANCO) ===
                    elif key == ord('s') and self.mode == "train_static" and self.new_gesture_name and self.new_gesture_data:
                        min_samples = CONFIG["min_samples_per_class"]
                        if len(self.new_gesture_data) < min_samples:
                            print(f"[WARNING] Poucos samples: {len(self.new_gesture_data)} (mínimo: {min_samples})")
                        else:
                            # === 1. CARREGA TODOS OS DADOS DO BANCO ===
                            all_dict, all_labels, all_data, _ = self.db.load_gestures(gesture_type=self.gesture_type)
                            
                            # === 2. ADICIONA OS NOVOS ===
                            new_labels = [self.new_gesture_name] * len(self.new_gesture_data)
                            new_types = [self.gesture_type] * len(self.new_gesture_data)
                            
                            # === 3. SALVA APENAS OS NOVOS NO BANCO ===
                            self.db.save_gestures(new_labels, self.new_gesture_data, new_types)
                            
                            # === 4. JUNTA TUDO EM MEMÓRIA ===
                            all_labels.extend(new_labels)
                            all_data.extend(self.new_gesture_data)
                            all_data, all_labels = shuffle(all_data, all_labels, random_state=42)
                            
                            # === 5. TREINA COM TUDO ===
                            success = self.model_manager.train(all_data, all_labels)
                            num_classes = len(set(all_labels))
                            print(f"[SUCESSO] '{self.new_gesture_name}' salvo! Total: {num_classes} classes -> {set(all_labels)}")
                            
                            # === 6. ATUALIZA VARIÁVEIS GLOBAIS ===
                            self.labels = all_labels
                            self.data = all_data
                            self.data_dict = all_dict
                            self.data_dict[self.new_gesture_name] = self.new_gesture_data

                        # === LIMPA PARA PRÓXIMO ===
                        self.new_gesture_name = ""
                        self.new_gesture_data = []
                        self.sample_count = 0

        except Exception as e:
            logging.error(f"Erro na execução: {e}")
            print(f"[ERROR] {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.db.close()
            print("[INFO] GestureApp encerrado")

if __name__ == "__main__":
    app = GestureApp(gesture_type="letter")
    app.run()