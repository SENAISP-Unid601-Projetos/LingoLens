import cv2
import mediapipe as mp
import logging
import numpy as np
import time
import platform
import os
from Config import CONFIG, validate_gesture_type
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Preprocess_landmarks import preprocess_landmarks
from Dynamic_gesture_recognizer import DynamicGestureRecognizer
from sklearn.utils import shuffle

# Suprimir avisos do TensorFlow Lite e oneDNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

class GestureApp:
    def __init__(self, gesture_type="letter"):
        """Inicializa o aplicativo com suporte a gestos estáticos e dinâmicos."""
        validate_gesture_type(gesture_type)
        print("[INFO] Inicializando GestureApp...")
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Verificar compatibilidade de CPU
        self._check_cpu_compatibility()

        try:
            # Inicializar componentes
            self.db = DatabaseManager(CONFIG["db_path"])
            self.model_manager = ModelManager(gesture_type=gesture_type)
            self.ui = UIManager()
            self.dynamic_recognizer = DynamicGestureRecognizer(CONFIG)
            self.gesture_type = gesture_type.lower()

            # Carregar dados estáticos
            self.data_dict, self.labels, self.data, self.gesture_names = self.db.load_gestures(gesture_type=self.gesture_type)
            if self.labels:
                success = self.model_manager.train(self.data, self.labels)
                if success:
                    num_classes = len(set(self.labels))
                    print(f"[INFO] Modelo estático carregado com {num_classes} gesto(s): {set(self.labels)}")
                    if num_classes == 1:
                        print("[INFO] Apenas 1 classe disponível. Predições estáticas serão não-discriminativas.")
                else:
                    print("[WARNING] Falha ao carregar modelo estático.")
            else:
                print("[INFO] Nenhum gesto estático treinado ainda.")

            # Carregar dados dinâmicos
            try:
                self.dynamic_recognizer.load_model_lstm()
            except FileNotFoundError:
                print("[INFO] Modelo LSTM não encontrado. Treine gestos dinâmicos para criar o modelo.")

            # Configurar webcam
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.error("Não foi possível abrir a webcam")
                raise RuntimeError("Não foi possível abrir a webcam")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
            self.cap.set(cv2.CAP_PROP_FPS, CONFIG["train_fps"])

            # Configurar MediaPipe Hands
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=CONFIG["max_num_hands"],
                min_detection_confidence=CONFIG["min_detection_confidence"],
                static_image_mode=False,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

            # Variáveis de estado
            self.current_word = ""
            self.mode = "train_static"  # Modos: "train_static", "train_dynamic", "recognize"
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
        except Exception as e:
            logging.error(f"Erro na inicialização do GestureApp: {e}")
            raise

    def _check_cpu_compatibility(self):
        """Verifica a compatibilidade da CPU com instruções necessárias."""
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            cpu_flags = info.get('flags', [])
            # Adicionar 'sse3' explicitamente como sinônimo de 'ssse3' ou 'pni'
            if 'ssse3' in cpu_flags or 'pni' in cpu_flags:
                cpu_flags.append('sse3')  # Corrigir detecção de SSE3
            required_flags = ['sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2', 'avx', 'avx2', 'fma']
            missing_flags = [flag for flag in required_flags if flag not in cpu_flags]
            if missing_flags:
                logging.warning(f"CPU não suporta as seguintes instruções: {', '.join(missing_flags)}. O desempenho pode ser reduzido.")
                print(f"[WARNING] CPU não suporta: {', '.join(missing_flags)}. Considere atualizar o hardware ou usar uma versão otimizada do TensorFlow.")
            else:
                logging.info("CPU compatível com todas as instruções necessárias.")
                print("[INFO] CPU compatível com todas as instruções necessárias.")
        except ImportError:
            logging.warning("Pacote 'py-cpuinfo' não instalado. Não foi possível verificar compatibilidade da CPU.")
            print("[WARNING] Instale 'py-cpuinfo' para verificar compatibilidade da CPU: pip install py-cpuinfo")
        except Exception as e:
            logging.error(f"Erro ao verificar compatibilidade da CPU: {e}")
            print(f"[ERROR] Erro ao verificar CPU: {e}")

    def normalize_landmarks(self, landmarks):
        """Normaliza landmarks pela distância entre pulso (0) e base do dedo médio (9)."""
        try:
            wrist = np.array(landmarks[0:3])
            middle_finger_base = np.array(landmarks[27:30])
            scale = np.linalg.norm(wrist - middle_finger_base)
            if scale == 0:
                logging.warning("Escala zero detectada na normalização de landmarks")
                return landmarks
            return [coord / scale for coord in landmarks]
        except Exception as e:
            logging.error(f"Erro ao normalizar landmarks: {e}")
            return landmarks

    def is_hand_stable(self, landmarks):
        """Verifica se a mão está estável com base na variância dos landmarks."""
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            self.landmark_history = [landmarks]
            return False
        try:
            normalized_landmarks = self.normalize_landmarks(landmarks)
            self.landmark_history.append(normalized_landmarks)
            if len(self.landmark_history) > 3:
                self.landmark_history.pop(0)
            smoothed_landmarks = np.mean(self.landmark_history, axis=0)
            key_indices = [0, 5, 9, 13, 17]
            key_landmark_indices = [i * 3 for i in key_indices] + [i * 3 + 1 for i in key_indices] + [i * 3 + 2 for i in key_indices]
            key_landmarks = [smoothed_landmarks[i] for i in key_landmark_indices]
            key_prev_landmarks = [self.prev_landmarks[i] for i in key_landmark_indices]
            self.variance = np.var(np.array(key_landmarks) - np.array(key_prev_landmarks))
            self.prev_landmarks = smoothed_landmarks
            return self.variance < CONFIG["movement_threshold"]
        except Exception as e:
            logging.error(f"Erro ao verificar estabilidade da mão: {e}")
            return False

    def run(self):
        """Loop principal do aplicativo."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Falha ao capturar frame")
                    print("[WARNING] Falha ao capturar frame")
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
                                image,
                                hand,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                            )
                        landmarks = preprocess_landmarks(hand, image_shape=image.shape)
                        if landmarks is not None and len(landmarks) > 0:
                            normalized_landmarks = self.normalize_landmarks(landmarks)
                            if self.mode == "train_static" and self.new_gesture_name:
                                self.hand_stable = self.is_hand_stable(normalized_landmarks)
                                if self.hand_stable:
                                    self.new_gesture_data.append(normalized_landmarks)
                                    self.sample_count += 1
                            elif self.mode == "train_dynamic" and self.new_gesture_name:
                                self.dynamic_recognizer.training_mode = True
                                self.dynamic_recognizer.current_gesture_name = self.new_gesture_name
                                self.dynamic_recognizer.dynamic_mode = True
                                image = self.dynamic_recognizer.process_frame(image, normalized_landmarks)
                            elif self.mode == "recognize":
                                if self.model_manager.trained and self.is_hand_stable(normalized_landmarks) and (time.time() - self.last_prediction_time) >= (CONFIG["prediction_cooldown"] / CONFIG["target_fps"]):
                                    pred, prob = self.model_manager.predict(normalized_landmarks)
                                    if pred and prob >= CONFIG["confidence_threshold"]:
                                        self.current_word += pred
                                        cv2.putText(image, f"Estático: {pred} ({prob:.2f})",
                                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        logging.info(f"Predição estática: {pred} | Probabilidade: {prob:.2f}")
                                        self.last_prediction_time = time.time()
                                if self.dynamic_recognizer.model_trained:
                                    image = self.dynamic_recognizer.process_frame(image, normalized_landmarks)
                                else:
                                    cv2.putText(image, "Modelo dinâmico não treinado", 
                                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)

                # Preparar dados para UI
                status = f"Modo: {'Treino Estático' if self.mode == 'train_static' else 'Treino Dinâmico' if self.mode == 'train_dynamic' else 'Reconhecimento'} ({self.gesture_type})"
                gesture_list = self.gesture_list if self.delete_mode else None
                selected_index = self.selected_index if self.delete_mode else None
                dynamic_status = self.dynamic_recognizer.status_message if self.mode == "train_dynamic" or self.mode == "recognize" else ""

                self.ui.draw_ui(
                    image, status, CONFIG["prediction_cooldown"] / CONFIG["target_fps"], 
                    self.current_word, self.sample_count, self.input_text,
                    self.is_input_active, self.new_gesture_name, gesture_list, selected_index,
                    hand_stable=self.hand_stable, variance=self.variance,
                    dynamic_status=dynamic_status, mode=self.mode,
                    dynamic_sequence_length=len(self.dynamic_recognizer.dynamic_sequence),
                    saving_status=self.saving_status
                )

                cv2.imshow("GestureApp", image)
                key = cv2.waitKey(1) & 0xFF

                # Modo de exclusão
                if self.delete_mode:
                    if key == 27:  # ESC
                        self.delete_mode = False
                        self.gesture_list = []
                        self.selected_index = 0
                        print("[INFO] Modo Excluir desativado")
                    elif key == ord('n') or key == 40:  # N ou seta para baixo
                        if self.gesture_list:
                            self.selected_index = (self.selected_index + 1) % len(self.gesture_list)
                    elif key == ord('p') or key == 38:  # P ou seta para cima
                        if self.gesture_list:
                            self.selected_index = (self.selected_index - 1) % len(self.gesture_list)
                    elif key == 13 and self.gesture_list:  # Enter
                        gesture_to_delete = self.gesture_list[self.selected_index]
                        print(f"[INFO] Tentando deletar: {gesture_to_delete}")
                        if self.mode == "train_dynamic":
                            if self.db.delete_dynamic_gesture(gesture_to_delete):
                                print(f"[INFO] Gesto dinâmico '{gesture_to_delete}' deletado")
                                self.dynamic_recognizer.data_dynamic, self.dynamic_recognizer.labels_dynamic = self.db.load_dynamic_gestures()
                                self.dynamic_recognizer._init_model()
                        else:
                            if self.db.delete_gesture(self.gesture_type, gesture_to_delete):
                                print(f"[INFO] Gesto estático '{gesture_to_delete}' deletado")
                                self.data_dict, self.labels, self.data, self.gesture_names = self.db.load_gestures(self.gesture_type)
                                if self.labels:
                                    success = self.model_manager.train(self.data, self.labels)
                                    if success:
                                        print(f"[INFO] Modelo estático atualizado com {len(set(self.labels))} classe(s)")
                                else:
                                    self.model_manager.trained = False
                        self.gesture_list = self.db.list_gestures(self.gesture_type) if self.mode != "train_dynamic" else self.db.list_dynamic_gestures()
                        if self.gesture_list:
                            self.selected_index = min(self.selected_index, len(self.gesture_list) - 1)
                        else:
                            self.selected_index = 0
                            self.delete_mode = False
                            print("[INFO] Modo Excluir desativado: nenhum gesto restante")

                # Modo de entrada de texto
                elif self.is_input_active:
                    if key == 13:  # Enter
                        self.new_gesture_name = self.input_text.upper()
                        self.is_input_active = False
                        self.input_text = ""
                        self.sample_count = 0
                        if self.new_gesture_name:
                            print(f"[INFO] Modo {'Treino Estático' if self.mode == 'train_static' else 'Treino Dinâmico'} ativado para '{self.new_gesture_name}'")
                    elif key == 27:  # ESC
                        self.is_input_active = False
                        self.input_text = ""
                        self.mode = "train_static"
                        print("[INFO] Entrada de texto cancelada. Retornando ao modo Treino Estático")
                    elif key == 8:  # Backspace
                        self.input_text = self.input_text[:-1]
                    elif 65 <= key <= 90 or 97 <= key <= 122:  # Letras A-Z
                        self.input_text += chr(key).upper()
                # Teclas normais
                else:
                    if key == ord("q"):
                        print("[INFO] Saindo do aplicativo...")
                        break
                    elif key == ord("c"):
                        self.current_word = ""
                        self.new_gesture_data = []
                        self.sample_count = 0
                        self.dynamic_recognizer.dynamic_sequence = []
                        print("[INFO] Palavra atual e dados de treino limpos")
                    elif key == ord("t"):
                        if self.mode == "train_dynamic":
                            self.saving_status = "Iniciando treinamento do modelo dinâmico..."
                            self.dynamic_recognizer.train_and_save_model_lstm()
                            print("[INFO] Treinamento do modelo dinâmico iniciado em segundo plano")
                        else:
                            self.mode = "train_static"
                            self.is_input_active = True
                            self.input_text = ""
                            self.new_gesture_name = ""
                            self.new_gesture_data = []
                            self.dynamic_recognizer.dynamic_sequence = []
                            self.delete_mode = False
                            self.prev_landmarks = None
                            self.landmark_history = []
                            print("[INFO] Modo Treino Estático ativado")
                    elif key == ord("m"):
                        self.mode = "train_dynamic"
                        self.is_input_active = True
                        self.input_text = ""
                        self.new_gesture_name = ""
                        self.new_gesture_data = []
                        self.dynamic_recognizer.dynamic_sequence = []
                        self.dynamic_recognizer.training_mode = True
                        self.delete_mode = False
                        print("[INFO] Modo Treino Dinâmico ativado")
                    elif key == ord("r"):
                        self.mode = "recognize"
                        self.is_input_active = False
                        self.new_gesture_name = ""
                        self.new_gesture_data = []
                        self.dynamic_recognizer.dynamic_sequence = []
                        self.delete_mode = False
                        print("[INFO] Modo Reconhecimento ativado")
                    elif key == ord("s") and self.mode in ["train_static", "train_dynamic"]:
                        self.saving_status = "Salvando..."
                        if self.mode == "train_static" and self.new_gesture_name and self.new_gesture_data:
                            if len(self.new_gesture_data) < CONFIG["min_samples_per_class"]:
                                print(f"[WARNING] Coletados poucos samples ({len(self.new_gesture_data)}). Recomenda-se {CONFIG['min_samples_per_class']}")
                            new_labels = [self.new_gesture_name] * len(self.new_gesture_data)
                            new_gesture_types = [self.gesture_type] * len(self.new_gesture_data)
                            self.labels.extend(new_labels)
                            self.data.extend(self.new_gesture_data)
                            self.data, self.labels = shuffle(self.data, self.labels, random_state=42)
                            self.db.save_gestures(new_labels, self.new_gesture_data, new_gesture_types)
                            success = self.model_manager.train(self.data, self.labels)
                            if not success:
                                self.model_manager.trained = False
                                print("[WARNING] Falha ao treinar modelo estático")
                            else:
                                num_classes = len(set(self.labels))
                                print(f"[INFO] Gestos estáticos de '{self.new_gesture_name}' salvos e modelo atualizado ({num_classes} classe(s))")
                            self.new_gesture_name = ""
                            self.new_gesture_data = []
                            self.is_input_active = False
                            self.sample_count = 0
                            print("[INFO] Modo Treino Estático ativado")
                        elif self.mode == "train_dynamic" and self.new_gesture_name and self.dynamic_recognizer.dynamic_sequence:
                            self.dynamic_recognizer.save_gesture_data()
                            self.new_gesture_name = ""
                            self.dynamic_recognizer.dynamic_sequence = []
                            self.is_input_active = False
                            print("[INFO] Gesto dinâmico salvo. Use 'T' para treinar o modelo.")
                        elif self.mode in ["train_static", "train_dynamic"] and not (self.new_gesture_data or self.dynamic_recognizer.dynamic_sequence):
                            print("[WARNING] Nenhum dado de gesto capturado para salvar")
                        self.saving_status = ""
                    elif key == ord("d") and self.mode in ["train_static", "train_dynamic"]:
                        self.delete_mode = True
                        self.gesture_list = self.db.list_gestures(self.gesture_type) if self.mode != "train_dynamic" else self.db.list_dynamic_gestures()
                        self.selected_index = 0
                        if not self.gesture_list:
                            print("[WARNING] Nenhum gesto encontrado para excluir")
                            self.delete_mode = False
                        else:
                            print(f"[INFO] Modo Excluir ativado. {len(self.gesture_list)} gestos encontrados.")
                            print("Use N (próximo), P (anterior), ENTER para excluir, ESC para sair")

        except Exception as e:
            logging.error(f"Erro durante execução do GestureApp: {e}")
            print(f"[ERROR] Erro durante execução: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.db.close()
            print("[INFO] GestureApp encerrado")