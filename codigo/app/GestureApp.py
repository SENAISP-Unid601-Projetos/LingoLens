import cv2
import mediapipe as mp
import logging
import numpy as np
import time
from Config import CONFIG, validate_gesture_type
from Database_manager import DatabaseManager
from Model_manager import ModelManager
from Ui_manager import UIManager
from Extract_landmarks import extract_landmarks

class GestureApp:
    def __init__(self, gesture_type="letter"):
        validate_gesture_type(gesture_type)
        print("[INFO] Inicializando GestureApp...")
        logging.basicConfig(
            filename=CONFIG["log_file"],
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        try:
            self.db = DatabaseManager(CONFIG["db_path"])
            self.model_manager = ModelManager(gesture_type=gesture_type)
            self.ui = UIManager()
            self.gesture_type = gesture_type.lower()

            self.data_dict, self.labels, self.data, self.gesture_names = self.db.load_gestures(gesture_type=self.gesture_type)
            if self.labels:
                success = self.model_manager.train(self.data, self.labels)
                if success:
                    num_classes = len(set(self.labels))
                    print(f"[INFO] Modelo carregado com {num_classes} gesto(s): {set(self.labels)}")
                    if num_classes == 1:
                        print(f"[INFO] Apenas 1 classe disponivel. Predições serao nao-discriminativas.")
                else:
                    print("[WARNING] Falha ao carregar modelo.")
            else:
                print("[INFO] Nenhum gesto treinado ainda.")

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.error("Nao foi possivel abrir a webcam")
                raise RuntimeError("Nao foi possível abrir a webcam")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera_resolution"][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera_resolution"][1])
            self.cap.set(cv2.CAP_PROP_FPS, CONFIG["train_fps"])

            self.hands = mp.solutions.hands.Hands(
                max_num_hands=CONFIG["max_num_hands"],
                min_detection_confidence=CONFIG["min_detection_confidence"],
                static_image_mode=False,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands

            self.current_word = ""
            self.mode = "teste"
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
        except Exception as e:
            logging.error(f"Erro na inicialização do GestureApp: {e}")
            raise

    def is_hand_stable(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return False
        try:
            variance = np.var(np.array(landmarks) - np.array(self.prev_landmarks))
            self.prev_landmarks = landmarks
            return variance < 0.005
        except Exception as e:
            logging.error(f"Erro ao verificar estabilidade da mao: {e}")
            return False

    def run(self):
        print("[INFO] Teclas: Q=Sair C:Limpar T:Treino S:Gesto D:Excluir")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Falha ao capturar frame")
                    print("[WARNING] Falha ao capturar frame")
                    break

                self.frame_count += 1
                if self.frame_count % (CONFIG["target_fps"] // CONFIG["train_fps"]) != 0 and self.mode == "treino":
                    continue

                image = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                if results.multi_hand_landmarks and not self.delete_mode:
                    for hand in results.multi_hand_landmarks:
                        if self.mode == "treino":
                            self.mp_drawing.draw_landmarks(
                                image,
                                hand,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                            )
                        landmarks = extract_landmarks(hand, image_shape=image.shape)
                        if landmarks is not None and len(landmarks) > 0:
                            if self.mode == "treino" and self.new_gesture_name and self.is_hand_stable(landmarks):
                                self.new_gesture_data.append(landmarks)
                                self.sample_count += 1
                            elif self.mode == "teste" and self.labels and (time.time() - self.last_prediction_time) >= (CONFIG["prediction_cooldown"] / CONFIG["target_fps"]):
                                pred, prob = self.model_manager.predict(landmarks)
                                if pred and prob >= CONFIG["confidence_threshold"]:
                                    self.current_word += pred
                                    cv2.putText(image, f"Predicao: {pred} ({prob:.2f})",
                                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    logging.info(f"Predicao: {pred} | Probabilidade: {prob:.2f}")
                                    self.last_prediction_time = time.time()
                                else:
                                    logging.debug(f"Predicao ignorada: {pred} com probabilidade {prob:.2f} < {CONFIG['confidence_threshold']}")

                # Preparar dados para UI
                status = f"Modo: {'Treino' if self.mode == 'treino' else 'Teste'} ({self.gesture_type})"
                gesture_list = self.gesture_list if self.delete_mode else None
                selected_index = self.selected_index if self.delete_mode else None

                self.ui.draw_ui(
                    image, status, CONFIG["prediction_cooldown"] / CONFIG["target_fps"], 
                    self.current_word, self.sample_count, self.input_text,
                    self.is_input_active, self.new_gesture_name, gesture_list, selected_index
                )

                cv2.imshow("GestureApp", image)
                key = cv2.waitKey(1) & 0xFF

                # Modo de exclusão
                if self.delete_mode:
                    if key == 27:  # ESC para sair do modo de exclusão
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
                    elif key == 13 and self.gesture_list:  # Enter para confirmar exclusão
                        gesture_to_delete = self.gesture_list[self.selected_index]
                        print(f"[INFO] Tentando deletar: {gesture_to_delete}")
                        if self.db.delete_gesture(self.gesture_type, gesture_to_delete):
                            print(f"[INFO] Gesto '{gesture_to_delete}' deletado com sucesso")
                            # Recarregar dados após exclusão
                            self.data_dict, self.labels, self.data, self.gesture_names = self.db.load_gestures(self.gesture_type)
                            self.gesture_list = self.db.list_gestures(self.gesture_type)
                            if self.gesture_list:
                                self.selected_index = min(self.selected_index, len(self.gesture_list) - 1)
                            else:
                                self.selected_index = 0
                            if self.labels:
                                success = self.model_manager.train(self.data, self.labels)
                                if success:
                                    print(f"[INFO] Modelo atualizado com {len(set(self.labels))} classe(s): {set(self.labels)}")
                                else:
                                    print("[ERROR] Falha ao atualizar modelo apos exclusao")
                            else:
                                self.model_manager.trained = False
                                print("[INFO] Nenhum gesto restante para treinar")
                        else:
                            print(f"[ERROR] Falha ao deletar gesto '{gesture_to_delete}'")
                        if not self.gesture_list:
                            self.delete_mode = False
                            print("[INFO] Modo Excluir desativado: nenhum gesto restante")

                # Modo de input ativo
                elif self.is_input_active:
                    if key == 13:  # Enter
                        self.new_gesture_name = self.input_text.upper()
                        self.is_input_active = False
                        self.input_text = ""
                        self.sample_count = 0
                        if self.new_gesture_name:
                            print(f"[INFO] Modo Treino ativado para '{self.new_gesture_name}'")
                    elif key == 27:  # ESC
                        self.is_input_active = False
                        self.input_text = ""
                        self.mode = "teste"
                        print("[INFO] Entrada de texto cancelada. Retornando ao modo Teste")
                    elif key == 8:  # Backspace
                        self.input_text = self.input_text[:-1]
                    elif 65 <= key <= 90 or 97 <= key <= 122:  # Letras A-Z, a-z
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
                        print("[INFO] Palavra atual e dados de treino limpos")
                    elif key == ord("t"):
                        self.mode = "treino"
                        self.is_input_active = True
                        self.input_text = ""
                        self.new_gesture_data = []
                        self.sample_count = 0
                        self.prev_landmarks = None
                        print("[INFO] Modo de entrada de texto ativado. Digite na janela (ESC para cancelar)")
                    elif key == ord("s"):
                        if self.mode == "treino" and self.new_gesture_name and self.new_gesture_data:
                            if len(self.new_gesture_data) < CONFIG["min_samples_per_class"]:
                                print(f"[WARNING] Coletados poucos samples ({len(self.new_gesture_data)}). Recomenda-se {CONFIG['min_samples_per_class']}")
                            
                            new_labels = [self.new_gesture_name] * len(self.new_gesture_data)
                            new_gesture_types = [self.gesture_type] * len(self.new_gesture_data)
                            self.labels.extend(new_labels)
                            self.data.extend(self.new_gesture_data)
                            self.db.save_gestures(new_labels, self.new_gesture_data, new_gesture_types)
                            
                            success = self.model_manager.train(self.data, self.labels)
                            if success:
                                num_classes = len(set(self.labels))
                                print(f"[INFO] Gestos de '{self.new_gesture_name}' salvos e modelo atualizado ({num_classes} classe(s): {set(self.labels)})")
                                if num_classes == 1:
                                    print(f"[INFO] Adicione mais letras para predicoes discriminativas.")
                            else:
                                print("[ERROR] Falha ao treinar modelo após salvar gestos.")
                            
                            self.mode = "teste"
                            self.new_gesture_name = ""
                            self.new_gesture_from_data = []
                            self.is_input_active = False
                            self.sample_count = 0
                            print("[INFO] Modo Teste ativado")
                        elif self.mode == "treino" and not self.new_gesture_data:
                            print("[WARNING] Nenhum dado de gesto capturado para salvar")
                    elif key == ord("d"):
                        self.delete_mode = True
                        self.gesture_list = self.db.list_gestures(self.gesture_type)
                        self.selected_index = 0
                        if not self.gesture_list:
                            print("[WARNING] Nenhum gesto encontrado para excluir")
                            self.delete_mode = False
                        else:
                            print(f"[INFO] Modo Excluir ativado. {len(self.gesture_list)} gestos encontrados.")
                            print("Use N (próximo), P (anterior), ENTER para excluir, ESC para sair")

        except Exception as e:
            logging.error(f"Erro durante execucao do GestureApp: {e}")
            print(f"[ERROR] Erro durante execucao: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.db.close()
            print("[INFO] GestureApp encerrado")