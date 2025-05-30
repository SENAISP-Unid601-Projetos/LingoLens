import cv2
import mediapipe as mp
import numpy as np
import pickle
import os


class GestureRecognizer:
    def __init__(self):
        # Configurações
        self.MODEL_PATH = "gesture_model.pkl"
        self.GESTURES_PATH = "gestures_data.pkl"
        self.CONFIDENCE_THRESHOLD = 0.85

        # Inicializar Mediapipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.7)

        # Variáveis de estado
        self.data = []
        self.labels = []
        self.current_word = ""
        self.number_mode = False
        self.training_mode = False
        self.current_gesture_name = ""
        self.gesture_names = {}

        # Carregar dados existentes
        self.load_saved_data()

        # Configurar captura de vídeo
        self.cap = cv2.VideoCapture(0)
        self.set_camera_resolution(1280, 720)

        # Configurações de tela
        self.target_width = 1280  # Largura base para redimensionamento
        self.ui_scale = 1.0       # Fator de escala para elementos de UI

    def load_saved_data(self):
        """Carrega dados de gestos salvos anteriormente"""
        if os.path.exists(self.GESTURES_PATH):
            with open(self.GESTURES_PATH, 'rb') as f:
                self.gesture_names = pickle.load(f)

        if os.path.exists(self.MODEL_PATH):
            with open(self.MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
        else:
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(n_neighbors=3)

    def set_camera_resolution(self, width, height):
        """Configura a resolução da câmera"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def extract_landmarks(self, hand_landmarks):
        """Extrai os pontos de referência da mão"""
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    def resize_with_aspect_ratio(self, image, target_width=None):
        """Redimensiona a imagem mantendo a proporção"""
        (h, w) = image.shape[:2]

        if target_width is None:
            return image

        ratio = target_width / float(w)
        dim = (target_width, int(h * ratio))

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def calculate_ui_scale(self, screen_width):
        """Calcula o fator de escala para elementos de UI"""
        base_width = 1280  # Largura de referência para design
        return screen_width / base_width

    def draw_ui_elements(self, image, screen_width):
        """Desenha todos os elementos de interface do usuário"""
        # Calcular escala baseada na largura da tela
        self.ui_scale = self.calculate_ui_scale(screen_width)

        # Barra de status superior
        status_text = f'Modo: {"Treino" if self.training_mode else "Reconhecimento"} | '
        status_text += f'Entrada: {"Número" if self.number_mode else "Letra"}'

        # Ajustar tamanhos e posições com base na escala
        status_bar_height = int(40 * self.ui_scale)
        font_scale_status = 0.7 * self.ui_scale
        font_scale_word = 1.0 * self.ui_scale
        font_scale_instructions = 0.5 * self.ui_scale

        cv2.rectangle(
            image, (0, 0), (image.shape[1], status_bar_height), (0, 0, 0), -1)
        cv2.putText(image, status_text,
                    (int(10 * self.ui_scale), int(25 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_status,
                    (255, 255, 255), int(2 * self.ui_scale))

        # Palavra atual
        cv2.putText(image, f'Palavra: {self.current_word}',
                    (image.shape[1]//2 - int(150 * self.ui_scale),
                     int(70 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_word,
                    (255, 255, 255), int(2 * self.ui_scale))

        # Instruções
        instructions = "Q: Sair | C: Limpar | N: Num/Letra | T: Modo Treino | S: Novo Gesto"
        cv2.putText(image, instructions,
                    (int(10 * self.ui_scale),
                     image.shape[0] - int(10 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_instructions,
                    (200, 200, 200), int(1 * self.ui_scale))

    def process_gestures(self, image, landmarks):
        """Processa os gestos reconhecidos"""

        if self.training_mode and self.current_gesture_name:
            self.data.append(landmarks)
            self.labels.append(self.current_gesture_name)

            font_scale = 0.8 * self.ui_scale
            cv2.putText(image, f"Coletando: {self.current_gesture_name}",
                        (int(10 * self.ui_scale), int(160 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 255), int(2 * self.ui_scale))

            # Mostrar contagem de amostras
            sample_count = sum(
                1 for label in self.labels if label == self.current_gesture_name)
            cv2.putText(image, f"Amostras: {sample_count}",
                        (int(10 * self.ui_scale), int(190 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                        (0, 255, 255), int(2 * self.ui_scale))
            return

        if not self.training_mode and len(self.labels) > 0:
            prediction = self.model.predict([landmarks])[0]
            probability = self.model.predict_proba([landmarks]).max()

            font_scale = 1.0 * self.ui_scale  # Definido aqui para evitar erro

            if probability >= self.CONFIDENCE_THRESHOLD:
                label = self.gesture_names.get(prediction, prediction)

                if (self.number_mode and label.isdigit()) or (not self.number_mode and not label.isdigit()):
                    self.current_word += label

                # Desenhar resultado do reconhecimento
                cv2.putText(image, f'{label} ({probability*100:.1f}%)',
                            (int(10 * self.ui_scale), int(120 * self.ui_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 255, 0), int(2 * self.ui_scale))
            else:
                cv2.putText(image, 'Desconhecido',
                            (int(10 * self.ui_scale), int(120 * self.ui_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 255), int(2 * self.ui_scale))

    def handle_key_commands(self, key):
        """Processa os comandos do teclado"""
        if key == ord('q'):
            return False  # Sair

        elif key == ord('c'):
            self.current_word = ""

        elif key == ord('n'):
            self.number_mode = not self.number_mode

        elif key == ord('t'):
            self.training_mode = not self.training_mode
            if not self.training_mode and len(self.labels) > 0:
                self.train_and_save_model()

        elif key == ord('s'):
            self.current_gesture_name = input(
                "Nome do novo gesto (ex: A, B, C ou 1, 2, 3): ").strip().upper()
            if self.current_gesture_name:
                self.gesture_names[self.current_gesture_name] = self.current_gesture_name
                print(
                    f"Gesto '{self.current_gesture_name}' pronto para treinamento!")

        return True  # Continuar execução

    def train_and_save_model(self):
        """Treina e salva o modelo de reconhecimento"""
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(self.data, self.labels)

        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.GESTURES_PATH, 'wb') as f:
            pickle.dump(self.gesture_names, f)

        print("Modelo treinado e salvo.")

    def run(self):
        """Método principal para executar o aplicativo"""
        cv2.namedWindow('Reconhecimento de Gestos', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Reconhecimento de Gestos',
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Falha na captura de vídeo")
                break

            # Processar imagem
            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            # Obter dimensões da tela
            screen_width = cv2.getWindowImageRect(
                'Reconhecimento de Gestos')[2]

            # Processar gestos
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = self.extract_landmarks(hand_landmarks)
                    self.process_gestures(image, landmarks)

            # Desenhar elementos de UI
            self.draw_ui_elements(image, screen_width)

            # Redimensionar para tela cheia mantendo proporção
            resized_image = self.resize_with_aspect_ratio(
                image, target_width=screen_width)

            # Exibir imagem
            cv2.imshow('Reconhecimento de Gestos', resized_image)

            # Processar comandos do teclado
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key_commands(key):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Iniciar aplicativo
if __name__ == "__main__":
    app = GestureRecognizer()
    app.run()
