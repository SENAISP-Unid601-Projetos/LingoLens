import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import json
from pathlib import Path

class GestureRecognizer:
    def __init__(self):
        # Configurações
        self.DB_PATH = "gestures.db"
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

        self.show_text_input = False
        self.input_text = ""
        self.input_prompt = ""

        # 🔧 MOD: Controle de repetição e cooldown
        self.last_prediction = ""
        self.prediction_cooldown = 20
        self.cooldown_counter = 0

        self.init_db()
        self.load_saved_data()

        self.cap = cv2.VideoCapture(0)
        self.set_camera_resolution(1280, 720)

        self.target_width = 1280
        self.ui_scale = 1.0

    def init_db(self):
        self.conn = sqlite3.connect(self.DB_PATH)
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS gestures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            landmarks TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS gesture_names (
            name TEXT PRIMARY KEY NOT NULL
        )''')
        self.conn.commit()

    def load_saved_data(self):
        cursor = self.conn.execute('SELECT name FROM gesture_names')
        self.gesture_names = {name[0]: name[0] for name in cursor.fetchall()}

        cursor = self.conn.execute('SELECT name, landmarks FROM gestures')
        for name, landmarks_json in cursor.fetchall():
            landmarks = json.loads(landmarks_json)
            self.labels.append(name)
            self.data.append(landmarks)

        self.init_model()

    def init_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=3)
        if len(set(self.labels)) > 1:  # 🔧 MOD: precisa de pelo menos 2 classes
            self.model.fit(self.data, self.labels)

    def save_gesture_data(self):
        self.conn.execute('DELETE FROM gestures')
        for name, landmarks in zip(self.labels, self.data):

            if isinstance(landmarks, np.ndarray):
                landmarks_serializable = landmarks.tolist()
            else:
                landmarks_serializable = landmarks    
            self.conn.execute(
                'INSERT INTO gestures (name, landmarks) VALUES (?, ?)',
                (name, json.dumps(landmarks_serializable)))
        self.conn.execute('DELETE FROM gesture_names')
        for name in set(self.labels):
            self.conn.execute(
                'INSERT OR IGNORE INTO gesture_names (name) VALUES (?)',
                (name,))
        self.conn.commit()

    def set_camera_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def extract_landmarks(self, hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

    def resize_with_aspect_ratio(self, image, target_width=None):
        (h, w) = image.shape[:2]
        if target_width is None:
            return image
        ratio = target_width / float(w)
        dim = (target_width, int(h * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def calculate_ui_scale(self, screen_width):
        base_width = 1280
        return screen_width / base_width

    def draw_ui_elements(self, image, screen_width):
        self.ui_scale = self.calculate_ui_scale(screen_width)

        status_text = f'Modo: {"Treino" if self.training_mode else "Reconhecimento"} | '
        status_text += f'Entrada: {"Número" if self.number_mode else "Letra"}'

        status_bar_height = int(40 * self.ui_scale)
        font_scale_status = 0.7 * self.ui_scale
        font_scale_word = 1.0 * self.ui_scale
        font_scale_instructions = 0.5 * self.ui_scale

        cv2.rectangle(image, (0, 0), (image.shape[1], status_bar_height), (0, 0, 0), -1)
        cv2.putText(image, status_text,
                    (int(10 * self.ui_scale), int(25 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_status,
                    (255, 255, 255), int(2 * self.ui_scale))

        cv2.putText(image, f'Palavra: {self.current_word}',
                    (image.shape[1]//2 - int(150 * self.ui_scale), int(70 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_word,
                    (255, 255, 255), int(2 * self.ui_scale))

        instructions = "Q: Sair | C: Limpar | N: Num/Letra | T: Modo Treino | S: Novo Gesto"
        cv2.putText(image, instructions,
                    (int(10 * self.ui_scale), image.shape[0] - int(10 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_instructions,
                    (200, 200, 200), int(1 * self.ui_scale))

        if self.show_text_input:
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            box_width, box_height = 600, 150
            x = (image.shape[1] - box_width) // 2
            y = (image.shape[0] - box_height) // 2

            cv2.rectangle(image, (x, y), (x+box_width, y+box_height), (50, 50, 50), -1)
            cv2.rectangle(image, (x, y), (x+box_width, y+box_height), (255, 255, 255), 2)

            cv2.putText(image, self.input_prompt,
                        (x + 20, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(image, self.input_text,
                        (x + 20, y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            cv2.putText(image, "Enter: confirmar | Esc: cancelar",
                        (x + 20, y + 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

    def process_gestures(self, image, landmarks):
        if self.show_text_input:
            return

        if self.training_mode and self.current_gesture_name:
            self.data.append(landmarks)
            self.labels.append(self.current_gesture_name)

            font_scale = 0.8 * self.ui_scale
            cv2.putText(image, f"Coletando: {self.current_gesture_name}",
                        (int(10 * self.ui_scale), int(160 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 255), int(2 * self.ui_scale))
            sample_count = sum(1 for label in self.labels if label == self.current_gesture_name)
            cv2.putText(image, f"Amostras: {sample_count}",
                        (int(10 * self.ui_scale), int(190 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                        (0, 255, 255), int(2 * self.ui_scale))
            return

        if not self.training_mode and len(set(self.labels)) > 1:
            prediction = self.model.predict([landmarks])[0]
            probability = self.model.predict_proba([landmarks]).max()

            font_scale = 1.0 * self.ui_scale
            if probability >= self.CONFIDENCE_THRESHOLD:
                label = self.gesture_names.get(prediction, prediction)

                # 🔧 MOD: evitar repetição e aplicar cooldown
                if self.cooldown_counter == 0:
                    if label != self.last_prediction:
                        if (self.number_mode and label.isdigit()) or (not self.number_mode and not label.isdigit()):
                            self.current_word += label
                            self.last_prediction = label
                            self.cooldown_counter = self.prediction_cooldown

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
        if self.show_text_input:
            return True

        if key == ord('q'):
            return False
        elif key == ord('c'):
            self.current_word = ""
        elif key == ord('n'):
            self.number_mode = not self.number_mode
        elif key == ord('t'):
            self.training_mode = not self.training_mode
            if not self.training_mode and len(set(self.labels)) > 1:
                self.train_and_save_model()
        elif key == ord('s'):
            self.show_text_input = True
            self.input_text = ""
            self.input_prompt = "Digite o nome do novo gesto:"
            self.current_gesture_name = ""
        return True

    def process_text_input(self, key):
        if not self.show_text_input:
            return
        if key == 13:  # Enter
            if self.input_text:
                self.current_gesture_name = self.input_text.strip().upper()
                self.gesture_names[self.current_gesture_name] = self.current_gesture_name
                print(f"Gesto '{self.current_gesture_name}' pronto para treinamento!")
            self.show_text_input = False
            self.input_text = ""
        elif key == 27:  # Esc
            self.show_text_input = False
            self.input_text = ""
            self.current_gesture_name = ""
        elif key == 8:  # Backspace
            self.input_text = self.input_text[:-1]
        elif 32 <= key <= 126:
            self.input_text += chr(key)

    def train_and_save_model(self):
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=3)
        if len(set(self.labels)) > 1:
            self.model.fit(self.data, self.labels)
            self.save_gesture_data()
            print("Modelo treinado e salvo no banco de dados.")
        else:
            print("É necessário mais de um gesto diferente para treinar o modelo.")

    def run(self):
        cv2.namedWindow('Reconhecimento de Gestos', cv2.WINDOW_NORMAL)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Falha na captura de vídeo")
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            screen_width = cv2.getWindowImageRect('Reconhecimento de Gestos')[2]

            if results.multi_hand_landmarks and not self.show_text_input:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = self.extract_landmarks(hand_landmarks)
                    self.process_gestures(image, landmarks)

            self.draw_ui_elements(image, screen_width)
            cv2.imshow('Reconhecimento de Gestos', image)

            key = cv2.waitKey(1) & 0xFF
            if self.show_text_input:
                self.process_text_input(key)
            else:
                if not self.handle_key_commands(key):
                    break

            # 🔧 MOD: reduzir cooldown por frame
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1

        self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    app = GestureRecognizer()
    app.run()
