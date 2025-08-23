# Bibliotecas padrão do Python
import json
import logging
import sqlite3
import unicodedata
from pathlib import Path
from datetime import datetime

# Bibliotecas externas
import cv2
import mediapipe as mediapipe
import numpy as np

class GestureRecognizer:
    """Classe para reconhecimento de gestos de Libras usando MediaPipe e scikit-learn."""
    
    def __init__(self):
        """Inicializa o reconhecedor com configurações padrão, câmera e banco de dados."""
        if not self.check_dependencies():
            raise ImportError("Dependências ausentes. Programa encerrado.")
        
        # Configurações centralizadas
        self.config = {
            'db_path': 'gestures.db',
            'confidence_threshold': 0.85,
            'prediction_cooldown': 20,
            'camera_resolution': (640, 480),
            'target_fps': 30,
            'max_num_hands': 1,
            'min_detection_confidence': 0.7,
            'knn_neighbors': 3
        }
        
        self.DB_PATH = self.config['db_path']
        self.CONFIDENCE_THRESHOLD = self.config['confidence_threshold']
        self.prediction_cooldown = self.config['prediction_cooldown']
        
        # Inicializar logging
        logging.basicConfig(
            filename='gesture_recognizer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Inicializando GestureRecognizer")
        
        # Inicializar Mediapipe
        self.mp_hands = mediapipe.solutions.hands
        self.mp_drawing = mediapipe.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=self.config['max_num_hands'],
            min_detection_confidence=self.config['min_detection_confidence']
        )
        
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
        
        # Controle de repetição e cooldown
        self.last_prediction = ""
        self.cooldown_counter = 0
        self.cached_prediction = None
        
        # Gestos válidos para Libras (alfabeto e números)
        self.valid_gestures = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # Estado para exportação
        self.export_mode = False
        
        # Interface - Estado para menu de ajuda e mensagens de erro
        self.show_help = False
        self.error_message = ""
        self.error_message_timer = 0
        self.error_message_duration = 90  # 3 segundos a 30 FPS
        
        # 🔧 MOD: Tela 15.6 - Exibir resolução atual
        self.current_resolution = (0, 0)
        
        self.init_db()
        self.load_saved_data()
        
        self.cap = cv2.VideoCapture(0)
        self.set_camera_resolution(*self.config['camera_resolution'])
        self.cap.set(cv2.CAP_PROP_FPS, self.config['target_fps'])
        
        self.target_width = 1280
        self.ui_scale = 1.0

    def check_dependencies(self):
        """Verifica se todas as dependências estão instaladas."""
        required_modules = ['cv2', 'mediapipe', 'numpy', 'sklearn']
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                logging.error(f"Erro: Módulo '{module}' não está instalado.")
                print(f"Erro: Por favor, instale o módulo '{module}' usando 'pip install {module}'.")
                return False
        return True

    def __del__(self):
        """Libera recursos ao destruir o objeto."""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'conn'):
            self.conn.close()
        logging.info("Recursos liberados")

    def init_db(self):
        """Inicializa o banco de dados SQLite para armazenar gestos."""
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
        logging.info("Banco de dados inicializado")

    def load_saved_data(self):
        """Carrega dados salvos do banco de dados com validação."""
        logging.info("Carregando dados do banco de dados")
        try:
            cursor = self.conn.execute('SELECT name FROM gesture_names')
            self.gesture_names = {name[0]: name[0] for name in cursor.fetchall()}

            cursor = self.conn.execute('SELECT name, landmarks FROM gestures')
            for name, landmarks_json in cursor.fetchall():
                try:
                    landmarks = json.loads(landmarks_json)
                    if len(landmarks) == 63:
                        self.labels.append(name)
                        self.data.append(landmarks)
                    else:
                        logging.warning(f"Dados inválidos para gesto '{name}': tamanho incorreto dos landmarks")
                except json.JSONDecodeError:
                    logging.error(f"Erro: Dados inválidos para gesto '{name}' no banco de dados.")
                    continue
        except sqlite3.Error as e:
            logging.error(f"Erro ao acessar o banco de dados: {e}")
        finally:
            self.init_model()

    def init_model(self):
        """Inicializa o modelo KNN se houver dados suficientes."""
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=self.config['knn_neighbors'])
        if len(set(self.labels)) > 1:
            self.model.fit(self.data, self.labels)
            logging.info("Modelo KNN inicializado")

    def save_gesture_data(self):
        """Salva os dados de gestos no banco de dados."""
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
        logging.info("Dados de gestos salvos no banco de dados")

    def export_gestures_to_json(self, filename='gestures_export.json', filter_type=None):
        """Exporta os gestos salvos para um arquivo JSON com validação e metadados."""
        logging.info(f"Tentando exportar gestos para {filename}")
        try:
            base_path = Path(filename)
            if base_path.exists():
                counter = 1
                while base_path.exists():
                    base_path = Path(f"{filename.rsplit('.', 1)[0]}_{counter:03d}.json")
                    counter += 1
                filename = str(base_path)

            gestures = []
            for name, landmarks in zip(self.labels, self.data):
                if not isinstance(landmarks, (list, np.ndarray)) or len(landmarks) != 63:
                    logging.warning(f"Gesto '{name}' ignorado: landmarks inválidos")
                    continue
                if name not in self.valid_gestures:
                    logging.warning(f"Gesto '{name}' ignorado: não está em valid_gestures")
                    continue
                if filter_type == 'letters' and not name.isalpha():
                    continue
                if filter_type == 'numbers' and not name.isdigit():
                    continue
                gestures.append({'name': name, 'landmarks': np.array(landmarks).tolist()})

            if not gestures:
                logging.warning("Nenhum gesto válido para exportar")
                print("Nenhum gesto válido para exportar.")
                self.error_message = "Nenhum gesto válido para exportar"
                self.error_message_timer = self.error_message_duration
                return False

            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'gesture_count': len(gestures),
                    'filter_type': filter_type or 'all'
                },
                'gestures': gestures
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"{len(gestures)} gestos exportados para {filename}")
            logging.info(f"{len(gestures)} gestos exportados para {filename}")
            return True

        except Exception as e:
            logging.error(f"Erro ao exportar gestos: {e}")
            print(f"Erro ao exportar gestos: {e}")
            self.error_message = f"Erro ao exportar: {e}"
            self.error_message_timer = self.error_message_duration
            return False

    def set_camera_resolution(self, width, height):
        """Define a resolução da câmera."""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logging.info(f"Resolução da câmera definida para {width}x{height}")

    def extract_landmarks(self, hand_landmarks):
        """Extrai coordenadas (x, y, z) dos pontos de referência da mão."""
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        expected_size = 63
        if landmarks.size != expected_size:
            logging.error(f"Erro: Landmarks com tamanho inválido ({landmarks.size} != {expected_size})")
            self.error_message = "Landmarks inválidos detectados"
            self.error_message_timer = self.error_message_duration
            return None
        return landmarks

    def resize_with_aspect_ratio(self, image, target_width=None):
        """Redimensiona a imagem mantendo a proporção."""
        (h, w) = image.shape[:2]
        if target_width is None:
            return image
        ratio = target_width / float(w)
        dim = (target_width, int(h * ratio))
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def calculate_ui_scale(self, screen_width, screen_height):
        """Calcula a escala da UI com base na resolução da tela."""
        # 🔧 MOD: Tela 15.6 - Ajustar escala para resoluções comuns (1366x768, 1920x1080)
        base_width = 1366  # Otimizado para HD, comum em 15,6"
        scale = screen_width / base_width
        # Ajustar para Full HD (1920x1080) ou resoluções menores
        if screen_width >= 1920:
            scale *= 1.2  # Aumentar ligeiramente para Full HD
        return max(0.8, min(1.5, scale))  # Limitar entre 0.8 e 1.5

    def draw_ui_elements(self, image, screen_width, screen_height):
        """Desenha elementos da interface do usuário otimizados para tela de 15,6 polegadas."""
        # 🔧 MOD: Tela 15.6 - Passar altura para ui_scale
        self.ui_scale = self.calculate_ui_scale(screen_width, screen_height)
        height, width = image.shape[:2]

        # 🔧 MOD: Tela 15.6 - Seções otimizadas
        top_bar_height = int(50 * self.ui_scale)
        bottom_bar_height = int(60 * self.ui_scale)
        left_panel_width = int(width * 0.25)  # Reduzido para evitar sobreposição
        right_panel_width = int(width * 0.25)

        # 🔧 MOD: Tela 15.6 - Barra superior (status)
        cv2.rectangle(image, (0, 0), (width, top_bar_height), (30, 30, 30), -1)
        status_text = f'Modo: {"Treino" if self.training_mode else "Reconhecimento"} | '
        status_text += f'Entrada: {"Número" if self.number_mode else "Letra"}'
        cv2.putText(image, status_text,
                    (int(10 * self.ui_scale), int(35 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.ui_scale,
                    (255, 255, 255), int(2 * self.ui_scale), lineType=cv2.LINE_AA)

        # 🔧 MOD: Tela 15.6 - Indicador de resolução
        res_text = f"Res: {self.current_resolution[0]}x{self.current_resolution[1]}"
        cv2.putText(image, res_text,
                    (width - int(150 * self.ui_scale), int(35 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                    (200, 200, 200), int(2 * self.ui_scale), lineType=cv2.LINE_AA)

        # 🔧 MOD: Tela 15.6 - Palavra formada (centro superior)
        cv2.rectangle(image, (width//2 - int(150 * self.ui_scale), top_bar_height),
                      (width//2 + int(150 * self.ui_scale), top_bar_height + int(60 * self.ui_scale)),
                      (50, 50, 50, 200), -1)
        cv2.putText(image, f'Palavra: {self.current_word}',
                    (width//2 - int(140 * self.ui_scale), top_bar_height + int(45 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * self.ui_scale,
                    (255, 255, 255), int(3 * self.ui_scale), lineType=cv2.LINE_AA)

        # 🔧 MOD: Tela 15.6 - Painel inferior (instruções compactas)
        cv2.rectangle(image, (0, height - bottom_bar_height), (width, height), (30, 30, 30), -1)
        instructions = "Q:Sair C:Limpar N:Num/Letra T:Treino S:Gesto E:Exportar H:Ajuda"
        cv2.putText(image, instructions,
                    (int(10 * self.ui_scale), height - int(15 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                    (200, 200, 200), int(2 * self.ui_scale), lineType=cv2.LINE_AA)

        # 🔧 MOD: Tela 15.6 - Retângulo de foco menor
        focus_size = int(min(width, height) * 0.35)  # Reduzido para telas menores
        focus_x = (width - focus_size) // 2
        focus_y = (height - focus_size) // 2
        cv2.rectangle(image, (focus_x, focus_y), (focus_x + focus_size, focus_y + focus_size),
                      (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(image, "Posicione a mão aqui",
                    (focus_x, focus_y - int(10 * self.ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.ui_scale,
                    (255, 255, 255), int(2 * self.ui_scale), lineType=cv2.LINE_AA)

        # 🔧 MOD: Tela 15.6 - Barra de progresso para cooldown (maior)
        if self.cooldown_counter > 0:
            bar_width = int(150 * self.ui_scale)
            bar_height = int(15 * self.ui_scale)
            progress = self.cooldown_counter / self.prediction_cooldown
            filled_width = int(bar_width * (1 - progress))
            cv2.rectangle(image, (int(10 * self.ui_scale), int(80 * self.ui_scale)),
                          (int(10 * self.ui_scale) + bar_width, int(80 * self.ui_scale) + bar_height),
                          (100, 100, 100), -1)
            cv2.rectangle(image, (int(10 * self.ui_scale), int(80 * self.ui_scale)),
                          (int(10 * self.ui_scale) + filled_width, int(80 * self.ui_scale) + bar_height),
                          (255, 165, 0), -1)

        # 🔧 MOD: Tela 15.6 - Menu de ajuda compacto
        if self.show_help:
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
            help_text = [
                "Instruções:",
                "Q: Sair",
                "C: Limpar palavra",
                "N: Alternar número/letra",
                "T: Modo treino",
                "S: Novo gesto (A-Z, 0-9)",
                "E: Exportar JSON",
                "H: Mostrar/esconder ajuda",
                "Posicione a mão no retângulo",
                "Pressione H para voltar"
            ]
            for i, line in enumerate(help_text):
                cv2.putText(image, line,
                            (int(20 * self.ui_scale), int(80 * self.ui_scale) + i * int(30 * self.ui_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                            (255, 255, 255), int(2 * self.ui_scale), lineType=cv2.LINE_AA)

        # 🔧 MOD: Tela 15.6 - Mensagem de erro temporária
        if self.error_message_timer > 0:
            cv2.rectangle(image, (0, height - bottom_bar_height - int(40 * self.ui_scale)),
                          (width, height - bottom_bar_height), (200, 50, 50), -1)
            cv2.putText(image, self.error_message,
                        (int(10 * self.ui_scale), height - bottom_bar_height - int(10 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                        (255, 255, 255), int(2 * self.ui_scale), lineType=cv2.LINE_AA)
            self.error_message_timer -= 1

        if self.show_text_input:
            overlay = image.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

            box_width, box_height = int(500 * self.ui_scale), int(120 * self.ui_scale)
            x = (width - box_width) // 2
            y = (height - box_height) // 2

            cv2.rectangle(image, (x, y), (x + box_width, y + box_height), (50, 50, 50), -1)
            cv2.rectangle(image, (x, y), (x + box_width, y + box_height), (255, 255, 255), 2)

            cv2.putText(image, self.input_prompt,
                        (x + int(15 * self.ui_scale), y + int(30 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                        (255, 255, 255), int(2 * self.ui_scale), lineType=cv2.LINE_AA)
            cv2.putText(image, self.input_text,
                        (x + int(15 * self.ui_scale), y + int(80 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9 * self.ui_scale,
                        (255, 255, 255), int(3 * self.ui_scale), lineType=cv2.LINE_AA)
            cv2.putText(image, "Enter: confirmar | Esc: cancelar",
                        (x + int(15 * self.ui_scale), y + int(110 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self.ui_scale,
                        (200, 200, 200), int(2 * self.ui_scale), lineType=cv2.LINE_AA)

    def process_gestures(self, image, landmarks):
        """Processa os gestos detectados, incluindo treino e reconhecimento."""
        if landmarks is None:
            return
        if self.show_text_input or self.show_help:
            return

        if self.training_mode and self.current_gesture_name:
            self.data.append(landmarks)
            self.labels.append(self.current_gesture_name)

            font_scale = 0.9 * self.ui_scale
            cv2.putText(image, f"Coletando: {self.current_gesture_name}",
                        (int(10 * self.ui_scale), int(120 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 255), int(3 * self.ui_scale), lineType=cv2.LINE_AA)
            sample_count = sum(1 for label in self.labels if label == self.current_gesture_name)
            cv2.putText(image, f"Amostras: {sample_count}",
                        (int(10 * self.ui_scale), int(160 * self.ui_scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * self.ui_scale,
                        (0, 255, 255), int(2 * self.ui_scale), lineType=cv2.LINE_AA)
            return

        if not self.training_mode and len(set(self.labels)) > 1:
            if self.cooldown_counter == 0:
                prediction = self.model.predict([landmarks])[0]
                probability = self.model.predict_proba([landmarks]).max()
                self.cached_prediction = (prediction, probability)
            else:
                prediction, probability = self.cached_prediction

            font_scale = 1.0 * self.ui_scale
            if probability >= self.CONFIDENCE_THRESHOLD:
                label = self.gesture_names.get(prediction, prediction)
                if self.cooldown_counter == 0:
                    if label != self.last_prediction:
                        if (self.number_mode and label.isdigit()) or (not self.number_mode and not label.isdigit()):
                            self.current_word += label
                            self.last_prediction = label
                            self.cooldown_counter = self.prediction_cooldown
                            logging.info(f"Gesto reconhecido: {label} ({probability*100:.1f}%)")
                            if self.current_word and self.current_word[-1] != ' ':
                                if self.cooldown_counter > self.prediction_cooldown * 2:
                                    self.current_word += ' '

                cv2.putText(image, f'{label} ({probability*100:.1f}%)',
                            (int(10 * self.ui_scale), int(80 * self.ui_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 255, 0), int(3 * self.ui_scale), lineType=cv2.LINE_AA)
            else:
                cv2.putText(image, 'Desconhecido',
                            (int(10 * self.ui_scale), int(80 * self.ui_scale)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 255), int(3 * self.ui_scale), lineType=cv2.LINE_AA)

    def handle_key_commands(self, key):
        """Processa comandos de teclado para controle do aplicativo."""
        if self.show_text_input or self.show_help:
            if key == ord('h') and self.show_help:
                self.show_help = False
                return True
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
            self.export_mode = False
        elif key == ord('e'):
            self.show_text_input = True
            self.input_text = "gestures_export.json"
            self.input_prompt = "Digite o nome do arquivo para exportação (ex.: gestos.json):"
            self.export_mode = True
        elif key == ord('h'):
            self.show_help = not self.show_help
        return True

    def process_text_input(self, key):
        """Processa entrada de texto para nomear gestos ou arquivos de exportação."""
        if not self.show_text_input:
            return
        if key == 13:  # Enter
            if self.input_text:
                normalized_text = ''.join(c for c in unicodedata.normalize('NFD', self.input_text)
                                        if unicodedata.category(c) != 'Mn').strip()
                if self.export_mode:
                    if not normalized_text.endswith('.json'):
                        normalized_text += '.json'
                    self.export_gestures_to_json(normalized_text)
                    self.export_mode = False
                else:
                    normalized_text = normalized_text.upper()
                    if normalized_text in self.valid_gestures:
                        self.current_gesture_name = normalized_text
                        self.gesture_names[normalized_text] = normalized_text
                        print(f"Gesto '{self.current_gesture_name}' pronto para treinamento!")
                        logging.info(f"Novo gesto criado: {self.current_gesture_name}")
                    else:
                        print(f"Erro: '{normalized_text}' não é um gesto válido. Use letras (A-Z) ou números (0-9).")
                        logging.warning(f"Tentativa de criar gesto inválido: {normalized_text}")
                        self.error_message = f"Gesto '{normalized_text}' inválido"
                        self.error_message_timer = self.error_message_duration
                        return
            self.show_text_input = False
            self.input_text = ""
        elif key == 27:  # Esc
            self.show_text_input = False
            self.input_text = ""
            self.current_gesture_name = ""
            self.export_mode = False
        elif key == 8:  # Backspace
            self.input_text = self.input_text[:-1]
        elif 32 <= key <= 126 or key in [192, 193, 194, 195, 199, 231]:
            self.input_text += chr(key)

    def train_and_save_model(self):
        """Treina o modelo KNN e salva os dados."""
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=self.config['knn_neighbors'])
        if len(set(self.labels)) > 1:
            self.model.fit(self.data, self.labels)
            self.save_gesture_data()
            print("Modelo treinado e salvo no banco de dados.")
            logging.info("Modelo treinado e salvo")
        else:
            print("É necessário mais de um gesto diferente para treinar o modelo.")
            logging.warning("Tentativa de treino com dados insuficientes")
            self.error_message = "Mais de um gesto necessário para treino"
            self.error_message_timer = self.error_message_duration

    def run(self):
        """Executa o loop principal do aplicativo."""
        if not self.cap.isOpened():
            print("Erro: Não foi possível abrir a câmera.")
            logging.error("Falha ao abrir a câmera")
            self.error_message = "Falha ao abrir a câmera"
            self.error_message_timer = self.error_message_duration
            return

        cv2.namedWindow('Reconhecimento de Gestos', cv2.WINDOW_NORMAL)
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Falha na captura de vídeo")
                    logging.error("Falha na captura de vídeo")
                    self.error_message = "Falha na captura de vídeo"
                    self.error_message_timer = self.error_message_duration
                    break

                image = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)

                # 🔧 MOD: Tela 15.6 - Obter resolução atual
                _, _, screen_width, screen_height = cv2.getWindowImageRect('Reconhecimento de Gestos')
                self.current_resolution = (screen_width, screen_height)

                if results.multi_hand_landmarks and not self.show_text_input and not self.show_help:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        landmarks = self.extract_landmarks(hand_landmarks)
                        self.process_gestures(image, landmarks)
                else:
                    if not self.show_text_input and not self.show_help:
                        self.error_message = "Nenhuma mão detectada"
                        self.error_message_timer = self.error_message_duration

                self.draw_ui_elements(image, screen_width, screen_height)
                cv2.imshow('Reconhecimento de Gestos', image)

                key = cv2.waitKey(1) & 0xFF
                if self.show_text_input:
                    self.process_text_input(key)
                else:
                    if not self.handle_key_commands(key):
                        break

                if self.cooldown_counter > 0:
                    self.cooldown_counter -= 1

        except Exception as e:
            print(f"Erro durante a execução: {e}")
            logging.error(f"Erro durante a execução: {e}")
            self.error_message = f"Erro: {e}"
            self.error_message_timer = self.error_message_duration
        finally:
            self.__del__()

if __name__ == "__main__":
    app = GestureRecognizer()
    app.run()

# 🔧 Seção de Testes Unitários
import unittest

class TestGestureRecognizer(unittest.TestCase):
    def test_extract_landmarks(self):
        app = GestureRecognizer()
        class MockLandmark:
            def __init__(self): self.x, self.y, self.z = 0.1, 0.2, 0.3
        class MockHandLandmarks:
            landmark = [MockLandmark() for _ in range(21)]
        landmarks = app.extract_landmarks(MockHandLandmarks())
        self.assertEqual(landmarks.size, 63)
        self.assertIsNotNone(landmarks)

    def test_export_gestures_to_json(self):
        app = GestureRecognizer()
        app.labels = ['A', 'B']
        app.data = [np.array([0.1] * 63), np.array([0.2] * 63)]
        result = app.export_gestures_to_json('test_export.json')
        self.assertTrue(result)
        self.assertTrue(Path('test_export.json').exists())

if __name__ == '__main__':
    unittest.main()
