import sqlite3
import pickle
import os
import logging

logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "Gesture_recognizer.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self.create_table()
        except sqlite3.Error as e:
            logging.error(f"Erro ao conectar ao banco de dados: {e}")
            raise

    def create_table(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_type TEXT NOT NULL,
                    gesture_name TEXT NOT NULL,
                    data BLOB NOT NULL
                )
            ''')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_gesture_type_name ON gestures (gesture_type, gesture_name)')
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Erro ao criar tabela: {e}")
            raise

    def save_gestures(self, labels, data, gesture_types):
        if not labels or not data or not gesture_types or len(labels) != len(data) or len(labels) != len(gesture_types):
            logging.error("Entradas inválidas para labels, data ou gesture_types")
            raise ValueError("Entradas inválidas para labels, data ou gesture_types")
        
        gesture_type = gesture_types[0]
        if not all(t == gesture_type for t in gesture_types):
            logging.error("Todos os gesture_types devem ser iguais")
            raise ValueError("Todos os gesture_types devem ser iguais")
        
        try:
            existing_data = self.load_gestures(gesture_type)[0]
            for label, datum in zip(labels, data):
                if label not in existing_data:
                    existing_data[label] = []
                existing_data[label].append(datum)

            serialized_data = pickle.dumps(existing_data)
            with self.conn:
                for label in set(labels):
                    self.cursor.execute('SELECT id FROM gestures WHERE gesture_type = ? AND gesture_name = ?', 
                                     (gesture_type, label))
                    result = self.cursor.fetchone()
                    if result:
                        self.cursor.execute('UPDATE gestures SET data = ? WHERE gesture_type = ? AND gesture_name = ?', 
                                         (serialized_data, gesture_type, label))
                    else:
                        self.cursor.execute('INSERT INTO gestures (gesture_type, gesture_name, data) VALUES (?, ?, ?)', 
                                         (gesture_type, label, serialized_data))
            logging.info(f"Gestos salvos para {gesture_type}")
        except (sqlite3.Error, pickle.PickleError) as e:
            logging.error(f"Erro ao salvar gestos para {gesture_type}: {e}")
            raise

    def load_gestures(self, gesture_type):
        try:
            self.cursor.execute('SELECT gesture_name, data FROM gestures WHERE gesture_type = ?', (gesture_type,))
            results = self.cursor.fetchall()
            data_dict = {}
            labels = []
            data = []
            gesture_names = []
            for gesture_name, blob in results:
                gesture_data = pickle.loads(blob)
                data_dict.update(gesture_data)
                gesture_names.append(gesture_name)
                for label, landmarks_list in gesture_data.items():
                    labels.extend([label] * len(landmarks_list))
                    data.extend(landmarks_list)
            return data_dict, labels, data, gesture_names
        except (sqlite3.Error, pickle.PickleError) as e:
            logging.error(f"Erro ao carregar gestos para {gesture_type}: {e}")
            return {}, [], [], []

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except sqlite3.Error as e:
            logging.error(f"Erro ao fechar conexão com o banco de dados: {e}")