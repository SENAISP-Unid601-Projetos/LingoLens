import sqlite3
import pickle
import os
import numpy as np

class DatabaseManager:
    def __init__(self, db_path=os.path.join('app', 'data', 'gestures.db')):
        self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), db_path))
        self.create_table()

    def create_table(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    landmarks BLOB NOT NULL
                )
            ''')
            conn.commit()

    def save_gestures(self, gesture_name, gesture_type, landmarks):
        try:
            existing_dict = self.load_gestures(gesture_type)
            if gesture_name not in existing_dict:
                existing_dict[gesture_name] = []
            existing_dict[gesture_name].extend(landmarks)
            print(f"[DEBUG] Gestos existentes antes de salvar ({gesture_type}): {existing_dict.keys()}")
            
            serialized_data = pickle.dumps(existing_dict)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id FROM gestures WHERE type = ?
                ''', (gesture_type,))
                result = cursor.fetchone()
                
                if result:
                    cursor.execute('''
                        UPDATE gestures SET landmarks = ? WHERE type = ?
                    ''', (serialized_data, gesture_type))
                else:
                    cursor.execute('''
                        INSERT INTO gestures (name, type, landmarks)
                        VALUES (?, ?, ?)
                    ''', (gesture_name, gesture_type, serialized_data))
                conn.commit()
            print(f"[INFO] Gestos de '{gesture_name}' ({gesture_type}) salvos no banco de dados.")
        except Exception as e:
            print(f"[ERROR] Erro ao salvar gestos: {e}")

    def load_gestures(self, gesture_type):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT landmarks FROM gestures WHERE type = ?
                ''', (gesture_type,))
                result = cursor.fetchone()
                if result:
                    return pickle.loads(result[0])
                return {}
        except Exception as e:
            print(f"[ERROR] Erro ao carregar gestos: {e}")
            return {}