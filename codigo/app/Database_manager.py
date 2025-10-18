import sqlite3
import pickle
import os
import logging

logging.basicConfig(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "Gesture_recognizer.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS gestures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture_type TEXT NOT NULL,
                data BLOB NOT NULL
            )
        ''')
        self.conn.commit()

    def save_gestures(self, labels, data, gesture_types):
        # Assume all gesture_types are the same
        gesture_type = gesture_types[0]
        existing_data = self.load_gestures(gesture_type)

        # Combine new data with existing
        for label, datum in zip(labels, data):
            if label not in existing_data:
                existing_data[label] = []
            existing_data[label].append(datum)

        serialized_data = pickle.dumps(existing_data)

        self.cursor.execute('''
            SELECT id FROM gestures WHERE gesture_type = ?
        ''', (gesture_type,))
        result = self.cursor.fetchone()

        if result:
            self.cursor.execute('''
                UPDATE gestures SET data = ? WHERE gesture_type = ?
            ''', (serialized_data, gesture_type))
        else:
            self.cursor.execute('''
                INSERT INTO gestures (gesture_type, data)
                VALUES (?, ?)
            ''', (gesture_type, serialized_data))
        self.conn.commit()
        logging.info(f"Gestos salvos para {gesture_type}")

    def load_gestures(self, gesture_type):
        self.cursor.execute('''
            SELECT data FROM gestures WHERE gesture_type = ?
        ''', (gesture_type,))
        result = self.cursor.fetchone()
        if result:
            return pickle.loads(result[0])
        return {}

    def close(self):
        self.conn.close()