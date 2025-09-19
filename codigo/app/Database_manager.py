import os
from Config import CONFIG

import sqlite3
import json
import logging

os.makedirs(os.path.dirname(CONFIG["db_path"]), exist_ok=True)

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
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

    def save_gestures(self, labels, data):
        self.conn.execute('DELETE FROM gestures')
        for name, landmarks in zip(labels, data):
            self.conn.execute(
                "INSERT INTO gestures (name, landmarks) VALUES (?, ?)",
                (name, json.dumps(landmarks))
            )
        self.conn.execute('DELETE FROM gesture_names')
        for name in set(labels):
            self.conn.execute(
                "INSERT OR IGNORE INTO gesture_names (name) VALUES (?)",
                (name,)
            )
        self.conn.commit()

    def load_gestures(self):
        labels, data, gesture_names = [], [], {}
        cursor = self.conn.execute("SELECT name FROM gesture_names")
        gesture_names = {name[0]: name[0] for name in cursor.fetchall()}

        cursor = self.conn.execute("SELECT name, landmarks FROM gestures")
        for name, landmarks_json in cursor.fetchall():
            try:
                landmarks = json.loads(landmarks_json)
                if len(landmarks) == 63:
                    labels.append(name)
                    data.append(landmarks)
            except json.JSONDecodeError:
                logging.error(f"Erro ao carregar gesto {name}")
        return labels, data, gesture_names

    def close(self):
        self.conn.close()
