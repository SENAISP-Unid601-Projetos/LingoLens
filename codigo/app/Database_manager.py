import os
import sqlite3
import json
import logging
from Config import CONFIG
from collections import Counter

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
            type TEXT NOT NULL,
            landmarks TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS gesture_names (
            name TEXT PRIMARY KEY NOT NULL,
            type TEXT NOT NULL
        )''')
        self.conn.commit()

    def save_gestures(self, labels, data, types=None):
        """
        Salva gestos no banco, preservando dados existentes.
        """
        if types is None:
            types = ['letter'] * len(labels)

        # Verificar balanceamento
        label_counts = Counter(labels)
        for name, count in label_counts.items():
            if count < CONFIG["min_samples_per_class"]:
                logging.warning(f"Classe '{name}' tem apenas {count} samples. Recomenda-se pelo menos {CONFIG['min_samples_per_class']}.")

        # Insere novos dados
        for name, landmarks, g_type in zip(labels, data, types):
            self.conn.execute(
                "INSERT INTO gestures (name, type, landmarks) VALUES (?, ?, ?)",
                (name, g_type, json.dumps(landmarks))
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO gesture_names (name, type) VALUES (?, ?)",
                (name, g_type)
            )

        self.conn.commit()

    def load_gestures(self, gesture_type=None):
        """
        Carrega gestos do banco.
        """
        labels, data, gesture_names = [], [], {}

        query_names = "SELECT name, type FROM gesture_names"
        if gesture_type:
            query_names += f" WHERE type='{gesture_type}'"
        cursor = self.conn.execute(query_names)
        gesture_names = {name: name for name, _ in cursor.fetchall()}

        query_gestures = "SELECT name, type, landmarks FROM gestures"
        if gesture_type:
            query_gestures += f" WHERE type='{gesture_type}'"
        cursor = self.conn.execute(query_gestures)

        for name, g_type, landmarks_json in cursor.fetchall():
            try:
                landmarks = json.loads(landmarks_json)
                if (isinstance(landmarks, list) and
                    (len(landmarks) >= 63 or isinstance(landmarks[0], list))):
                    labels.append(name)
                    data.append(landmarks)
            except json.JSONDecodeError:
                logging.error(f"Erro ao carregar gesto {name}")

        return labels, data, gesture_names

    def close(self):
        self.conn.close()