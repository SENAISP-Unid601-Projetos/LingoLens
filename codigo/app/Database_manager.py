import os
import sqlite3
import json
import logging
from Config import CONFIG

os.makedirs(os.path.dirname(CONFIG["db_path"]), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)

class DatabaseManager:
    def __init__(self, db_path=CONFIG["db_path"]):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS gestures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            landmarks TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
        )

        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS gesture_names (
            name TEXT PRIMARY KEY NOT NULL,
            type TEXT NOT NULL
        )"""
        )
        self.conn.commit()

    def add_gesture(self, name, landmarks, g_type="letter"):
        try:
            landmarks_json = json.dumps(landmarks.tolist() if hasattr(landmarks, 'tolist') else landmarks)
            self.conn.execute(
                "INSERT INTO gestures (name, type, landmarks) VALUES (?, ?, ?)",
                (name, g_type, landmarks_json)
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO gesture_names (name, type) VALUES (?, ?)",
                (name, g_type)
            )
            self.conn.commit()
            return True
        except Exception as e:
            logging.error(f"Erro ao adicionar gesto: {e}")
            return False

    def save_gestures(self, labels, data, types=None):
        if types is None:
            types = ["letter"] * len(labels)

        self.conn.execute("DELETE FROM gestures")
        self.conn.execute("DELETE FROM gesture_names")

        for name, landmarks, g_type in zip(labels, data, types):
            self.conn.execute(
                "INSERT INTO gestures (name, type, landmarks) VALUES (?, ?, ?)",
                (name, g_type, json.dumps(landmarks)),
            )

        for name, g_type in zip(labels, types):
            self.conn.execute(
                "INSERT OR IGNORE INTO gesture_names (name, type) VALUES (?, ?)",
                (name, g_type),
            )

        self.conn.commit()

    def save_movements(self, labels, data):
        return self.save_gestures(labels, data, types=["movement"] * len(labels))

    def load_gestures(self, gesture_type=None):
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
                if isinstance(landmarks, list) and (
                    len(landmarks) == 63 or isinstance(landmarks[0], list)
                ):
                    labels.append(name)
                    data.append(landmarks)
            except json.JSONDecodeError:
                logging.error(f"Erro ao carregar gesto {name}")

        return labels, data, gesture_names

    def load_movements(self):
        return self.load_gestures(gesture_type="movement")

    def delete_gesture(self, gesture_name):
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM gestures WHERE name=?", (gesture_name,)
        )
        if cursor.fetchone()[0] == 0:
            return False

        self.conn.execute("DELETE FROM gestures WHERE name=?", (gesture_name,))
        self.conn.execute("DELETE FROM gesture_names WHERE name=?", (gesture_name,))
        self.conn.commit()
        return True

    def close(self):
        self.conn.close()