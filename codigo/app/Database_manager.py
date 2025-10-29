import sqlite3
import pickle
import os
from Config import get_logger, CONFIG

logger = get_logger("Database")

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = db_path or CONFIG["db_path"]
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._ensure_directory()
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._create_tables()
            logger.info(f"Banco aberto: {self.db_path}")
        except Exception as e:
            logger.error(f"Erro ao abrir banco: {e}")
            raise

    def _ensure_directory(self):
        dir_path = os.path.dirname(self.db_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Pasta criada: {dir_path}")

    def _create_tables(self):
        self.cursor.executescript("""
            CREATE TABLE IF NOT EXISTS gestures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                data BLOB NOT NULL,
                gesture_type TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dynamics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                sequence BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_static (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model BLOB NOT NULL,
                accuracy REAL
            );
            CREATE TABLE IF NOT EXISTS model_lstm (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weights BLOB NOT NULL,
                classes BLOB NOT NULL
            );
        """)
        self.conn.commit()

    def save_gestures(self, labels, data_list, gesture_types):
        data = [(l, pickle.dumps(d), t) for l, d, t in zip(labels, data_list, gesture_types)]
        self.cursor.executemany("INSERT INTO gestures (label, data, gesture_type) VALUES (?, ?, ?)", data)
        self.conn.commit()

    def save_dynamic_gesture(self, label, sequence):
        blob = pickle.dumps(sequence)
        self.cursor.execute("SELECT sequence FROM dynamics WHERE label = ?", (label,))
        row = self.cursor.fetchone()
        if row:
            seqs = pickle.loads(row[0])
            if not isinstance(seqs, list):
                seqs = [seqs]
            seqs.append(sequence)
            blob = pickle.dumps(seqs)
            self.cursor.execute("UPDATE dynamics SET sequence = ? WHERE label = ?", (blob, label))
        else:
            self.cursor.execute("INSERT INTO dynamics (label, sequence) VALUES (?, ?)", (label, pickle.dumps([sequence])))
        self.conn.commit()

    def load_gestures(self, gesture_type="letter"):
        self.cursor.execute("SELECT label, data FROM gestures WHERE gesture_type = ?", (gesture_type,))
        rows = self.cursor.fetchall()
        data_dict, labels, data = {}, [], []
        for label, blob in rows:
            d = pickle.loads(blob)
            data_dict.setdefault(label, []).append(d)
            labels.append(label)
            data.append(d)
        return data_dict, labels, data, list(data_dict.keys())

    def load_all_dynamic_gestures(self):
        self.cursor.execute("SELECT label, sequence FROM dynamics")
        result = {}
        for label, blob in self.cursor.fetchall():
            seq = pickle.loads(blob)
            if isinstance(seq, list):
                result.setdefault(label, []).extend(seq)
            else:
                result.setdefault(label, []).append(seq)
        return result

    def list_gestures(self, gesture_type):
        self.cursor.execute("SELECT DISTINCT label FROM gestures WHERE gesture_type = ?", (gesture_type,))
        return [row[0] for row in self.cursor.fetchall()]

    def list_dynamic_gestures(self):
        self.cursor.execute("SELECT DISTINCT label FROM dynamics")
        return [row[0] for row in self.cursor.fetchall()]

    def delete_gesture(self, gesture_type, label):
        self.cursor.execute("DELETE FROM gestures WHERE gesture_type = ? AND label = ?", (gesture_type, label))
        self.conn.commit()

    def delete_dynamic_gesture(self, label):
        self.cursor.execute("DELETE FROM dynamics WHERE label = ?", (label,))
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()