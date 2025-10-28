import sqlite3
import pickle
import logging
import os

class DatabaseManager:
    def __init__(self, db_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if db_path is None:
            db_path = os.path.join(base_dir, "data", "gestures.db")
        self.db_path = db_path
        self.base_dir = base_dir
        self._ensure_directory()
        
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._create_tables()
            log_path = os.path.join(base_dir, "logs", "database.log")
            logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")
            logging.info(f"Banco aberto: {db_path}")
            print(f"[INFO] Banco aberto: {db_path}")
        except Exception as e:
            error_msg = f"Erro ao abrir banco: {e}"
            print(f"[FATAL] {error_msg}")
            logging.error(error_msg)
            raise

    def _ensure_directory(self):
        directory = os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"[INFO] Pasta criada: {directory}")
            logging.info(f"Pasta criada: {directory}")

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
                accuracy REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS model_lstm (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weights BLOB NOT NULL,
                classes BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()
        logging.info("Tabelas criadas/verificados")

    def save_gestures(self, labels, data_list, gesture_types):
        data = [(l, pickle.dumps(d), t) for l, d, t in zip(labels, data_list, gesture_types)]
        self.cursor.executemany("INSERT INTO gestures (label, data, gesture_type) VALUES (?, ?, ?)", data)
        self.conn.commit()

    def save_dynamic_gestures(self, labels, sequences):
        for label, seq in zip(labels, sequences):
            if len(seq) == 0: continue
            blob = pickle.dumps(seq)
            self.cursor.execute("SELECT id FROM dynamics WHERE label = ?", (label,))
            if self.cursor.fetchone():
                self.cursor.execute("UPDATE dynamics SET sequence = ? WHERE label = ?", (blob, label))
            else:
                self.cursor.execute("INSERT INTO dynamics (label, sequence) VALUES (?, ?)", (label, blob))
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

    def load_dynamic_gestures(self):
        self.cursor.execute("SELECT label, sequence FROM dynamics")
        rows = self.cursor.fetchall()
        X, y = [], []
        for label, blob in rows:
            seq = pickle.loads(blob)
            if isinstance(seq, list):
                X.extend(seq)
                y.extend([label] * len(seq))
            else:
                X.append(seq)
                y.append(label)
        return X, y

    def save_static_model(self, model, accuracy):
        blob = pickle.dumps(model)
        self.cursor.execute("DELETE FROM model_static")
        self.cursor.execute("INSERT INTO model_static (model, accuracy) VALUES (?, ?)", (blob, accuracy))
        self.conn.commit()

    def load_static_model(self):
        self.cursor.execute("SELECT model FROM model_static ORDER BY id DESC LIMIT 1")
        row = self.cursor.fetchone()
        return pickle.loads(row[0]) if row else None

    def save_lstm_model(self, model):
        weights = model.get_weights()
        classes = getattr(model, 'classes_', [])
        w_blob = pickle.dumps(weights)
        c_blob = pickle.dumps(classes)
        self.cursor.execute("DELETE FROM model_lstm")
        self.cursor.execute("INSERT INTO model_lstm (weights, classes) VALUES (?, ?)", (w_blob, c_blob))
        self.conn.commit()

    def load_lstm_model(self, model_structure):
        self.cursor.execute("SELECT weights, classes FROM model_lstm ORDER BY id DESC LIMIT 1")
        row = self.cursor.fetchone()
        if row:
            weights = pickle.loads(row[0])
            model_structure.set_weights(weights)
            return pickle.loads(row[1])
        return None

    def close(self):
        if self.conn:
            self.conn.close()