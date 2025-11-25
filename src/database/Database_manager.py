import sqlite3
import pickle
import os
import logging
from config.Config import CONFIG

log_dir = os.path.dirname(CONFIG["log_file"])
os.makedirs(log_dir, exist_ok=True)

# Configura o logging usando o mesmo arquivo da raiz
logging.basicConfig(
    filename=CONFIG["log_file"],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    force=True  # força reconfiguração mesmo se já existir
)

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self._initialize_database()
            self._upgrade_static_data_to_67_features()  # CONVERSÃO AUTOMÁTICA
            print(f"[INFO] Banco conectado: {db_path}")
        except sqlite3.Error as e:
            logging.error(f"Erro ao conectar: {e}")
            raise

    def _initialize_database(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_type TEXT NOT NULL,
                    gesture_name TEXT NOT NULL,
                    data BLOB NOT NULL
                )
            ''')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_gesture ON gestures (gesture_type, gesture_name)')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS gestures_dynamic (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_name TEXT NOT NULL,
                    data BLOB NOT NULL
                )
            ''')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_dynamic ON gestures_dynamic (gesture_name)')

            self.cursor.execute("SELECT COUNT(DISTINCT gesture_name) FROM gestures WHERE gesture_type = 'letter'")
            static_count = self.cursor.fetchone()[0]
            self.cursor.execute("SELECT COUNT(DISTINCT gesture_name) FROM gestures_dynamic")
            dynamic_count = self.cursor.fetchone()[0]

            self.conn.commit()
            print(f"[INFO] Tabelas OK: {static_count} estáticas | {dynamic_count} dinâmicas")
        except sqlite3.Error as e:
            logging.error(f"Erro ao criar tabelas: {e}")
            raise

    def _upgrade_static_data_to_67_features(self):
        """Converte dados antigos (65 features) → 67 features"""
        try:
            self.cursor.execute("SELECT gesture_name, data FROM gestures WHERE gesture_type = 'letter'")
            results = self.cursor.fetchall()
            updated = 0
            for name, blob in results:
                try:
                    old_data = pickle.loads(blob)
                    if isinstance(old_data, dict):
                        new_data = {}
                        for label, samples in old_data.items():
                            new_samples = [s + [0.0, 0.0] if len(s) == 65 else s for s in samples]
                            new_data[label] = new_samples
                    else:
                        new_data = [s + [0.0, 0.0] if len(s) == 65 else s for s in old_data]
                    new_blob = pickle.dumps(new_data)
                    with self.conn:
                        self.cursor.execute("UPDATE gestures SET data = ? WHERE gesture_name = ?", (new_blob, name))
                    updated += 1
                except Exception as e:
                    logging.error(f"Erro ao converter {name}: {e}")
            if updated > 0:
                print(f"[INFO] {updated} gestos convertidos para 67 features")
            else:
                print(f"[INFO] Todos os gestos já estão em 67 features")
        except Exception as e:
            logging.error(f"Erro na conversão: {e}")

    def save_gestures(self, labels, data, gesture_name, is_dynamic=False):
        # validações básicas
        if not labels or not data or not gesture_name:
            return False
        if not all(l == gesture_name for l in labels):
            return False

        try:
            serialized = pickle.dumps(data)
            table = "gestures_dynamic" if is_dynamic else "gestures"
            with self.conn:
                self.cursor.execute(f"SELECT id FROM {table} WHERE gesture_name = ?", (gesture_name,))
                if self.cursor.fetchone():
                    self.cursor.execute(f"UPDATE {table} SET data = ? WHERE gesture_name = ?", (serialized, gesture_name))
                    action = "atualizado"
                else:
                    if is_dynamic:
                        self.cursor.execute(f"INSERT INTO {table} (gesture_name, data) VALUES (?, ?)", (gesture_name, serialized))
                    else:
                        self.cursor.execute(f"INSERT INTO {table} (gesture_type, gesture_name, data) VALUES (?, ?, ?)",
                                          ("letter", gesture_name, serialized))
                    action = "criado"
            # usar '->' em vez de '→' para evitar problemas de encoding em alguns consoles
            print(f"[INFO] {gesture_name} {action} -> {len(data)} amostras")
            logging.info(f"{gesture_name} {action} -> {len(data)} amostras")
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar {gesture_name}: {e}")
            return False

    def load_gestures(self, is_dynamic=False):
        try:
            table = "gestures_dynamic" if is_dynamic else "gestures"
            query = f"SELECT gesture_name, data FROM {table}"
            if not is_dynamic:
                query += " WHERE gesture_type = 'letter'"
            self.cursor.execute(query)
            results = self.cursor.fetchall()

            data_dict = {}
            labels = []
            data = []
            names = []

            for name, blob in results:
                try:
                    gesture_data = pickle.loads(blob)
                    names.append(name)
                    if isinstance(gesture_data, dict):
                        data_dict.update(gesture_data)
                        for label, landmarks in gesture_data.items():
                            labels.extend([label] * len(landmarks))
                            data.extend(landmarks)
                    else:
                        labels.extend([name] * len(gesture_data))
                        data.extend(gesture_data)
                except Exception as e:
                    logging.error(f"Erro ao ler {name}: {e}")
                    continue

            table_type = "dinâmica" if is_dynamic else "estática"
            print(f"[INFO] Carregou {len(names)} gestos {table_type}")
            return data_dict, labels, data, names
        except Exception as e:
            logging.error(f"Erro ao carregar: {e}")
            return {}, [], [], []

    def list_gestures(self, is_dynamic=False):
        table = "gestures_dynamic" if is_dynamic else "gestures"
        query = f"SELECT DISTINCT gesture_name FROM {table}"
        if not is_dynamic:
            query += " WHERE gesture_type = 'letter'"
        self.cursor.execute(query)
        return [row[0] for row in self.cursor.fetchall()]

    def delete_gesture(self, gesture_name, is_dynamic=False):
        table = "gestures_dynamic" if is_dynamic else "gestures"
        with self.conn:
            self.cursor.execute(f"DELETE FROM {table} WHERE gesture_name = ?", (gesture_name,))
        deleted = self.cursor.rowcount > 0
        if deleted:
            print(f"[INFO] {gesture_name} deletado")
            logging.info(f"{gesture_name} deletado")
        return deleted

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
            print("[INFO] Banco fechado")
        except:
            pass
