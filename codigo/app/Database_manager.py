import sqlite3
import pickle
import os
import logging
from Config import CONFIG

# Configurar logging usando CONFIG["log_file"]
logging.basicConfig(
    filename=CONFIG["log_file"],
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
            self.create_dynamic_gestures_table()
            logging.info("Banco de dados inicializado com sucesso")
        except sqlite3.Error as e:
            logging.error(f"Erro ao conectar ao banco de dados: {e}")
            raise

    def create_table(self):
        """Cria a tabela para gestos estáticos."""
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
            logging.info("Tabela 'gestures' criada ou verificada")
        except sqlite3.Error as e:
            logging.error(f"Erro ao criar tabela 'gestures': {e}")
            raise

    def create_dynamic_gestures_table(self):
        """Cria a tabela para gestos dinâmicos, recriando se necessário."""
        try:
            self.cursor.execute("PRAGMA table_info(dynamic_gestures)")
            columns = [col[1] for col in self.cursor.fetchall()]
            expected_columns = {'id', 'gesture_name', 'data'}
            if columns and set(columns) != expected_columns:
                logging.warning("Esquema da tabela 'dynamic_gestures' incorreto. Recriando tabela.")
                self.cursor.execute("DROP TABLE IF EXISTS dynamic_gestures")
                self.conn.commit()

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS dynamic_gestures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_name TEXT NOT NULL,
                    data BLOB NOT NULL
                )
            ''')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_dynamic_gesture_name ON dynamic_gestures (gesture_name)')
            self.conn.commit()
            logging.info("Tabela 'dynamic_gestures' criada ou verificada")
        except sqlite3.Error as e:
            logging.error(f"Erro ao criar tabela 'dynamic_gestures': {e}")
            raise

    def save_gestures(self, labels, data, gesture_types):
        """Salva gestos estáticos no banco de dados, otimizado para evitar recarregar todos os dados."""
        if not labels or not data or not gesture_types or len(labels) != len(data) or len(labels) != len(gesture_types):
            logging.error("Entradas inválidas para labels, data ou gesture_types")
            raise ValueError("Entradas inválidas para labels, data ou gesture_types")
        
        gesture_type = gesture_types[0]
        if not all(t == gesture_type for t in gesture_types):
            logging.error("Todos os gesture_types devem ser iguais")
            raise ValueError("Todos os gesture_types devem ser iguais")
        
        try:
            with self.conn:
                for label, datum in zip(labels, data):
                    # Carregar apenas os dados do gesture_name específico
                    self.cursor.execute('SELECT data FROM gestures WHERE gesture_type = ? AND gesture_name = ?', 
                                     (gesture_type, label))
                    result = self.cursor.fetchone()
                    gesture_data = pickle.loads(result[0]) if result else {label: []}
                    gesture_data[label].append(datum)
                    serialized_data = pickle.dumps(gesture_data)
                    
                    if result:
                        self.cursor.execute('UPDATE gestures SET data = ? WHERE gesture_type = ? AND gesture_name = ?', 
                                         (serialized_data, gesture_type, label))
                    else:
                        self.cursor.execute('INSERT INTO gestures (gesture_type, gesture_name, data) VALUES (?, ?, ?)', 
                                         (gesture_type, label, serialized_data))
            logging.info(f"Gestos estáticos salvos para {gesture_type}")
        except (sqlite3.Error, pickle.PickleError) as e:
            logging.error(f"Erro ao salvar gestos estáticos para {gesture_type}: {e}")
            raise

    def save_dynamic_gestures(self, labels, data):
        """Salva gestos dinâmicos no banco de dados."""
        if not labels or not data or len(labels) != len(data):
            logging.error("Entradas inválidas para labels ou data de gestos dinâmicos")
            raise ValueError("Entradas inválidas para labels ou data de gestos dinâmicos")
        
        try:
            with self.conn:
                for label, datum in zip(labels, data):
                    serialized_data = pickle.dumps(datum)
                    self.cursor.execute('INSERT INTO dynamic_gestures (gesture_name, data) VALUES (?, ?)', 
                                     (label, serialized_data))
            logging.info(f"Gestos dinâmicos salvos: {labels}")
        except (sqlite3.Error, pickle.PickleError) as e:
            logging.error(f"Erro ao salvar gestos dinâmicos: {e}")
            raise

    def load_gestures(self, gesture_type):
        """Carrega gestos estáticos do banco de dados."""
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
            logging.error(f"Erro ao carregar gestos estáticos para {gesture_type}: {e}")
            return {}, [], [], []

    def load_dynamic_gestures(self):
        """Carrega gestos dinâmicos do banco de dados."""
        try:
            self.cursor.execute('SELECT gesture_name, data FROM dynamic_gestures')
            results = self.cursor.fetchall()
            data = []
            labels = []
            for gesture_name, blob in results:
                gesture_data = pickle.loads(blob)
                labels.append(gesture_name)
                data.append(gesture_data)
            return data, labels
        except (sqlite3.Error, pickle.PickleError) as e:
            logging.error(f"Erro ao carregar gestos dinâmicos: {e}")
            return [], []

    def list_gestures(self, gesture_type):
        """Retorna uma lista de nomes de gestos estáticos para um dado gesture_type."""
        try:
            self.cursor.execute('SELECT DISTINCT gesture_name FROM gestures WHERE gesture_type = ?', (gesture_type,))
            results = self.cursor.fetchall()
            return [row[0] for row in results]
        except sqlite3.Error as e:
            logging.error(f"Erro ao listar gestos estáticos para {gesture_type}: {e}")
            return []

    def list_dynamic_gestures(self):
        """Retorna uma lista de nomes de gestos dinâmicos."""
        try:
            self.cursor.execute('SELECT DISTINCT gesture_name FROM dynamic_gestures')
            results = self.cursor.fetchall()
            return [row[0] for row in results]
        except sqlite3.Error as e:
            logging.error(f"Erro ao listar gestos dinâmicos: {e}")
            return []

    def delete_gesture(self, gesture_type, gesture_name):
        """Deleta um gesto estático específico."""
        try:
            with self.conn:
                self.cursor.execute('DELETE FROM gestures WHERE gesture_type = ? AND gesture_name = ?', 
                                  (gesture_type, gesture_name))
            if self.cursor.rowcount > 0:
                logging.info(f"Gesto estático {gesture_name} do tipo {gesture_type} deletado com sucesso")
                return True
            else:
                logging.warning(f"Nenhum gesto estático encontrado para {gesture_name} do tipo {gesture_type}")
                return False
        except sqlite3.Error as e:
            logging.error(f"Erro ao deletar gesto estático {gesture_name} do tipo {gesture_type}: {e}")
            return False

    def delete_dynamic_gesture(self, gesture_name):
        """Deleta um gesto dinâmico específico."""
        try:
            with self.conn:
                self.cursor.execute('DELETE FROM dynamic_gestures WHERE gesture_name = ?', (gesture_name,))
            if self.cursor.rowcount > 0:
                logging.info(f"Gesto dinâmico {gesture_name} deletado com sucesso")
                return True
            else:
                logging.warning(f"Nenhum gesto dinâmico encontrado para {gesture_name}")
                return False
        except sqlite3.Error as e:
            logging.error(f"Erro ao deletar gesto dinâmico {gesture_name}: {e}")
            return False

    def close(self):
        """Fecha a conexão com o banco de dados."""
        try:
            if self.conn:
                self.conn.commit()
                self.conn.close()
                logging.info("Conexão com o banco de dados fechada")
        except sqlite3.Error as e:
            logging.error(f"Erro ao fechar conexão com o banco de dados: {e}")