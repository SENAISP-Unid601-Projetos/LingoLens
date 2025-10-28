import os
import sqlite3
import json
import logging
from Config import CONFIG
import numpy as np

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
        logging.info("✅ Tabelas do banco verificadas/criadas")

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
            logging.debug(f"✅ Gesto '{name}' salvo no banco (tipo: {g_type})")
            return True
        except Exception as e:
            logging.error(f"❌ Erro ao adicionar gesto: {e}")
            return False

    def save_gestures(self, labels, data, types=None):
        """Salva gestos de um tipo específico SEM apagar outros tipos"""
        if types is None:
            types = ["letter"] * len(labels)

        if not labels or not data:
            logging.warning("⚠️ Nenhum dado para salvar em save_gestures")
            return False

        try:
            # 🔥 CORREÇÃO CRÍTICA: Apagar apenas gestos do tipo específico
            # Encontrar tipos únicos que estamos salvando
            unique_types = list(set(types))
            
            for g_type in unique_types:
                # Apagar apenas gestos do tipo atual
                self.conn.execute("DELETE FROM gestures WHERE type=?", (g_type,))
                logging.info(f"🗑️ Gestos do tipo '{g_type}' removidos para substituição")
            
            # Inserir novos gestos
            for name, landmarks, g_type in zip(labels, data, types):
                # Converta para lista se for ndarray
                if isinstance(landmarks, np.ndarray):
                    landmarks = landmarks.tolist()
                elif hasattr(landmarks, 'tolist'):  # Para outros tipos com tolist()
                    landmarks = landmarks.tolist()

                landmarks_json = json.dumps(landmarks)
                self.conn.execute(
                    "INSERT INTO gestures (name, type, landmarks) VALUES (?, ?, ?)",
                    (name, g_type, landmarks_json),
                )

            # Atualizar gesture_names - remover apenas os tipos que estamos salvando
            for g_type in unique_types:
                self.conn.execute("DELETE FROM gesture_names WHERE type=?", (g_type,))
            
            for name, g_type in zip(labels, types):
                self.conn.execute(
                    "INSERT OR IGNORE INTO gesture_names (name, type) VALUES (?, ?)",
                    (name, g_type),
                )

            self.conn.commit()
            logging.info(f"💾 {len(labels)} gestos salvos no banco (tipos: {unique_types})")
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro ao salvar gestos: {e}")
            return False

    def save_movements(self, labels, data):
        """Salva movimentos SEM apagar gestos de outros tipos"""
        return self.save_gestures(labels, data, types=["movement"] * len(labels))

    def load_gestures(self, gesture_type=None):
        """Carrega gestos do banco"""
        labels, data, gesture_names = [], [], {}

        try:
            # Carregar nomes
            query_names = "SELECT name, type FROM gesture_names"
            if gesture_type:
                query_names += " WHERE type=?"
                cursor = self.conn.execute(query_names, (gesture_type,))
            else:
                cursor = self.conn.execute(query_names)
                
            gesture_names = {name: name for name, _ in cursor.fetchall()}

            # Carregar gestos
            query_gestures = "SELECT name, type, landmarks FROM gestures"
            if gesture_type:
                query_gestures += " WHERE type=?"
                cursor = self.conn.execute(query_gestures, (gesture_type,))
            else:
                cursor = self.conn.execute(query_gestures)

            for name, g_type, landmarks_json in cursor.fetchall():
                try:
                    landmarks = json.loads(landmarks_json)
                    if isinstance(landmarks, list):
                        labels.append(name)
                        data.append(landmarks)
                except json.JSONDecodeError:
                    logging.error(f"❌ Erro ao decodificar gesto {name}")

            logging.info(f"📂 {len(labels)} gestos carregados do banco (tipo: {gesture_type})")
            return labels, data, gesture_names
            
        except Exception as e:
            logging.error(f"❌ Erro ao carregar gestos: {e}")
            return [], [], {}

    def load_movements(self):
        """Carrega movimentos do banco"""
        return self.load_gestures(gesture_type="movement")

    def delete_gesture(self, gesture_name):
        """Deleta um gesto específico"""
        try:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM gestures WHERE name=?", (gesture_name,)
            )
            if cursor.fetchone()[0] == 0:
                return False

            self.conn.execute("DELETE FROM gestures WHERE name=?", (gesture_name,))
            
            # Verificar se ainda existe em outros tipos
            cursor = self.conn.execute("SELECT COUNT(*) FROM gestures WHERE name=?", (gesture_name,))
            if cursor.fetchone()[0] == 0:
                self.conn.execute("DELETE FROM gesture_names WHERE name=?", (gesture_name,))
                
            self.conn.commit()
            logging.info(f"🗑️ Gesto '{gesture_name}' deletado do banco")
            return True
        except Exception as e:
            logging.error(f"❌ Erro ao deletar gesto '{gesture_name}': {e}")
            return False

    def delete_movement(self, movement_name):
        """Deleta um movimento específico"""
        try:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM gestures WHERE name=? AND type='movement'", 
                (movement_name,)
            )
            if cursor.fetchone()[0] == 0:
                logging.warning(f"Movimento '{movement_name}' não encontrado")
                return False

            self.conn.execute("DELETE FROM gestures WHERE name=? AND type='movement'", (movement_name,))
            
            # Verificar se ainda existe em outros tipos
            cursor = self.conn.execute("SELECT COUNT(*) FROM gestures WHERE name=?", (movement_name,))
            if cursor.fetchone()[0] == 0:
                self.conn.execute("DELETE FROM gesture_names WHERE name=?", (movement_name,))
                
            self.conn.commit()
            logging.info(f"🗑️ Movimento '{movement_name}' deletado do banco")
            return True
        except Exception as e:
            logging.error(f"❌ Erro ao deletar movimento '{movement_name}': {e}")
            return False

    def get_all_gestures_count(self):
        """Retorna contagem de todos os gestos no banco (para debug)"""
        try:
            cursor = self.conn.execute("SELECT type, COUNT(*) FROM gestures GROUP BY type")
            result = cursor.fetchall()
            logging.info(f"📊 Estatísticas do banco: {dict(result)}")
            return dict(result)
        except Exception as e:
            logging.error(f"❌ Erro ao contar gestos: {e}")
            return {}

    def close(self):
        """Fecha conexão com o banco"""
        try:
            self.conn.close()
            logging.info("🔒 Conexão com banco fechada")
        except Exception as e:
            logging.error(f"❌ Erro ao fechar banco: {e}")