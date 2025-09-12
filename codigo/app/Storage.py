import sqlite3
import json



def salvar_gesto(nome, landmarks, db="gestures.db"):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gestures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            landmarks TEXT
        )
    ''')
    cursor.execute("INSERT INTO gestures (name, landmarks) VALUES (?, ?)",
                   (nome, json.dumps(landmarks)))
    conn.commit()
    conn.close()

def carregar_dados(db="gestures.db"):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT name, landmarks FROM gestures")
    dados = cursor.fetchall()
    conn.close()
    return dados
