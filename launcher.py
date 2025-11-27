# launcher.py — VERSÃO QUE FUNCIONA 100% COMO .EXE
import os
import sys
import pathlib

# Descobre onde o programa está rodando (dentro do .exe ou não)
if getattr(sys, 'frozen', False):
    # Rodando como .exe
    BASE_DIR = os.path.dirname(sys.executable)
    # PyInstaller extrai os arquivos em sys._MEIPASS
    RESOURCE_DIR = sys._MEIPASS
else:
    # Rodando com python normal
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESOURCE_DIR = BASE_DIR

# Força o Python a achar as pastas
sys.path.insert(0, os.path.join(RESOURCE_DIR, "src"))
sys.path.insert(0, os.path.join(RESOURCE_DIR, "config"))

# Muda o diretório atual pra pasta do .exe (importante pro banco)
os.chdir(BASE_DIR)

# Agora importa e roda o app
from src.core.GestureApp import GestureApp

if __name__ == "__main__":
    app = GestureApp()
    app.run()