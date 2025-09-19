@echo off
REM Ativar o ambiente virtual
call venv\Scripts\activate.bat

REM Instalar dependências do requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

REM Rodar o Main.py
python codigo\app\Main.py

REM Pausar para ver mensagens
pause
