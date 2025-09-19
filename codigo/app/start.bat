@echo off
REM Cria o venv se não existir
if not exist venv (
    python -m venv venv
)

REM Ativa o venv
call venv\Scripts\activate.bat

REM Instala dependências (opcional, só se quiser garantir)
pip install --upgrade pip
pip install -r requirements.txt

REM Roda o Main.py usando o Python do venv
python codigo\app\Main.py

REM Desativa o venv no final
deactivate
