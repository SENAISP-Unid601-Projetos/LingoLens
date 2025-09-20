@echo off
REM Caminho do projeto
set PROJECT_DIR=%~dp0

REM Cria o venv se não existir
if not exist "%PROJECT_DIR%venv" (
    python -m venv "%PROJECT_DIR%venv"
)

REM Ativa o venv
call "%PROJECT_DIR%venv\Scripts\activate.bat"

REM Atualiza pip e instala dependências (só se precisar)
pip install --upgrade pip
if exist "%PROJECT_DIR%requirements.txt" (
    pip install -r "%PROJECT_DIR%requirements.txt"
)

REM Roda o Main.py usando o Python do venv
python "%PROJECT_DIR%codigo\app\Main.py"

REM Desativa o venv no final
deactivate
pause
