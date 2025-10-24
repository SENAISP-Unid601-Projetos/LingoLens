@echo off
setlocal enabledelayedexpansion

echo ========================================
echo     Gesture Recognizer - Instalacao
echo ========================================
echo.

REM Definir caminhos relativos
set PROJECT_DIR=%~dp0
set CODE_DIR=%PROJECT_DIR%codigo\app
set LOGS_DIR=%PROJECT_DIR%logs
set DATA_DIR=%PROJECT_DIR%data
set REQUIREMENTS=%PROJECT_DIR%requirements.txt
set MAIN_PY=%CODE_DIR%\main.py

REM Criar diretórios necessários
if not exist "%LOGS_DIR%" (
    mkdir "%LOGS_DIR%"
    echo [INFO] Diretorio logs criado: %LOGS_DIR%
)
if not exist "%DATA_DIR%" (
    mkdir "%DATA_DIR%"
    echo [INFO] Diretorio data criado: %DATA_DIR%
)

REM Instalar dependências GLOBALMENTE no PC
echo [INFO] Instalando dependencias GLOBALMENTE no PC...
if exist "%REQUIREMENTS%" (
    pip install --user -r "%REQUIREMENTS%"
) else (
    pip install --user mediapipe opencv-python numpy scikit-learn tensorflow sqlite3
)

REM Baixar/Atualizar modelo MediaPipe (garante disponibilidade)
echo [INFO] Verificando modelo MediaPipe...
python -c "import mediapipe as mp; print('MediaPipe OK:', mp.solutions.hands.Hands())"

REM Verificar se main.py existe
if not exist "%MAIN_PY%" (
    echo [ERROR] Arquivo main.py nao encontrado em %MAIN_PY%
    echo [INFO] Verifique o caminho e tente novamente.
    pause
    exit /b 1
)

REM Executar o main.py
echo.
echo ========================================
echo     Executando GestureApp...
echo ========================================
echo.

python "%MAIN_PY%"

echo.
echo [INFO] Execucao finalizada. Pressione qualquer tecla para fechar...
pause >nul