@echo off
setlocal enabledelayedexpansion

echo ========================================
echo     Gesture Recognizer - Instalacao
echo ========================================
echo.

set "PROJECT_DIR=%~dp0"
set "CODE_DIR=%PROJECT_DIR%codigo\app"
set "LOGS_DIR=%CODE_DIR%\logs"
set "DATA_DIR=%CODE_DIR%\data"
set "MODELS_DIR=%CODE_DIR%\models"
set "REQUIREMENTS=%PROJECT_DIR%requirements.txt"
set "MAIN_PY=%CODE_DIR%\Main.py"

for %%D in ("%LOGS_DIR%" "%DATA_DIR%" "%MODELS_DIR%") do (
    if not exist "%%D" (
        mkdir "%%D"
        echo [INFO] Diretorio criado: %%D
    )
)

echo [INFO] Instalando dependencias...
if exist "%REQUIREMENTS%" (
    pip install --user -r "%REQUIREMENTS%"
) else (
    pip install --user mediapipe opencv-python numpy scikit-learn tensorflow
)

echo [INFO] Testando MediaPipe...
python -c "import mediapipe as mp; print('MediaPipe OK:', mp.__version__)" >nul 2>&1
if errorlevel 1 (
    echo [ERRO] MediaPipe falhou.
    goto :error
)

if not exist "%MAIN_PY%" (
    echo [ERRO] Main.py nao encontrado: %MAIN_PY%
    goto :error
)

echo.
echo ========================================
echo     Executando GestureApp...
echo ========================================
echo.

python "%MAIN_PY%"

echo.
echo [INFO] Finalizado.
pause >nul
exit /b 0

:error
echo [ERRO] Falha na instalacao.
pause >nul
exit /b 1