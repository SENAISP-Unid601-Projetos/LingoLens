@echo off
setlocal EnableDelayedExpansion
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo ========================================
echo    TRADUTOR DE LIBRAS - INICIADOR
echo ========================================

echo.
echo [1/4] Verificando ambiente virtual...

if not exist "venv" (
    echo Criando ambiente virtual...
    python -m venv venv
    echo.
    echo [2/4] Instalando dependências...
    call "venv\Scripts\activate.bat"
    python -m pip install --upgrade pip
    pip install opencv-python numpy mediapipe scikit-learn pillow
) else (
    echo Ambiente virtual encontrado.
    call "venv\Scripts\activate.bat"
    echo.
    echo [2/4] Verificando dependências...
    python -c "import cv2" 2>nul
    if errorlevel 1 (
        echo Instalando dependências...
        pip install opencv-python numpy mediapipe scikit-learn pillow
    ) else (
        echo Dependências já instaladas.
    )
)

echo.
echo [3/4] Verificando instalação...
python -c "import cv2; print('✓ OpenCV OK -', cv2.__version__)"
python -c "import numpy; print('✓ NumPy OK -', numpy.__version__)"
python -c "import mediapipe; print('✓ MediaPipe OK')"

echo.
echo [4/4] Iniciando aplicação...
cd "codigo\App"
python Main.py

cd ..\..
call "venv\Scripts\deactivate.bat"

echo.
echo Aplicação encerrada.
pause