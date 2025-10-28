@echo off
cd /d "%~dp0"

echo ========================================
echo    INSTALADOR - TRADUTOR DE LIBRAS
echo ========================================

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python não encontrado!
    echo Instale o Python 3.8 ou superior e adicione ao PATH
    pause
    exit /b 1
)

echo Removendo ambiente virtual antigo...
rmdir /s /q venv 2>nul

echo Criando novo ambiente virtual...
python -m venv venv
if errorlevel 1 (
    echo ERRO: Falha ao criar ambiente virtual!
    pause
    exit /b 1
)

echo Ativando ambiente virtual...
call "venv\Scripts\activate.bat"

echo Instalando dependencias...
python -m pip install --upgrade pip
pip install opencv-python numpy mediapipe scikit-learn pillow joblib

echo Verificando instalacao...
python -c "import cv2; print('✅ OpenCV OK')"
python -c "import numpy; print('✅ NumPy OK')"
python -c "import mediapipe; print('✅ MediaPipe OK')"
python -c "import sklearn; print('✅ Scikit-learn OK')"
python -c "import PIL; print('✅ Pillow OK')"
python -c "import joblib; print('✅ Joblib OK')"

echo.
echo Criando estrutura de pastas...
mkdir "codigo\App\data" 2>nul
mkdir "codigo\App\Logs" 2>nul

echo.
echo Instalacao completa! Agora use start.bat
pause