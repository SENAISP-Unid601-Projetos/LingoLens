@echo off
setlocal EnableDelayedExpansion
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo ========================================
echo    TRADUTOR DE LIBRAS - Python 3.12
echo ========================================

echo.
echo [1/5] Verificando Python...
python --version
if errorlevel 1 (
    echo ERRO: Python n£o encontrado!
    pause
    exit /b 1
)

echo.
echo [2/5] Removendo ambiente antigo...
rmdir /s /q "venv" 2>nul

echo.
echo [3/5] Criando ambiente virtual...
python -m venv venv
if not exist "venv" (
    echo ERRO: Falha ao criar venv!
    pause
    exit /b 1
)

echo.
echo [4/5] Instalando dependências COMPATÍVEIS com Python 3.12...
call "venv\Scripts\activate.bat"

echo Atualizando pip...
python -m pip install --upgrade pip

echo Instalando NumPy...
pip install numpy --prefer-binary

echo Instalando OpenCV...
pip install opencv-python --prefer-binary

echo Instalando MediaPipe...
pip install mediapipe --prefer-binary

echo Instalando Scikit-learn...
pip install scikit-learn --prefer-binary

echo Instalando Pillow...
pip install pillow --prefer-binary

echo.
echo [5/5] Verificando instalação...
python -c "import numpy; print('✓ NumPy -', numpy.__version__)" && echo.
python -c "import cv2; print('✓ OpenCV -', cv2.__version__)" && echo.
python -c "import mediapipe; print('✓ MediaPipe OK')" && echo.
python -c "import sklearn; print('✓ Scikit-learn -', sklearn.__version__)" && echo.
python -c "from PIL import Image; print('✓ Pillow OK')" && echo.

echo.
echo ========================================
echo     TUDO INSTALADO COM SUCESSO!
echo ========================================

echo Iniciando aplicação...
cd "codigo\App"
python Main.py

echo.
echo Aplicação encerrada.
pause