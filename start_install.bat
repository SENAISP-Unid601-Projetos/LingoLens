@echo off
cd /d "%~dp0"
echo INSTALADOR COMPLETO - Use apenas na primeira vez
echo.
rmdir /s /q venv 2>nul
python -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install opencv-python numpy mediapipe scikit-learn pillow
echo.
echo Instalação completa! Agora use start.bat para abrir rapidamente.
pause