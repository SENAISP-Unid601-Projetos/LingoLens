@echo off
setlocal enabledelayedexpansion

echo ===========================
echo INICIANDO TRADUTOR DE LIBRAS
echo ===========================

:: Lista de pacotes necessários
set REQUIRED_PACKAGES=opencv-python mediapipe tensorflow scikit-learn

echo Verificando dependências...

FOR %%P IN (%REQUIRED_PACKAGES%) DO (
    pip show %%P >nul 2>&1
    IF ERRORLEVEL 1 (
        echo Instalando %%P...
        pip install %%P
    ) ELSE (
        echo %%P já instalado.
    )
)

echo ===========================
echo INICIANDO O PROJETO
echo ===========================
python hand_finger_counter.py

pause
