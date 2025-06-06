@echo off
setlocal enabledelayedexpansion

:: Configurações
set "SCRIPT_PATH=codigo\py\hand_finger_counter.py"
set "REQUIRED_PACKAGES=opencv-python mediapipe tensorflow scikit-learn"

:: Cores (para versões de Windows que suportam)
set "RED="
set "GREEN="
set "RESET="
if defined ANSI_CODE (
    set "RED=[91m"
    set "GREEN=[92m"
    set "RESET=[0m"
)

echo ===========================
echo  INICIANDO TRADUTOR DE LIBRAS
echo ===========================
echo.

echo Verificando dependências...
echo.

:: Verificar se o Python está instalado
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%Erro: Python não está instalado ou não está no PATH.%RESET%
    echo Por favor, instale Python antes de executar este script.
    pause
    exit /b 1
)

:: Verificar e instalar pacotes necessários
for %%P in (%REQUIRED_PACKAGES%) do (
    pip show %%P >nul 2>&1
    if !errorlevel! equ 0 (
        echo %GREEN%%%P já está instalado.%RESET%
    ) else (
        echo Instalando %%P...
        pip install %%P --quiet
        if !errorlevel! neq 0 (
            echo %RED%Falha ao instalar %%P%RESET%
            set "INSTALL_ERROR=1"
        )
    )
)

if defined INSTALL_ERROR (
    echo %RED%Erro: Falha ao instalar algumas dependências.%RESET%
    pause
    exit /b 1
)

echo.
echo ===========================
echo  INICIANDO O PROJETO
echo ===========================
echo.

:: Verificar se o arquivo principal existe
if not exist "%SCRIPT_PATH%" (
    echo %RED%Erro: Arquivo %SCRIPT_PATH% não encontrado.%RESET%
    echo Certifique-se de que a estrutura de pastas está correta.
    pause
    exit /b 1
)

:: Executar o script Python
echo Executando %SCRIPT_PATH%...
python "%SCRIPT_PATH%"

if %errorlevel% neq 0 (
    echo %RED%Erro: O script falhou com código de erro %errorlevel%.%RESET%
    echo Verifique se todos os requisitos estão instalados corretamente.
)

pause