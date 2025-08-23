@echo off
setlocal enabledelayedexpansion

:: Configurações
set "SCRIPT_NAME=hand_finger_counter.py"
set "SCRIPT_PATH=codigo\py\%SCRIPT_NAME%"
set "SCRIPT_DIR=%~dp0"
set "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_PATH%"

:: Cores (mantidas do seu script original)
set "RED=[91m"
set "GREEN=[92m"
set "RESET=[0m"
reg query HKCU\Console /v VirtualTerminalLevel >nul 2>&1
if %errorlevel% neq 0 (
    set "RED="
    set "GREEN="
    set "RESET="
)

echo ===========================
echo  INICIANDO TRADUTOR DE LIBRAS
echo ===========================
echo.

:: Debugging: Mostrar caminhos para diagnóstico
echo Diretorio do script batch: %SCRIPT_DIR%
echo Caminho esperado do script Python: %FULL_SCRIPT_PATH%
echo Diretorio de trabalho atual: %CD%
echo.

:: Verificar se o arquivo principal existe
if not exist "%FULL_SCRIPT_PATH%" (
    echo %RED%Erro: Arquivo %FULL_SCRIPT_PATH% nao encontrado.%RESET%
    echo Possiveis causas:
    echo 1. O arquivo %SCRIPT_NAME% nao esta em %SCRIPT_DIR%%SCRIPT_PATH%.
    echo 2. A pasta codigo\py\ nao existe.
    echo 3. O nome do arquivo ou caminho contem erros de digitacao.
    echo.
    echo Solucoes:
    echo - Verifique se %SCRIPT_NAME% existe em %SCRIPT_DIR%%SCRIPT_PATH%.
    echo - Execute este script a partir do diretorio raiz do projeto.
    echo - Corrija o caminho em SCRIPT_PATH se necessario.
    pause
    exit /b 1
)

:: Executar o script Python
echo Executando %SCRIPT_PATH%...
python "%SCRIPT_PATH%"

if %errorlevel% neq 0 (
    echo %RED%Erro: O script falhou com código de erro %errorlevel%.%RESET%
    echo Verifique se todos os requisitos estao instalados corretamente.
)

pause