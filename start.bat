@echo off
setlocal enabledelayedexpansion

:: Configurações
set "SCRIPT_NAME=Main.py"
set "SCRIPT_PATH=codigo\py\%SCRIPT_NAME%"
set "SCRIPT_DIR=%~dp0"
set "FULL_SCRIPT_PATH=%SCRIPT_DIR%%SCRIPT_PATH%"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "REQUIREMENTS_FILE=requirements.txt"
set "PYTHON_VERSION=3.11.7"
set "PYTHON_INSTALLER=python-%PYTHON_VERSION%-amd64.exe"
set "PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_INSTALLER%"
set "PYTHON_INSTALL_DIR=%SCRIPT_DIR%python"

:: Cores
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

:: Verificar se o Python está instalado
echo Verificando instalacao do Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%Python nao encontrado. Tentando instalar automaticamente...%RESET%
    
    :: Verificar se o instalador já foi baixado
    if not exist "%SCRIPT_DIR%%PYTHON_INSTALLER%" (
        echo Baixando instalador do Python %PYTHON_VERSION%...
        powershell -Command "Invoke-WebRequest -Uri '%PYTHON_DOWNLOAD_URL%' -OutFile '%SCRIPT_DIR%%PYTHON_INSTALLER%'"
        if %errorlevel% neq 0 (
            echo %RED%Erro: Falha ao baixar o instalador do Python.%RESET%
            echo Acesse https://www.python.org/downloads/ e instale o Python manualmente.
            pause
            exit /b 1
        )
    )

    :: Instalar Python em modo silencioso
    echo Instalando Python %PYTHON_VERSION%...
    start /wait "" "%SCRIPT_DIR%%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 TargetDir="%PYTHON_INSTALL_DIR%"
    if %errorlevel% neq 0 (
        echo %RED%Erro: Falha ao instalar o Python.%RESET%
        echo Tente executar este script como administrador ou instale o Python manualmente.
        pause
        exit /b 1
    )

    :: Atualizar PATH para usar o Python recém-instalado
    set "PATH=%PYTHON_INSTALL_DIR%;%PYTHON_INSTALL_DIR%\Scripts;%PATH%"
    echo %GREEN%Python instalado com sucesso.%RESET%
)

:: Verificar novamente se o Python está acessível
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%Erro: Python ainda nao encontrado apos tentativa de instalacao.%RESET%
    echo Instale o Python manualmente em https://www.python.org/downloads/.
    pause
    exit /b 1
)

:: Criar requirements.txt se não existir
if not exist "%SCRIPT_DIR%%REQUIREMENTS_FILE%" (
    echo Criando arquivo %REQUIREMENTS_FILE%...
    (
        echo opencv-python
        echo mediapipe
        echo numpy
        echo scikit-learn
    ) > "%SCRIPT_DIR%%REQUIREMENTS_FILE%"
)

:: Criar e ativar ambiente virtual
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Criando ambiente virtual em %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo %RED%Erro: Falha ao criar ambiente virtual.%RESET%
        pause
        exit /b 1
    )
)

echo Ativando ambiente virtual...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo %RED%Erro: Falha ao ativar ambiente virtual.%RESET%
    pause
    exit /b 1
)

:: Atualizar pip
echo Atualizando pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo %RED%Erro: Falha ao atualizar pip.%RESET%
    pause
    exit /b 1
)

:: Instalar dependências
echo Instalando dependencias do projeto...
pip install -r "%SCRIPT_DIR%%REQUIREMENTS_FILE%"
if %errorlevel% neq 0 (
    echo %RED%Erro: Falha ao instalar dependencias.%RESET%
    echo Verifique sua conexao com a internet ou o arquivo %REQUIREMENTS_FILE%.
    pause
    exit /b 1
)

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
python "%FULL_SCRIPT_PATH%"
if %errorlevel% neq 0 (
    echo %RED%Erro: O script falhou com codigo de erro %errorlevel%.%RESET%
    echo Verifique se todos os requisitos estao instalados corretamente.
    echo Possiveis problemas:
    echo - Webcam nao detectada ou em uso por outro programa.
    echo - Permissoes insuficientes para criar arquivos (ex.: gestures.db).
    echo - Versoes incompativeis de bibliotecas.
)

:: Desativar ambiente virtual
echo Desativando ambiente virtual...
call "%VENV_DIR%\Scripts\deactivate.bat"

pause