@echo off
cd /d "%~dp0"

echo ========================================
echo    TRADUTOR DE LIBRAS - NOVA ESTRUTURA
echo ========================================

echo Verificando ambiente virtual...
if not exist "venv" (
    echo ERRO: Ambiente virtual nao encontrado!
    echo Execute o start_install.bat primeiro!
    pause
    exit /b 1
)

echo Ativando ambiente virtual...
call "venv\Scripts\activate.bat"

echo Verificando estrutura de arquivos...
if not exist "codigo\App\Main.py" (
    echo ERRO: Arquivo Main.py nao encontrado!
    echo Verifique a estrutura do projeto.
    pause
    exit /b 1
)

if not exist "codigo\App\core" (
    echo ERRO: Pasta 'core' nao encontrada!
    pause
    exit /b 1
)

if not exist "codigo\App\ui" (
    echo ERRO: Pasta 'ui' nao encontrada!
    pause
    exit /b 1
)

echo Estrutura verificada com sucesso!
echo.
echo Iniciando aplicacao...
cd "codigo\App"
python Main.py

echo.
echo Aplicacao encerrada.
pause