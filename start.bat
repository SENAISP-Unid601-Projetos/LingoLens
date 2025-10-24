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

echo.
echo Selecione o modo de execucao:
echo 1 - Sistema Original (Gestos + Movimentos)
echo 2 - Tradutor de Libras (Recomendado)
echo.

choice /c 12 /n /m "Digite sua escolha (1 ou 2): "

if errorlevel 2 (
    echo Iniciando Tradutor de Libras...
    cd "codigo\App"
    if exist "MainLibras.py" (
        python MainLibras.py
    ) else (
        echo ERRO: MainLibras.py nao encontrado!
        echo Executando sistema original...
        python Main.py
    )
) else (
    echo Iniciando Sistema Original...
    cd "codigo\App"
    python Main.py
)

echo.
echo Aplicacao encerrada.
pause