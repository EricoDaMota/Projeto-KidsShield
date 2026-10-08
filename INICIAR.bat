@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ==============================================
echo   ROSTO SEGURO - INICIALIZACAO AUTOMATICA
echo ==============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Criando ambiente virtual...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERRO: Nao foi possivel criar o ambiente virtual.
        echo Instale o Python 3 e tente novamente.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo ERRO: Nao foi possivel ativar o ambiente virtual.
    pause
    exit /b 1
)

echo [2/4] Atualizando o instalador de pacotes...
python -m pip install --upgrade pip >nul

echo [3/4] Verificando dependencias...
pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERRO: Nao foi possivel instalar as dependencias.
    pause
    exit /b 1
)

echo [4/4] Iniciando o aplicativo...
echo Quando o navegador abrir, use http://localhost:8501 se necessario.
echo Para encerrar, feche esta janela ou pressione Ctrl + C.
echo.
start "" powershell -WindowStyle Hidden -Command "Start-Sleep -Seconds 4; Start-Process 'http://localhost:8501'"
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false

endlocal
