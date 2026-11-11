@echo off
REM Crea el entorno virtual del proyecto e instala las dependencias.
REM Uso: doble clic en set_up.bat o ejecutarlo desde la carpeta del repositorio.

cd /d "%~dp0"

if not exist venv (
    echo Creando entorno virtual en la carpeta venv...
    python -m venv venv || goto :error
)

call venv\Scripts\activate.bat || goto :error
python -m pip install --upgrade pip || goto :error
pip install -r requirements.txt || goto :error

echo.
echo Entorno listo. Para activarlo en una nueva terminal ejecute: venv\Scripts\activate
echo Siguiente paso: python mlops_pipeline\model_training_evaluation.py
goto :eof

:error
echo.
echo Error durante la instalacion. Revise el mensaje anterior.
exit /b 1
