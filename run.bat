@echo off
REM One-shot launcher: install deps (first run only), train if artifacts missing, start Flask.
setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python not found on PATH. Install Python 3.11 and re-open the terminal.
  exit /b 1
)

python -c "import flask, tensorflow, sklearn, pandas, vaderSentiment" 2>nul
if errorlevel 1 (
  echo Installing requirements...
  python -m pip install -r requirements.txt || exit /b 1
)

if not exist "models\lstm_model.keras" (
  echo Training models...
  python src\train.py || exit /b 1
)

echo Starting Flask on http://127.0.0.1:5000
python app.py
