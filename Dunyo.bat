@echo off
REM Dunyo launcher (Windows). Uses your Python if the packages are there; otherwise sets up a local venv once.
cd /d "%~dp0"
python -c "import numpy, pandas, scipy, sklearn, requests, cryptography, reportlab" 2>nul
if %errorlevel%==0 (
  python -m dunyo app %*
  pause
  exit /b
)
if not exist venv\Scripts\python.exe (
  echo First run: creating a local Python environment...
  python -m venv venv || (echo Python 3.10-3.12 is required: https://www.python.org/downloads/ & pause & exit /b 1)
  venv\Scripts\python -m pip install --upgrade pip
  venv\Scripts\python -m pip install -r requirements.txt
)
venv\Scripts\python -m dunyo app %*
pause
