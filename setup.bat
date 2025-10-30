@echo off
REM Setup script for PDF RAG Evaluation System on Windows

echo ========================================
echo PDF RAG Evaluation System Setup
echo ========================================
echo.

REM Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    echo Please make sure you're in the correct directory.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
echo.

echo [2/3] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Could not upgrade pip, continuing anyway...
)
echo.

echo [3/3] Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    echo Please check the error messages above
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo To run the application, execute:
echo     streamlit run streamlit_app.py
echo.
pause
