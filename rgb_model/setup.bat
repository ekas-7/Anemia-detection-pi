@echo off
REM Setup Script for RGB Model
REM Windows Batch Script

echo ============================================================
echo RGB Model Setup - Anemia Detection
echo ============================================================

echo.
echo [Step 1/3] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [Step 3/3] Verifying Kaggle API setup...
if not exist "%USERPROFILE%\.kaggle\kaggle.json" (
    echo Warning: Kaggle API credentials not found
    echo.
    echo To set up Kaggle API:
    echo 1. Go to https://www.kaggle.com/
    echo 2. Sign in and go to Account settings
    echo 3. Click "Create New API Token"
    echo 4. Save kaggle.json to: %USERPROFILE%\.kaggle\
    echo.
) else (
    echo Kaggle API credentials found!
)

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Configure Kaggle API if not already done
echo 2. Run: cd data
echo 3. Run: python load_dataset.py
echo 4. Follow the Quick Start guide in README.md
echo.
pause
