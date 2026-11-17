@echo off
echo ========================================
echo   Agricultural Market Intelligence
echo   Enhanced NLP Pipeline Startup
echo ========================================
echo.

:: Check if virtual environment exists
if exist venv\ (
    echo [OK] Virtual environment found
    call venv\Scripts\activate
) else (
    echo [!] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo [OK] Virtual environment created
)

echo.
echo [*] Installing dependencies...
pip install -q -r requirements.txt
echo [OK] Dependencies installed

echo.
echo ========================================
echo   Choose what to run:
echo ========================================
echo   1. Test Enhanced NLP Module
echo   2. Build Enhanced Dataset
echo   3. Run Streamlit Dashboard
echo   4. Train Models
echo   5. Exit
echo ========================================
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo [*] Testing Enhanced NLP Module...
    python src\nlp_enhanced.py
    pause
) else if "%choice%"=="2" (
    echo.
    echo [*] Building Enhanced Dataset...
    python src\data_ingestion_enhanced.py
    pause
) else if "%choice%"=="3" (
    echo.
    echo [*] Starting Streamlit Dashboard...
    streamlit run app.py
) else if "%choice%"=="4" (
    echo.
    echo [*] Training Models...
    python src\train.py
    pause
) else if "%choice%"=="5" (
    echo.
    echo Goodbye!
    exit
) else (
    echo.
    echo Invalid choice. Please run again.
    pause
)
