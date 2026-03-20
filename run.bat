@echo off
echo ========================================
echo   HSI Stock Prediction Project
echo ========================================
echo.
echo Select an option:
echo 1. Install dependencies
echo 2. Collect data
echo 3. Train model
echo 4. Deploy with Docker
echo 5. Test API
echo 6. Stop services
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" (
    echo Installing dependencies...
    pip install -r requirements.txt
)
if "%choice%"=="2" (
    echo Collecting data...
    python src/data/collector.py
)
if "%choice%"=="3" (
    echo Training model...
    python src/models/finetune.py
)
if "%choice%"=="4" (
    echo Deploying...
    powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1
)
if "%choice%"=="5" (
    echo Testing API...
    powershell -ExecutionPolicy Bypass -File scripts\test_api.ps1
)
if "%choice%"=="6" (
    echo Stopping services...
    powershell -ExecutionPolicy Bypass -File scripts\stop.ps1
)

pause
