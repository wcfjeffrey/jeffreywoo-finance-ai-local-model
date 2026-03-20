# launch_webapp.ps1 - JeffreyWoo HSI Stock Predictor
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  JeffreyWoo HSI Stock Predictor" -ForegroundColor Green
Write-Host "  AI-Powered Stock Analysis" -ForegroundColor Green
Write-Host "  Developed by JeffreyWoo" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cd C:\Projects\hsi-stock-prediction-deepseek

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.venv\Scripts\Activate

# Install web dependencies if needed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
pip install flask flask-cors 2>$null

Write-Host ""
Write-Host "🚀 Starting JeffreyWoo HSI Stock Predictor..." -ForegroundColor Green
Write-Host "   Backend API: http://localhost:5001" -ForegroundColor Cyan
Write-Host "   Open in browser: http://localhost:5001" -ForegroundColor Cyan
Write-Host "   Developer: JeffreyWoo" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

# Start the web app
python webapp/app.py