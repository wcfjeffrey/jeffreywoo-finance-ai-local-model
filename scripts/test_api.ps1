Write-Host "Testing API..." -ForegroundColor Green
Write-Host ""

Write-Host "1. Health Check:" -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri http://localhost:5000/health -Method Get
    Write-Host "   Status: $($health.status)" -ForegroundColor Green
    Write-Host "   Device: $($health.device)" -ForegroundColor Green
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "2. Model Info:" -ForegroundColor Yellow
try {
    $info = Invoke-RestMethod -Uri http://localhost:5000/info -Method Get
    Write-Host "   Model: $($info.model)" -ForegroundColor Green
    Write-Host "   Fine-tuned: $($info.fine_tuned)" -ForegroundColor Green
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "3. Prediction:" -ForegroundColor Yellow
try {
    $body = @{
        instruction = "Analyze Tencent stock movement"
        input = "Tencent (0700.HK) trading at HKD 380, P/E ratio 25"
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $body -ContentType "application/json"
    Write-Host "   Response: $($response.response)" -ForegroundColor Green
    Write-Host "   Inference time: $($response.metadata.inference_time_ms) ms" -ForegroundColor Cyan
    Write-Host "   Device: $($response.metadata.device)" -ForegroundColor Cyan
} catch {
    Write-Host "   Error: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "✅ Testing complete!" -ForegroundColor Green