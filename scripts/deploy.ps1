# deploy.ps1 - Improved with error handling
Write-Host "Deploying HSI Stock Prediction Model..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Change to docker directory
Set-Location docker

# Check if Docker is running
$dockerRunning = docker info 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    Write-Host "   Look for the Docker whale icon in your system tray." -ForegroundColor Yellow
    exit 1
}

# Check which docker compose command is available
$dockerComposeCmd = if (Get-Command "docker-compose" -ErrorAction SilentlyContinue) { 
    "docker-compose" 
} elseif (Get-Command "docker" -ErrorAction SilentlyContinue) { 
    "docker compose" 
} else { 
    $null 
}

if (-not $dockerComposeCmd) {
    Write-Host "❌ Docker Compose not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Using: $dockerComposeCmd" -ForegroundColor Yellow

# Try to build and start
try {
    Write-Host "Building and starting services..." -ForegroundColor Yellow
    Invoke-Expression "$dockerComposeCmd up -d --build"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Deployment complete!" -ForegroundColor Green
        Write-Host "API available at: http://localhost:5000" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "To test the API:" -ForegroundColor Yellow
        Write-Host "  curl http://localhost:5000/health" -ForegroundColor Gray
        Write-Host ""
        Write-Host "To view logs:" -ForegroundColor Yellow
        Write-Host "  $dockerComposeCmd logs -f" -ForegroundColor Gray
    } else {
        Write-Host "❌ Deployment failed!" -ForegroundColor Red
        Write-Host "Check logs with: $dockerComposeCmd logs" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "❌ Error: $_" -ForegroundColor Red
}

Set-Location ..