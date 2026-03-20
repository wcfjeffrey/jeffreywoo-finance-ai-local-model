# stop.ps1
Write-Host "Stopping services..." -ForegroundColor Yellow
Set-Location docker
docker-compose down
Write-Host "??Services stopped" -ForegroundColor Green
