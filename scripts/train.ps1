# train.ps1
Write-Host "Starting training..." -ForegroundColor Green
python src/models/finetune.py --config configs/training_config.yaml
