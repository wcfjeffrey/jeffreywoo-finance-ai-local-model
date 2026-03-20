Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up RTX 5090 with PyTorch 2.7.1+cu128" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

cd C:\Projects\hsi-stock-prediction-deepseek

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.venv\Scripts\Activate

# Create test script
@"
import torch

print("=" * 50)
print("PyTorch GPU Test")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test GPU operation
    print("\nTesting GPU tensor operation...")
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    print("✅ GPU operation successful!")
else:
    print("\n❌ CUDA not available")
"@ | Out-File -FilePath test_gpu.py -Encoding UTF8 -NoNewline

# Check current PyTorch
Write-Host "Checking current PyTorch version..." -ForegroundColor Yellow
python -c "import torch; print('Current PyTorch:', torch.__version__)" 2>$null

Write-Host ""
Write-Host "To install PyTorch 2.7.1 with CUDA 12.8, run:" -ForegroundColor Yellow
Write-Host "pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128" -ForegroundColor White
Write-Host ""

# Run test
Write-Host "Running GPU test..." -ForegroundColor Yellow
python test_gpu.py