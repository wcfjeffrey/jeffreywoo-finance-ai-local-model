## JeffreyWoo HSI Stock Predictor (with Locally Deployed Fine-tuned Qwen2.5-7B with LoRA on RTX 5090 GPU)

## Fine-tuned Large Language Model (LLM) for Hang Seng Index (HSI) Stock Movement Prediction
(featured web interface, Docker containerization, and complete local deployment on personal hardware)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda)
[![Local Deployment](https://img.shields.io/badge/deployment-local-purple.svg)]()
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📊 Overview

**JeffreyWoo HSI Stock Predictor** is a **locally deployed AI system** that runs entirely on your own hardware. It fine-tunes large language models for Hong Kong stock market analysis without relying on cloud APIs or external services.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🏠 Local Deployment** | Runs 100% on your own computer - no cloud costs, no API calls |
| **⚡ RTX 5090 GPU** | Optimized for local GPU acceleration with CUDA 12.8 |
| **🔧 LoRA Fine-tuning** | Parameter-efficient training (0.5% parameters) on your local machine |
| **🐳 Docker Container** | One-command local deployment with containerization |
| **🌐 Web Interface** | Beautiful local web UI at http://localhost:5001 |
| **📊 Real-time Predictions** | Local inference with ~100-200ms response time |
| **🔒 Privacy First** | All data stays on your computer - no external servers |

## 🚀 Local Deployment

This project is designed to run **completely locally** on your own hardware. No cloud services, no API keys, no recurring costs.

### Hardware Requirements

| Component   | Minimum         | Recommended (Your Setup) |
|-------------|-----------------|--------------------------|
| **GPU**     | RTX 4090 (16GB) | ✅ RTX 5090 (24GB)       |
| **RAM**     | 32GB            | ✅ 96GB                  |
| **Storage** | 30GB free       | ✅ 6TB SSD               |
| **CUDA**    | 12.1            | ✅ 12.8                  |

## Quick Local Start

### 1. Clone the repository
git clone https://github.com/wcfjeffrey/jeffreywoo-finance-ai-local-model.git  
cd jeffreywoo-finance-ai-local-model

### 2. Create local virtual environment
python -m venv .venv  
source .venv/bin/activate  # Linux/Mac  
.venv\Scripts\activate     # Windows

### 3. Install dependencies locally
pip install -r requirements.txt  
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

### 4. Collect Data

python src/data/collector.py

### 5. Train Model

python src/models/finetune.py --config configs/training_config.yaml

### 6. Deploy with Docker

.\scripts\deploy.ps1

### 7. Launch the local web app
./launch_webapp.ps1  # Windows
# or
python webapp/app.py  # Cross-platform

## 8. Model Performance

| Metric | Value |
|--------|-------|
| Directional Accuracy | 67.3% |
| Inference Time | 85ms |
| Model Size (LoRA) | 48MB |

## 9. Project Structure

hsi-stock-prediction-deepseek/
├── 📁 webapp/                 # Local web interface
│   ├── app.py                 # Flask backend (runs locally)
│   └── static/                # CSS & JS (served locally)
├── 📁 src/                    # Local source code
│   ├── data/                  # Local data collection
│   ├── models/                # Local model training
│   └── deployment/            # Local API server
├── 📁 configs/       # Configuration files
├── 📁 scripts/       # Utility scripts
├── 📁 data/          # Data directory
├── 📁 models/        # Trained models
│   └── lora_adapters/final/   # LoRA weights (50MB)
├── 📁 docker/        # Docker deployment
├── 📄 launch_webapp.ps1       # One-click local launch
├── 📄 README.md               # This documentation
└── 📄 logs/          # Log files

## 10. Documentation

- [Training Guide](docs/training_guide.md)
- [Deployment Guide](docs/deployment_guide.md)
- [API Reference](docs/api_reference.md)
