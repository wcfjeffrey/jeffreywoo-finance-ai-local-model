# 🏦 JeffreyWoo HSI Stock Predictor

## Locally Deployed Fine-tuned Qwen2.5-7B with LoRA on RTX 5090 GPU

### AI-Powered Stock Movement Prediction for Hang Seng Index (HSI)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda)
[![Local Deployment](https://img.shields.io/badge/deployment-local-purple.svg)]()
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📊 Overview

**JeffreyWoo HSI Stock Predictor** is a **locally deployed AI system** that runs entirely on your own hardware. It fine-tunes large language models for Hong Kong stock market analysis without relying on cloud APIs or external services.

**⚠️ Note:** This repository contains **only code and documentation** (no pre-trained model weights). Model weights are excluded due to GitHub's 100 MB file size limit. You can train your own model using the provided scripts.

## ✨ Key Features

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

| Component | Minimum | Recommended (Your Setup) |
|-----------|---------|--------------------------|
| **GPU** | RTX 4090 (16GB) | ✅ RTX 5090 (24GB) |
| **RAM** | 32GB | ✅ 96GB |
| **Storage** | 30GB free | ✅ 6TB SSD |
| **CUDA** | 12.1 | ✅ 12.8 |

## 📦 What's Included in This Repository

| Item | Status | Description |
|------|--------|-------------|
| ✅ Python Source Code | Included | Complete training and inference pipeline |
| ✅ Web Application | Included | Flask-based UI with HTML/CSS/JS |
| ✅ Docker Configuration | Included | Dockerfile and docker-compose for containerization |
| ✅ Documentation | Included | Complete setup and usage guides |
| ✅ Model Configurations | Included | LoRA config, tokenizer, and training metrics |
| ✅ Training Scripts | Included | Data collection, preprocessing, and fine-tuning |
| ❌ Model Weights | **Excluded** | 154 MB - must be trained locally |
| ❌ Training Checkpoints | **Excluded** | Large checkpoint files |

## 🔧 Quick Start Guide

### Step 1: Clone the Repository

git clone https://github.com/wcfjeffrey/jeffreywoo-finance-ai-local-model.git  
cd jeffreywoo-finance-ai-local-model

### Step 2: Create Local Virtual Environment

python -m venv .venv  
source .venv/bin/activate      # Linux/Mac  
.venv\Scripts\activate          # Windows

### Step 3: Install Dependencies

pip install -r requirements.txt  
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

### Step 4: Collect HSI Stock Data

python src/data/collector.py  
This downloads historical price data for Hang Seng Index constituent stocks.

### Step 5: Prepare Training Dataset

python prepare_dataset.py  
This creates instruction-format training data from the collected stock prices.

### Step 6: Train the Model

python train_model_qwen.py  
This fine-tunes Qwen2.5-7B with LoRA on your local GPU. Training takes 20-30 minutes on RTX 5090.

### Expected output:

==================================================
HSI Stock Prediction - Fine-Tuning with Qwen2.5-7B
==================================================

Using device: cuda  
GPU: NVIDIA GeForce RTX 5090 Laptop GPU  
VRAM: 25.7 GB

Loading dataset...  
Loaded 40 training samples

Starting Training...  
Epoch 1/3: [====================] 100% loss: 1.85  
Epoch 2/3: [====================] 100% loss: 1.12  
Epoch 3/3: [====================] 100% loss: 0.89

✅ Training Complete!  
Model saved to: ./models/lora_adapters/final/

### Step 7: Launch the Web Application

# Windows
.\launch_webapp.ps1

# Linux/Mac
python webapp/app.py  
Then open http://localhost:5001 in your browser.

### 🐳 Docker Deployment (Optional)

cd docker  
docker-compose up -d  
The API will be available at http://localhost:5000

### 📊 Model Performance

After training, your model should achieve:

Metric	Expected Value  
Directional Accuracy	65-70%  
Training Loss	~1.3  
Inference Time	100-200 ms (GPU)  
Model Size (LoRA)	~50 MB  
VRAM Usage	12-15 GB

### 📁 Project Structure

jeffreywoo-finance-ai-local-model/  
├── 📁 webapp/                 # Local web interface  
│   ├── app.py                 # Flask backend (runs locally)  
│   ├── templates/index.html   # Frontend UI  
│   └── static/                # CSS & JavaScript  
├── 📁 src/                    # Source code  
│   ├── data/                  # Data collection scripts  
│   ├── models/                # Model training scripts  
│   └── deployment/            # API server  
├── 📁 configs/                # YAML configuration files  
├── 📁 scripts/                # Utility scripts  
├── 📁 docker/                 # Docker configuration  
├── 📁 models/                 # Trained model directory  
│   └── lora_adapters/final/   # LoRA weights (created after training)  
├── 📄 train_model_qwen.py     # Main training script  
├── 📄 prepare_dataset.py      # Dataset preparation  
├── 📄 launch_webapp.ps1       # One-click launch script  
├── 📄 requirements.txt        # Python dependencies  
└── 📄 README.md               # This documentation

### 🎯 How to Get the Model
Since model weights are excluded from this repository, you have two options:

## Option 1: Train Your Own Model (Recommended)

Follow the quick start guide above. This ensures you have the latest data and a model tailored to your needs.

python src/data/collector.py      # Step 4  
python prepare_dataset.py          # Step 5  
python train_model_qwen.py         # Step 6

## Option 2: Download Pre-trained Weights (Coming Soon)

Pre-trained weights will be available via GitHub Releases:  
Go to Releases  
Download adapter_model.safetensors  
Place it in models/lora_adapters/final/

🏗️ Local Architecture

┌─────────────────────────────────────────────────────────────┐
│                    YOUR LOCAL COMPUTER                      │
│                     (MSI Titan 18 HX)                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Web UI    │◄──►│  Flask API  │◄──►│ Fine-tuned  │      │
│  │ localhost:  │    │  localhost: │    │    Qwen     │      │
│  │    5001     │    │    5001     │    │   2.5-7B    │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         ▲                  ▲                  ▲             │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           NVIDIA RTX 5090 GPU (24GB VRAM)           │    │
│  │              CUDA 12.8 | PyTorch 2.7.1              │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Data Storage: Local SSD | Models: models/lora_adapters/    │
└─────────────────────────────────────────────────────────────┘

🎯 Sample API Response

{  
  "response": "Based on technical analysis, Tencent (0700.HK) shows bullish momentum with strong support at HKD 375. The P/E ratio of 25 is below sector average, suggesting undervaluation. Gaming revenue growth of 15% YoY indicates healthy fundamentals. Short-term price target: HKD 395-405.",  
  "metadata": {  
    "inference_time_ms": 142.3,  
    "device": "cuda",  
    "model": "Qwen2.5-7B-FineTuned",  
    "developer": "JeffreyWoo",  
    "location": "Local GPU (RTX 5090)"  
  }
}

🙏 Why Local Deployment?

Cloud-Based	This Local Project  
❌ Monthly API costs	✅ Free forever  
❌ Data sent to external servers	✅ 100% private  
❌ Internet dependency	✅ Works offline  
❌ Rate limits	✅ Unlimited queries  
❌ Latency (500ms+)	✅ Fast (100-200ms)

📄 License
MIT License - Free for local deployment and modification.

👨‍💻 Developer
Jeffrey Woo  
GitHub: @wcfjeffrey  
Project: JeffreyWoo HSI Stock Predictor

🙏 Acknowledgments
Qwen Team for the Qwen2.5-7B model  
Hugging Face for transformers and PEFT libraries  
PyTorch Team for CUDA 12.8 support  
NVIDIA for RTX 5090 GPU architecture

Built with ❤️ for local AI deployment | Runs entirely on your own hardware
