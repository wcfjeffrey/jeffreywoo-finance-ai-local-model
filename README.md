<div align="center">
  <img src="assets/JeffreyWooHSIStockPredictor.png" alt="JeffreyWooHSIStockPredictorBanner" width="1200" height="600" />
</div>

## AI-Powered Stock Movement Predictor for Hang Seng Index (HSI) Companies: <p> Locally Deployed Fine-tuned Qwen2.5-7B with LoRA on RTX 5090 GPU

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda)
[![Local Deployment](https://img.shields.io/badge/deployment-local-purple.svg)]()
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📊 Overview

Not your typical financial strategic decision assistant app!

**JeffreyWoo HSI Stock Predictor** is a **locally deployed AI-powered financial strategic decision assistant app** that runs entirely on your own hardware. It fine-tunes large language model (Qwen2.5-7B) with Low-Rank Adaptation (LoRA) running on high-performance GPUs (NVIDIA RTX 5090 GPU) for Hang Seng Index (HSI) company analysis without relying on cloud APIs or external services. It is designed to help businesses and professionals make smarter, faster and more confident financial choices.

**⚠️ Note:** This repository contains **only code and documentation** (no pre-trained model weights). Model weights are excluded due to GitHub's 100 MB file size limit. You can train your own model using the provided scripts.

## ✨ What It Does
- 📊 **Real-Time Market Intelligence** — analyze complex financial data and Hang Seng Index (HSI) market trends using predictive AI models
- 🧠 **AI-Powered Strategic Guidance** — deliver actionable recommendations for investment and risk management
- 🌍 **Hong Kong Stock Market Focus** — provide specialized analysis tailored to equities and financial trends in Hong Kong stock market
- 🔒 **Secure & Scalable Deployment** — built with reproducible workflows and scalable architecture designed for privacy, reliability and enterprise use

## 🚀 Why Choose JeffreyWoo HSI Stock Predictor
Most tools just crunch numbers. **JeffreyWoo HSI Stock Predictor** goes further — embedding AI into your decision-making process so you can anticipate risks, seize opportunities, and align financial strategies with long-term goals.

## 🤖 Tech Stack
- **Language** — Python 3.10 (backend), HTML/CSS/JavaScript (frontend)
- **Framework** — Flask (backend) + Vanilla HTML/CSS/JavaScript (frontend)
- **UI** — Standard HTML5/CSS3, styled with modern CSS, enhanced with vanilla JavaScript
- **ML/AI Libraries** — PyTorch 2.7.1, Transformers, PEFT, bitsandbytes, Accelerate
- **Model Architecture** — Qwen2.5-7B with LoRA (Low-Rank Adaptation)
- **Hardware** — NVIDIA RTX 5090 GPU (24GB VRAM) with CUDA 12.8
- **Containerization** — Docker with NVIDIA Container Toolkit
- **Data Processing** — Pandas, NumPy, yFinance

## 💰 Modeling & AI Techniques Applied
This app leverages AI and machine learning (ML) methods to automate analysis of Hang Seng Index (HSI) stock movements and generate predictive insights:
- **LoRA Fine‑Tuning** — parameter‑efficient adaptation of Qwen2.5‑7B for financial text and time‑series data
- **Transformer‑Based Language Modeling** — contextual understanding of financial news, market sentiment, and structured datasets
- **Time‑Series Forecasting** — integration of historical HSI data for directional movement prediction
- **Sentiment Analysis** — parsing financial headlines and reports to detect bullish/bearish signals
- **Feature Engineering** — combining technical indicators (moving averages, candlesticks, volatility measures) with textual features
- **Scenario Simulation** — generating “what‑if” outcomes based on market conditions and model predictions
- **Confidence Scoring** — probabilistic outputs to quantify prediction reliability
- **GPU‑Accelerated Inference** — optimized deployment on NVIDIA RTX 5090 with CUDA 12.8 + PyTorch 2.7.1

## 💡 Finance Transformation Impact
- Modernizing financial workflows with AI‑driven predictive modeling and real‑time market insights
- Empowering decision‑makers through scenario simulations and confidence scoring on HSI predictions
- Strengthening risk management with GPU‑accelerated forecasting tools that run entirely on local hardware
- Driving transformation by aligning AI‑powered analytics with organizational strategy and compliance needs
- Promoting responsible innovation with secure, local deployment that keeps sensitive financial data private

## ⭐ Finance Skills Strengthened
- Designing and deploying full‑stack AI applications for finance
- Implementing secure environment management and reproducible workflows
- Integrating fine‑tuned language models (LoRA + Qwen2.5‑7B) into financial analysis pipelines
- Building data preprocessing and transformation workflows for stock market datasets
- Developing interactive dashboards and APIs with React, Flask, and Node.js for real‑time insights

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
| ✅ Model Configurations | Included | LoRA config, tokenizer and training metrics |
| ✅ Training Scripts | Included | Data collection, preprocessing and fine-tuning |
| ❌ Model Weights | **Excluded** | 154 MB - must be trained locally |
| ❌ Training Checkpoints | **Excluded** | Large checkpoint files |

## ⚖️ Disclaimer
JeffreyWooFinance provides AI-driven insights for informational purposes only. It does not replace professional financial expert advice.

## ⚙️ Run Locally

**Prerequisites:** Node.js, Python, CUDA-enabled GPU

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

#### Expected output:

HSI Stock Prediction - Fine-Tuning with Qwen2.5-7B

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

#### Windows
.\launch_webapp.ps1

#### Linux/Mac
python webapp/app.py  

Then open http://localhost:5001 in your browser.

**Note:** Run **JeffreyWoo HSI Stock Predictor** to generate insights, simulations and recommendations.

## 🐳 Docker Deployment (Optional)

cd docker  
docker-compose up -d  

The API will be available at http://localhost:5000

## 📊 Model Performance

After training, your model should achieve:

| Metric | Expected Value | 
|--------|----------------|
| **Directional Accuracy** | 65-70% |  
| **Training Loss** | ~1.3 | 
| **Inference Time** | 100-200 ms (GPU) |  
| **Model Size (LoRA)** | ~50 MB | 
| **VRAM Usage** | 12-15 GB (approximate, depends on batch size and sequence length)| 

## 📁 Project Structure

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

## 🎯 How to Get the Model

Since model weights are **excluded from this repository**, you have two options:

### Option 1: Train Your Own Model (Recommended)

Follow the quick start guide above. This ensures you have the latest data and a model tailored to your needs.

python src/data/collector.py      # Step 4  
python prepare_dataset.py          # Step 5  
python train_model_qwen.py         # Step 6

### Option 2: Download Pre-trained Weights (Coming Soon)

Pre-trained weights will be available via GitHub Releases:  

1. Go to Releases  
2. Download adapter_model.safetensors  
3. Place it in models/lora_adapters/final/

## 🏗️ Local Architecture

<pre lang="markdown">
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
  └─────────────────────────────────────────────────────────────┘</pre>

## 🎯 Sample Response in Web Interface

The app provides HSI stock movement predictions via a Flask API and web UI at http://localhost:5001.

Example output:

  <img src="assets/JeffreyWooHSIStockPredictor1.png" alt="JeffreyWooHSIStockPredictor1" width="1200" height="800" />
  <img src="assets/JeffreyWooHSIStockPredictor2.png" alt="JeffreyWooHSIStockPredictor2" width="1200" height="800" />
  <img src="assets/JeffreyWooHSIStockPredictor3.png" alt="JeffreyWooHSIStockPredictor3" width="1200" height="800" />
  <img src="assets/JeffreyWooHSIStockPredictor4.png" alt="JeffreyWooHSIStockPredictor4" width="1200" height="800" />
  
## 🙏 Why Local Deployment?

| Cloud-Based | This Local Project |
|-------------|--------------------|
| ❌ Monthly API costs | ✅ Free forever |
| ❌ Data sent to external servers | ✅ 100% private |
| ❌ Internet dependency | ✅ Works offline |
| ❌ Rate limits | ✅ Unlimited queries |
| ❌ Latency (500ms+) | ✅ Fast (100-200ms) |

## 📄 License

MIT License - Free for local deployment and modification.

## 👨‍💻 Developer

**Jeffrey Woo**  
- **GitHub:** [@wcfjeffrey](https://github.com/wcfjeffrey/)  
- **Project:** JeffreyWoo HSI Stock Predictor

## 🙏 Acknowledgments

- **Qwen** for the Qwen2.5-7B model  
- **Hugging Face** for transformers and PEFT libraries  
- **PyTorch** for CUDA 12.8 support  
- **NVIDIA** for RTX 5090 GPU architecture

**Built with ❤️ for local AI deployment | Runs entirely on your own hardware**
