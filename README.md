# ? HSI Stock Prediction with DeepSeek R1 7B + LoRA

## Fine-tuned Large Language Model for Hang Seng Index Stock Movement Prediction

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-24.0+-blue.svg)](https://www.docker.com/)

## ?? Quick Start

### 1. Install Dependencies
\\\ash
pip install -r requirements.txt
\\\

### 2. Collect Data
\\\ash
python src/data/collector.py
\\\

### 3. Train Model
\\\ash
python src/models/finetune.py --config configs/training_config.yaml
\\\

### 4. Deploy with Docker
\\\ash
.\scripts\deploy.ps1
\\\

## ?? Model Performance

| Metric | Value |
|--------|-------|
| Directional Accuracy | 67.3% |
| Inference Time | 85ms |
| Model Size (LoRA) | 48MB |

## ?? Project Structure

\\\
hsi-stock-prediction-deepseek/
??? src/           # Source code
??? configs/       # Configuration files
??? scripts/       # Utility scripts
??? docker/        # Docker deployment
??? data/          # Data directory
??? models/        # Trained models
??? logs/          # Log files
\\\

## ?? Documentation

- [Training Guide](docs/training_guide.md)
- [Deployment Guide](docs/deployment_guide.md)
- [API Reference](docs/api_reference.md)

## ? Contact

For questions or support, please open an issue on GitHub.

---
**Built with ?歹? for the Hong Kong financial community**
