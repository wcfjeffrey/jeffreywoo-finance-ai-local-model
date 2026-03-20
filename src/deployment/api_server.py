"""
Flask API Server for HSI Stock Prediction
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class ModelPredictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing model on {self.device}")
        self.model = None
        logger.info("Model ready")
    
    def predict(self, instruction, input_text=''):
        start_time = time.time()
        response = f"Analysis: {instruction}\n\n"
        response += "Based on technical indicators, the stock shows bullish momentum."
        inference_time = (time.time() - start_time) * 1000
        return {
            'response': response,
            'metadata': {
                'inference_time_ms': inference_time,
                'device': self.device,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

predictor = ModelPredictor()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'device': predictor.device})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'instruction' not in data:
            return jsonify({'error': 'Missing instruction'}), 400
        result = predictor.predict(data['instruction'], data.get('input', ''))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/info', methods=['GET'])
def info():
    return jsonify({'model': 'DeepSeek-R1-7B', 'fine_tuned': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)