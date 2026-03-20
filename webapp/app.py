"""
JeffreyWoo HSI Stock Predictor
Fine-tuned Qwen2.5-7B Model | Developed by JeffreyWoo
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import time
import json
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Model configuration
class StockPredictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading fine-tuned model on {self.device}...")
        
        try:
            # Load base model
            base_model_name = "Qwen/Qwen2.5-7B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load fine-tuned LoRA adapters
            lora_path = "./models/lora_adapters/final"
            if os.path.exists(lora_path):
                self.model = PeftModel.from_pretrained(self.base_model, lora_path)
                logger.info("✅ Fine-tuned model loaded successfully!")
                self.is_fine_tuned = True
            else:
                self.model = self.base_model
                logger.info("⚠️ Using base model (no fine-tuned adapters found)")
                self.is_fine_tuned = False
            
            self.model.eval()
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
            self.is_fine_tuned = False
    
    def predict(self, instruction, input_text='', max_tokens=300, temperature=0.7):
        if not self.model_loaded:
            return {
                'response': "Model not loaded. Please check the server logs.",
                'error': True
            }
        
        start_time = time.time()
        
        # Format prompt for Qwen
        prompt = f"<|im_start|>system\nYou are a professional financial analyst specializing in Hong Kong stocks and the Hang Seng Index. Provide detailed, data-driven analysis.<|im_end|>\n<|im_start|>user\n{instruction}\n\nStock Data: {input_text}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        
        inference_time = (time.time() - start_time) * 1000
        
        return {
            'response': response,
            'metadata': {
                'inference_time_ms': round(inference_time, 2),
                'device': self.device,
                'model': 'Qwen2.5-7B-FineTuned' if self.is_fine_tuned else 'Qwen2.5-7B-Base',
                'fine_tuned': self.is_fine_tuned,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'error': False
        }

# Initialize predictor
predictor = StockPredictor()

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': predictor.device,
        'model_loaded': predictor.model_loaded,
        'fine_tuned': predictor.is_fine_tuned
    })

@app.route('/info', methods=['GET'])
def info():
    """Model information"""
    return jsonify({
        'model': 'Qwen2.5-7B-FineTuned' if predictor.is_fine_tuned else 'Qwen2.5-7B-Base',
        'fine_tuned': predictor.is_fine_tuned,
        'device': predictor.device,
        'status': 'ready' if predictor.model_loaded else 'error'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        instruction = data.get('instruction', 'Analyze stock movement')
        input_text = data.get('input', '')
        max_tokens = min(data.get('max_tokens', 300), 500)
        temperature = min(max(data.get('temperature', 0.7), 0.1), 1.5)
        
        result = predictor.predict(
            instruction=instruction,
            input_text=input_text,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if result.get('error'):
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/examples', methods=['GET'])
def examples():
    """Get example prompts"""
    examples = [
        {
            "instruction": "Analyze Tencent stock movement",
            "input": "Tencent (0700.HK) at HKD 380, P/E 25, gaming revenue +15% YoY",
            "description": "Tencent Stock Analysis"
        },
        {
            "instruction": "Predict Hang Seng Index movement",
            "input": "HSI at 16,500, US futures +0.5%, China stimulus expected",
            "description": "HSI Index Prediction"
        },
        {
            "instruction": "Compare Alibaba and JD.com",
            "input": "Alibaba (9988.HK) HKD 75, JD.com (9618.HK) HKD 105, e-commerce sector growth 12%",
            "description": "E-commerce Comparison"
        },
        {
            "instruction": "Analyze HKEX financials",
            "input": "HKEX (0388.HK) at HKD 280, trading volume +20%, IPO pipeline strong",
            "description": "HKEX Analysis"
        }
    ]
    return jsonify(examples)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)