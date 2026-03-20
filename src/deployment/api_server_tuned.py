"""
Updated API Server with Fine-Tuned Qwen Model
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class FineTunedPredictor:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading fine-tuned model on {self.device}...")
        
        # Load base model
        base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load fine-tuned LoRA adapters
        lora_path = "./models/lora_adapters/final"
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()
        
        logger.info("✅ Fine-tuned model loaded successfully!")
    
    def predict(self, instruction, input_text='', max_tokens=200):
        start_time = time.time()
        
        # Format prompt for Qwen
        prompt = f"<|im_start|>system\nYou are a financial analyst specializing in Hong Kong stocks.<|im_end|>\n<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
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
                'inference_time_ms': inference_time,
                'device': self.device,
                'model': 'Qwen2.5-7B-FineTuned',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }

predictor = FineTunedPredictor()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'device': predictor.device})

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'model': 'Qwen2.5-7B-FineTuned',
        'fine_tuned': True,
        'device': predictor.device,
        'training_completed': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'instruction' not in data:
            return jsonify({'error': 'Missing instruction'}), 400
        
        result = predictor.predict(
            instruction=data['instruction'],
            input_text=data.get('input', ''),
            max_tokens=data.get('max_tokens', 200)
        )
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)