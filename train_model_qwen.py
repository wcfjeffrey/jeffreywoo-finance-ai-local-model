"""
Fine-Tuning with Qwen2.5-7B (Fixed 4-bit Loading)
"""
import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 50)
    print("HSI Stock Prediction - Fine-Tuning with Qwen2.5-7B")
    print("=" * 50)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_path = "data/processed/training_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Creating sample dataset for testing...")
        
        # Create sample dataset
        sample_data = [
            {
                "instruction": "Analyze Tencent stock movement",
                "input": "Tencent (0700.HK) at HKD 380, RSI 65",
                "output": "Bullish momentum detected. RSI indicates strong buying pressure."
            },
            {
                "instruction": "Analyze Alibaba stock movement",
                "input": "Alibaba (9988.HK) at HKD 75, MACD bullish",
                "output": "Technical indicators show bullish signal."
            },
            {
                "instruction": "Analyze HSBC stock movement",
                "input": "HSBC (0005.HK) at HKD 60, high volume",
                "output": "Positive momentum with strong volume support."
            }
        ]
        
        os.makedirs("data/processed", exist_ok=True)
        with open(dataset_path, "w") as f:
            json.dump(sample_data, f, indent=2)
        print(f"Created sample dataset with {len(sample_data)} samples")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training samples")
    
    # Load model with correct 4-bit configuration
    print("\nLoading Qwen2.5-7B model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,  # Use quantization_config instead of load_in_4bit
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Prepare dataset
    print("\nPreparing dataset...")
    formatted_data = []
    for item in data:
        # Format for Qwen instruction format
        prompt = f"<|im_start|>system\nYou are a financial analyst specializing in Hong Kong stocks.<|im_end|>\n<|im_start|>user\n{item['instruction']}\n\n{item['input']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted_data.append({"text": prompt})
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Split dataset
    if len(tokenized_dataset) > 1:
        split_dataset = tokenized_dataset.train_test_split(test_size=min(0.1, 1/len(tokenized_dataset)), seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = tokenized_dataset
        eval_dataset = tokenized_dataset
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments (reduced for testing)
    training_args = TrainingArguments(
        output_dir="./models/lora_adapters/final",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if len(eval_dataset) > 0 else "no",
        load_best_model_at_end=True if len(eval_dataset) > 0 else False,
        save_total_limit=2,
        report_to="none",  # Disable wandb reporting
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)
    print("This will take 20-40 minutes on your RTX 5090")
    print("")
    
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./models/lora_adapters/final")
    
    # Save metrics
    metrics = {
        "train_loss": trainer.state.log_history[-1].get('train_loss', 0) if trainer.state.log_history else 0,
        "epochs_completed": trainer.state.epoch,
        "model": "Qwen2.5-7B-Instruct",
        "train_samples": len(train_dataset)
    }
    
    os.makedirs("./models/lora_adapters/final", exist_ok=True)
    with open("./models/lora_adapters/final/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print("=" * 50)
    print(f"Model saved to: ./models/lora_adapters/final/")
    print(f"Training metrics: {metrics}")
    
    # Test the model with a sample
    print("\nTesting the trained model...")
    model.eval()
    test_prompt = "<|im_start|>user\nAnalyze Tencent stock movement\n\nTencent at HKD 380<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test response: {response}")

if __name__ == "__main__":
    main()