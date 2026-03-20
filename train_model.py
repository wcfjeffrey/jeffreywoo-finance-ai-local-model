"""
Simplified Fine-Tuning for HSI Stock Prediction
This will create LoRA adapters for your model
"""
import torch
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("=" * 50)
    print("HSI Stock Prediction - Fine-Tuning")
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
        print("Run: python prepare_dataset.py first")
        return
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training samples")
    
    # Load model
    print("\nLoading DeepSeek R1 7B model...")
    model_name = "deepseek-ai/DeepSeek-R1-7B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load with 4-bit quantization to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
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
        prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
        formatted_data.append({"text": prompt})
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Split dataset
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models/lora_adapters/final",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
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
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "=" * 50)
    print("Starting Training...")
    print("=" * 50)
    print("This will take 30-60 minutes on your RTX 5090")
    print("")
    
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./models/lora_adapters/final")
    
    # Save metrics
    metrics = {
        "train_loss": trainer.state.log_history[-1].get('train_loss', 0),
        "eval_loss": trainer.state.log_history[-2].get('eval_loss', 0),
        "epochs_completed": trainer.state.epoch,
    }
    
    with open("./models/lora_adapters/final/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print("=" * 50)
    print(f"Model saved to: ./models/lora_adapters/final/")
    print(f"Training metrics: {metrics}")

if __name__ == "__main__":
    main()