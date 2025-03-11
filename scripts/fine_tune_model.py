#!/usr/bin/env python3
# fine_tune_model.py - Script for fine-tuning an LLM for SQL queries

import argparse
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a language model for SQL natural language queries')
    parser.add_argument('--training_data', required=True, help='Path to training data JSON file')
    parser.add_argument('--output_model', required=True, help='Path to save the ONNX model')
    parser.add_argument('--base_model', default='mistralai/Mistral-7B-Instruct-v0.2', help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()
    
    print(f"Loading training data from {args.training_data}")
    
    # Load training data
    with open(args.training_data, 'r') as f:
        data = json.load(f)
    
    # Ensure we have at least some training examples
    if len(data) < 5:
        print("Warning: Very few training examples. Adding some generic examples...")
        data.extend([
            {"Question": "What are the top export companies globally?", "Answer": json.dumps([{"name": "Example Corp", "export_volume": 1000}])},
            {"Question": "Which companies have the highest revenue?", "Answer": json.dumps([{"name": "Example Inc", "revenue": 5000}])}
        ])
    
    print(f"Loaded {len(data)} training examples")
    
    # Create dataset
    dataset = Dataset.from_dict({
        "question": [item["Question"] for item in data],
        "answer": [item["Answer"] for item in data]
    })
    
    # Format dataset for instruction fine-tuning
    def format_instruction(example):
        return {
            "text": f"""### Instruction:
Answer the following query about company export data:
{example['question']}

### Response:
{example['answer']}
"""
        }
    
    dataset = dataset.map(format_instruction)
    print(f"Prepared dataset with {len(dataset)} examples")
    
    # Try to use a smaller model if the requested one is too large
    try:
        print(f"Initializing tokenizer and model based on {args.base_model}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, load_in_8bit=True)
    except Exception as e:
        print(f"Error loading {args.base_model}: {e}")
        print("Falling back to smaller model: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Dataset tokenized successfully")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Train the model
    print("Starting training process...")
    trainer.train()
    print("Training completed successfully")
    
    # Save the model in HF format first
    hf_model_path = "./hf_model"
    trainer.save_model(hf_model_path)
    print(f"Model saved to {hf_model_path}")
    
    # Export to ONNX format
    print(f"Exporting model to ONNX format at {args.output_model}")
    try:
        # Create a directory for the ONNX model if it doesn't exist
        os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
        
        # Prepare dummy input for ONNX export
        dummy_input = tokenizer("This is a test", return_tensors="pt")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input.input_ids, dummy_input.attention_mask),
            args.output_model,
            export_params=True,
            opset_version=12,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            }
        )
        print(f"Model successfully exported to ONNX format at {args.output_model}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        print("Saving model in HF format only")
    
if __name__ == "__main__":
    main()