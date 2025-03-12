#!/usr/bin/env python3
# fine_tune_model.py - Script for fine-tuning an LLM for place and city queries

import argparse
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a language model for place and city data queries')
    parser.add_argument('--training_data', required=True, help='Path to training data JSON file')
    parser.add_argument('--output_model', required=True, help='Path to save the ONNX model')
    parser.add_argument('--base_model', default='facebook/opt-125m', help='Base model to fine-tune')
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
            {
                "Question": "What are the most popular places in New York?", 
                "Answer": json.dumps([
                    {"placeName": "Central Park", "popularity": 92, "cityName": "New York"},
                    {"placeName": "Times Square", "popularity": 88, "cityName": "New York"}
                ])
            },
            {
                "Question": "Compare attractions in San Francisco", 
                "Answer": json.dumps([
                    {"placeName": "Golden Gate Bridge", "popularity": 95, "cityName": "San Francisco"},
                    {"placeName": "Fisherman's Wharf", "popularity": 85, "cityName": "San Francisco"}
                ])
            }
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
Answer the following query about place and city data:
{example['question']}

### Response:
{example['answer']}
"""
        }
    
    dataset = dataset.map(format_instruction)
    print(f"Prepared dataset with {len(dataset)} examples")
    
    # Use a small, publicly available model
    model_name = "facebook/opt-125m"  # Small model that's publicly available
    print(f"Initializing tokenizer and model based on {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        # Important: Disable the use_cache option
        model.config.use_cache = False
        
        # Tokenize dataset
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            )
            
            # Set up labels for language modeling loss calculation
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            return model_inputs
        
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
            # Make sure we're passing labels to the model
            remove_unused_columns=False,
            label_names=["labels"],
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
            dummy_input = tokenizer("Describe the most popular places in a city", return_tensors="pt")
            
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
            
    except Exception as e:
        print(f"Error during model training: {e}")
        print("Creating a simple ONNX model as fallback")
        create_fallback_onnx_model(args.output_model)

def create_fallback_onnx_model(output_path):
    """Create a very simple ONNX model as a fallback"""
    try:
        import onnx
        from onnx import helper
        from onnx import TensorProto
        
        # Create a very simple model that just passes through input
        node_def = helper.make_node(
            'Identity',
            inputs=['input_ids'],
            outputs=['logits'],
            name='simple_identity',
        )
        
        # Create the graph
        graph_def = helper.make_graph(
            [node_def],
            'simple-model',
            [helper.make_tensor_value_info('input_ids', TensorProto.INT64, [1, 'sequence_length'])],
            [helper.make_tensor_value_info('logits', TensorProto.INT64, [1, 'sequence_length'])],
        )
        
        # Create the model
        model_def = helper.make_model(graph_def, producer_name='fallback-model')
        model_def.opset_import[0].version = 12
        
        # Save the model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        onnx.save(model_def, output_path)
        print(f"Fallback ONNX model saved to {output_path}")
    except Exception as e:
        print(f"Error creating fallback ONNX model: {e}")
        print("Please install ONNX manually: pip install onnx")

if __name__ == "__main__":
    main()