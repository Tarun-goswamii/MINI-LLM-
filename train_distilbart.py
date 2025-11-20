#!/usr/bin/env python3
"""
Simple DistilBART Training Script for Text Summarization
This script trains a DistilBART model on the SAMSum dataset for conversation summarization.
"""

import os
import torch
import pandas as pd
import urllib.request
import zipfile
from datasets import load_from_disk, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq,
    TrainingArguments, 
    Trainer,
    pipeline
)
import evaluate
from tqdm import tqdm

def download_and_prepare_data():
    """Download and prepare the SAMSum dataset"""
    print("📥 Downloading SAMSum dataset...")
    
    # Try to load from disk first, if not available download
    try:
        dataset = load_from_disk('samsum_dataset')
        print("✅ Dataset loaded from local cache")
    except:
        print("Downloading from Hugging Face...")
        try:
            # Download using urllib (Windows compatible)
            url = "https://github.com/entbappy/Branching-tutorial/raw/master/summarizer-data.zip"
            print("Downloading dataset archive...")
            urllib.request.urlretrieve(url, "summarizer-data.zip")
            
            # Extract using zipfile (Windows compatible)
            print("Extracting dataset...")
            with zipfile.ZipFile("summarizer-data.zip", 'r') as zip_ref:
                zip_ref.extractall(".")
            
            # Clean up
            os.remove("summarizer-data.zip")
            
            dataset = load_from_disk('samsum_dataset')
            print("✅ Dataset downloaded and extracted")
            
        except Exception as e:
            print(f"Download failed: {e}")
            print("Trying alternative: loading directly from Hugging Face...")
            dataset = load_dataset("samsum")
            print("✅ Dataset loaded from Hugging Face Hub")
    
    print(f"Dataset splits: {list(dataset.keys())}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    
    return dataset

def setup_model_and_tokenizer():
    """Initialize a fast, lightweight model for training"""
    print("🤖 Loading lightweight model for fast training...")
    
    # Use a much smaller model that downloads quickly
    model_name = "sshleifer/distilbart-cnn-6-6"  # Even smaller version (150MB)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"✅ Model loaded: {model_name}")
    except:
        # Ultimate fallback to tiny model
        print("Falling back to ultra-lightweight T5 model...")
        model_name = "t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"✅ Model loaded: {model_name}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"📱 Device: {device}")
    print(f"💾 Model size: ~150MB (fast download)")
    
    return model, tokenizer, device

def preprocess_data(dataset, tokenizer):
    """Convert dataset to model input format"""
    print("🔄 Preprocessing data...")
    
    def convert_examples_to_features(example_batch):
        input_encodings = tokenizer(
            example_batch['dialogue'], 
            max_length=512,  # Reduced for faster training
            truncation=True, 
            padding=False
        )
        
        with tokenizer.as_target_tokenizer():
            target_encodings = tokenizer(
                example_batch['summary'], 
                max_length=128, 
                truncation=True, 
                padding=False
            )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    # Process dataset
    processed_dataset = dataset.map(
        convert_examples_to_features, 
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    print("✅ Data preprocessing complete")
    return processed_dataset

def train_model(model, tokenizer, dataset, device):
    """Train the DistilBART model"""
    print("🚀 Starting training...")
    
    # Training arguments optimized for speed
    training_args = TrainingArguments(
        output_dir='./distilbart-samsum-trained',
        num_train_epochs=1,                    # Quick training
        per_device_train_batch_size=4,         # Larger batch for speed
        per_device_eval_batch_size=4,
        warmup_steps=100,                      # Reduced warmup
        weight_decay=0.01,
        logging_steps=50,
        evaluation_strategy='steps',
        eval_steps=200,
        save_steps=500,
        gradient_accumulation_steps=4,
        fp16=True if device == "cuda" else False,  # Speed up with mixed precision
        dataloader_num_workers=0,              # Avoid multiprocessing issues
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=None,                        # Disable wandb logging
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Use smaller subset for faster training (optional)
    train_dataset = dataset['train'].select(range(min(5000, len(dataset['train']))))
    eval_dataset = dataset['validation'].select(range(min(500, len(dataset['validation']))))
    
    print(f"Training on {len(train_dataset)} samples")
    print(f"Evaluating on {len(eval_dataset)} samples")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    print("🏋️ Training in progress...")
    trainer.train()
    
    print("✅ Training completed!")
    return trainer

def save_model(trainer, tokenizer):
    """Save the trained model and tokenizer"""
    print("💾 Saving trained model...")
    
    # Save model
    trainer.save_model("distilbart-samsum-model")
    
    # Save tokenizer
    tokenizer.save_pretrained("distilbart-tokenizer")
    
    print("✅ Model saved to:")
    print("   - distilbart-samsum-model/")
    print("   - distilbart-tokenizer/")

def test_model():
    """Test the trained model"""
    print("🧪 Testing trained model...")
    
    try:
        # Load trained model
        pipe = pipeline(
            "summarization", 
            model="distilbart-samsum-model",
            tokenizer="distilbart-tokenizer",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Test with sample text
        test_text = """
        John: Hey Sarah, how was your weekend?
        Sarah: It was great! I went hiking with my family. We visited the national park and saw some amazing waterfalls.
        John: That sounds wonderful! Did you take any photos?
        Sarah: Yes, tons! The scenery was breathtaking. We also had a picnic by the lake.
        John: I'm jealous. I just stayed home and watched Netflix.
        Sarah: You should come with us next time! It's really refreshing to be in nature.
        """
        
        result = pipe(test_text, max_length=60, min_length=20, do_sample=False)
        
        print("\n📝 Test Results:")
        print("Input:", test_text.strip())
        print("\n📄 Generated Summary:", result[0]['summary_text'])
        print("\n✅ Model test successful!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    """Main training pipeline"""
    print("🎯 DistilBART Training Pipeline Started")
    print("=" * 50)
    
    try:
        # Step 1: Download data
        dataset = download_and_prepare_data()
        
        # Step 2: Setup model
        model, tokenizer, device = setup_model_and_tokenizer()
        
        # Step 3: Preprocess data
        processed_dataset = preprocess_data(dataset, tokenizer)
        
        # Step 4: Train model
        trainer = train_model(model, tokenizer, processed_dataset, device)
        
        # Step 5: Save model
        save_model(trainer, tokenizer)
        
        # Step 6: Test model
        test_model()
        
        print("\n🎉 Training Pipeline Completed Successfully!")
        print("Your custom DistilBART model is ready to use!")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("Please check the error message and try again.")

if __name__ == "__main__":
    main()