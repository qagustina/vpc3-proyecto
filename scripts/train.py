#!/usr/bin/env python3
"""
Training script for OCR models.

This script provides functionality to train various OCR models on datasets.
Currently supports TrOCR with plans for additional models.

Usage:
    python scripts/train.py --model trocr --train-data data/processed/train --val-data data/processed/val --output models/my_model

Example:
    python scripts/train.py \
        --model trocr \
        --train-data data/processed/train \
        --val-data data/processed/val \
        --output models/finetuned_trocr \
        --epochs 5 \
        --batch-size 16 \
        --learning-rate 1e-4
"""

import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import get_model, MODEL_REGISTRY

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train OCR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to train"
    )
    parser.add_argument(
        "--train-data", 
        type=str, 
        required=True,
        help="Path to training dataset directory"
    )
    parser.add_argument(
        "--val-data", 
        type=str, 
        required=True,
        help="Path to validation dataset directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output directory for trained model"
    )
    
    # Optional training parameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--save-steps", 
        type=int, 
        default=1000,
        help="Save checkpoint every N steps (default: 1000)"
    )
    parser.add_argument(
        "--eval-steps", 
        type=int, 
        default=1000,
        help="Evaluate every N steps (default: 1000)"
    )
    parser.add_argument(
        "--logging-steps", 
        type=int, 
        default=50,
        help="Log every N steps (default: 50)"
    )
    
    # Model-specific arguments
    parser.add_argument(
        "--base-model", 
        type=str, 
        default=None,
        help="Base model name/path (model-specific)"
    )
    
    return parser.parse_args()


def validate_paths(args):
    """Validate input and output paths."""
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data directory not found: {args.train_data}")
    
    if not os.path.exists(args.val_data):
        raise FileNotFoundError(f"Validation data directory not found: {args.val_data}")
    
    train_labels = os.path.join(args.train_data, 'labels.csv')
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"Training labels file not found: {train_labels}")
    
    val_labels = os.path.join(args.val_data, 'labels.csv')
    if not os.path.exists(val_labels):
        raise FileNotFoundError(f"Validation labels file not found: {val_labels}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)


def save_training_config(args, output_dir):
    """Save training configuration to output directory."""
    config = {
        'model_type': args.model,
        'train_data': args.train_data,
        'val_data': args.val_data,
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'save_steps': args.save_steps,
            'eval_steps': args.eval_steps,
            'logging_steps': args.logging_steps
        },
        'base_model': args.base_model,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    print("🚀 Starting OCR Model Training")
    print("=" * 50)
    print(f"Model Type: {args.model}")
    print(f"Training Data: {args.train_data}")
    print(f"Validation Data: {args.val_data}")
    print(f"Output Directory: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print("=" * 50)
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Save training configuration
        save_training_config(args, args.output)
        
        # Initialize model
        model_kwargs = {}
        if args.base_model:
            model_kwargs['model_name'] = args.base_model
        
        model = get_model(args.model, **model_kwargs)
        print(f"✅ Initialized {args.model} model")
        
        # Start training
        print("🏋️ Starting training...")
        trainer = model.train(
            train_dataset_path=args.train_data,
            eval_dataset_path=args.val_data,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps
        )
        
        print("✅ Training completed successfully!")
        print(f"📁 Model saved to: {args.output}")
        
        # Print final metrics if available
        if hasattr(trainer, 'state') and trainer.state.log_history:
            final_metrics = trainer.state.log_history[-1]
            print("\n📊 Final Training Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 