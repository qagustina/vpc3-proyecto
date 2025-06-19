#!/usr/bin/env python3
"""
Evaluation script for OCR models.

This script provides functionality to evaluate trained OCR models on test datasets.

Usage:
    python src/evaluate.py --model trocr --test-data data/processed/test --model-path models/finetuned_trocr

Example:
    python src/evaluate.py \
        --model trocr \
        --test-data data/processed/test \
        --model-path models/finetuned_trocr \
        --output-dir results/evaluation \
        --save-predictions
"""

import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import get_model, MODEL_REGISTRY


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate OCR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to evaluate"
    )
    parser.add_argument(
        "--test-data", 
        type=str, 
        required=True,
        help="Path to test dataset directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model-path", 
        type=str, 
        default=None,
        help="Path to trained model (if not provided, uses base model)"
    )
    parser.add_argument(
        "--use-pretrained", 
        action="store_true",
        help="Use pretrained HuggingFace model instead of custom trained model (ignores --model-path)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Output directory for evaluation results (default: results)"
    )
    parser.add_argument(
        "--save-predictions", 
        action="store_true",
        help="Save detailed predictions to CSV file"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1,
        help="Batch size for evaluation (default: 1)"
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
    """Validate input paths."""
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data directory not found: {args.test_data}")
    
    test_labels = os.path.join(args.test_data, 'labels.csv')
    if not os.path.exists(test_labels):
        raise FileNotFoundError(f"Test labels file not found: {test_labels}")
    
    if args.model_path and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)


def save_evaluation_config(args, output_dir):
    """Save evaluation configuration to output directory."""
    config = {
        'model_type': args.model,
        'test_data': args.test_data,
        'model_path': args.model_path,
        'use_pretrained': args.use_pretrained,
        'base_model': args.base_model,
        'batch_size': args.batch_size,
        'save_predictions': args.save_predictions,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(output_dir, 'evaluation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Evaluation configuration saved to: {config_path}")


def print_metrics_summary(metrics):
    """Print a formatted summary of evaluation metrics."""
    print("\n📊 Evaluation Results Summary")
    print("=" * 50)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Mean Character Accuracy: {metrics['mean_char_accuracy']:.4f}")
    print(f"Mean Character Error Rate (CER): {metrics['mean_cer']:.4f}")
    print(f"Exact Match Rate: {metrics['exact_match_rate']:.4f}")
    print("=" * 50)


def save_results(metrics, results_df, output_dir, save_predictions):
    """Save evaluation results to files."""
    # Save metrics summary
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if hasattr(value, 'item'):  # numpy types
                metrics_json[key] = value.item()
            else:
                metrics_json[key] = value
        json.dump(metrics_json, f, indent=2)
    
    print(f"📁 Metrics saved to: {metrics_path}")
    
    # Save detailed predictions if requested
    if save_predictions and results_df is not None:
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        results_df.to_csv(predictions_path, index=False)
        print(f"📁 Detailed predictions saved to: {predictions_path}")
    
    # Save summary report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("OCR Model Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total Samples: {metrics['total_samples']}\n")
        f.write(f"Mean Character Accuracy: {metrics['mean_char_accuracy']:.4f}\n")
        f.write(f"Mean Character Error Rate (CER): {metrics['mean_cer']:.4f}\n")
        f.write(f"Exact Match Rate: {metrics['exact_match_rate']:.4f}\n")
        
        if results_df is not None:
            f.write("\nDetailed Statistics:\n")
            f.write(f"Character Accuracy Statistics:\n")
            f.write(f"  Min: {results_df['char_accuracy'].min():.4f}\n")
            f.write(f"  Max: {results_df['char_accuracy'].max():.4f}\n")
            f.write(f"  Std: {results_df['char_accuracy'].std():.4f}\n")
            f.write(f"  25th percentile: {results_df['char_accuracy'].quantile(0.25):.4f}\n")
            f.write(f"  75th percentile: {results_df['char_accuracy'].quantile(0.75):.4f}\n")
    
    print(f"📁 Evaluation report saved to: {report_path}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    print("🔍 Starting OCR Model Evaluation")
    print("=" * 50)
    print(f"Model Type: {args.model}")
    print(f"Test Data: {args.test_data}")
    model_source = "Using pretrained HuggingFace model" if args.use_pretrained else (args.model_path or "Using base model")
    print(f"Model Path: {model_source}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 50)
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Save evaluation configuration
        save_evaluation_config(args, args.output_dir)
        
        # Initialize model
        model_kwargs = {}
        if args.base_model:
            model_kwargs['model_name'] = args.base_model
        
        model = get_model(args.model, **model_kwargs)
        print(f"✅ Initialized {args.model} model")
        
        # Determine model path based on flags
        model_path_to_use = None if args.use_pretrained else args.model_path
        
        # Start evaluation
        print("🔍 Starting evaluation...")
        metrics, results_df = model.evaluate(
            test_dataset_path=args.test_data,
            model_path=model_path_to_use
        )
        
        print("✅ Evaluation completed successfully!")
        
        # Print and save results
        print_metrics_summary(metrics)
        save_results(metrics, results_df, args.output_dir, args.save_predictions)
        
        # Show some example predictions
        if results_df is not None and len(results_df) > 0:
            print("\n🎯 Sample Predictions:")
            print("-" * 80)
            # Show best and worst examples
            best_examples = results_df.nlargest(3, 'char_accuracy')
            worst_examples = results_df.nsmallest(3, 'char_accuracy')
            
            print("Best Examples (Highest Accuracy):")
            for idx, row in best_examples.iterrows():
                print(f"  Ground Truth: '{row['ground_truth']}'")
                print(f"  Prediction:   '{row['prediction']}'")
                print(f"  Accuracy:     {row['char_accuracy']:.4f}")
                print()
            
            print("Worst Examples (Lowest Accuracy):")
            for idx, row in worst_examples.iterrows():
                print(f"  Ground Truth: '{row['ground_truth']}'")
                print(f"  Prediction:   '{row['prediction']}'")
                print(f"  Accuracy:     {row['char_accuracy']:.4f}")
                print()
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 