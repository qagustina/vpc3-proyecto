#!/usr/bin/env python3
"""
Inference script for OCR models.

This script provides functionality to perform OCR inference on images using trained models.
Currently supports TrOCR with plans for additional models.

Usage:
    python src/infer.py --model trocr --image path/to/image.png --model-path models/finetuned_trocr

Examples:
    # Single image inference
    python src/infer.py \
        --model trocr \
        --image imgs/date.png \
        --model-path models/finetuned_trocr

    # Batch inference on a directory
    python src/infer.py \
        --model trocr \
        --input-dir path/to/images/ \
        --model-path models/finetuned_trocr \
        --output-file results.csv

    # Using base model (no fine-tuning)
    python src/infer.py \
        --model trocr \
        --image imgs/date.png
"""

import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import get_model, MODEL_REGISTRY


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform OCR inference on images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to use for inference"
    )
    
    # Image input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", 
        type=str,
        help="Path to single image file"
    )
    input_group.add_argument(
        "--input-dir", 
        type=str,
        help="Path to directory containing images"
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
        "--output-file", 
        type=str, 
        default=None,
        help="Output CSV file for batch inference results"
    )
    parser.add_argument(
        "--image-extensions", 
        type=str, 
        nargs='+',
        default=['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
        help="Image file extensions to process (default: png, jpg, jpeg, bmp, tiff)"
    )
    parser.add_argument(
        "--recursive", 
        action="store_true",
        help="Search for images recursively in subdirectories"
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
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Image file not found: {args.image}")
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    if args.model_path and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")


def find_images(input_dir, extensions, recursive=False):
    """Find image files in directory."""
    image_files = []
    
    if recursive:
        for ext in extensions:
            pattern = os.path.join(input_dir, "**", f"*{ext}")
            image_files.extend(glob.glob(pattern, recursive=True))
    else:
        for ext in extensions:
            pattern = os.path.join(input_dir, f"*{ext}")
            image_files.extend(glob.glob(pattern))
    
    return sorted(image_files)


def perform_single_inference(model, image_path, model_path=None):
    """Perform inference on a single image."""
    print(f"🖼️  Processing image: {image_path}")
    
    try:
        prediction = model.infer(image_path, model_path)
        print(f"📝 Predicted text: '{prediction}'")
        return prediction
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None


def perform_batch_inference(model, image_paths, model_path=None, output_file=None):
    """Perform inference on multiple images."""
    print(f"🖼️  Processing {len(image_paths)} images...")
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            prediction = model.infer(image_path, model_path)
            results.append({
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            })
            print(f"  📝 Predicted: '{prediction}'")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'image_path': image_path,
                'filename': os.path.basename(image_path),
                'prediction': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # Save results if output file specified
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"📁 Results saved to: {output_file}")
    
    return results


def print_batch_summary(results):
    """Print summary of batch inference results."""
    total = len(results)
    successful = sum(1 for r in results if r.get('prediction') is not None)
    failed = total - successful
    
    print("\n📊 Batch Inference Summary")
    print("=" * 40)
    print(f"Total images processed: {total}")
    print(f"Successful predictions: {successful}")
    print(f"Failed predictions: {failed}")
    print(f"Success rate: {successful/total*100:.1f}%")
    print("=" * 40)


def main():
    """Main inference function."""
    args = parse_args()
    
    print("🔮 Starting OCR Inference")
    print("=" * 50)
    print(f"Model Type: {args.model}")
    model_source = "Using pretrained HuggingFace model" if args.use_pretrained else (args.model_path or "Using base model")
    print(f"Model Path: {model_source}")
    
    if args.image:
        print(f"Input Image: {args.image}")
    else:
        print(f"Input Directory: {args.input_dir}")
        print(f"Recursive: {args.recursive}")
        print(f"Extensions: {args.image_extensions}")
    
    if args.output_file:
        print(f"Output File: {args.output_file}")
    
    print("=" * 50)
    
    try:
        # Validate paths
        validate_paths(args)
        
        # Initialize model
        model_kwargs = {}
        if args.base_model:
            model_kwargs['model_name'] = args.base_model
        
        model = get_model(args.model, **model_kwargs)
        print(f"✅ Initialized {args.model} model")
        
        # Determine model path based on flags
        model_path_to_use = None if args.use_pretrained else args.model_path
        
        # Perform inference
        if args.image:
            # Single image inference
            prediction = perform_single_inference(model, args.image, model_path_to_use)
            if prediction is None:
                sys.exit(1)
        else:
            # Batch inference
            image_paths = find_images(args.input_dir, args.image_extensions, args.recursive)
            
            if not image_paths:
                print("❌ No images found in the specified directory")
                sys.exit(1)
            
            results = perform_batch_inference(model, image_paths, model_path_to_use, args.output_file)
            print_batch_summary(results)
            
            # Show some example results
            print("\n🎯 Sample Results:")
            print("-" * 80)
            for i, result in enumerate(results[:5]):  # Show first 5 results
                if result.get('prediction'):
                    print(f"{i+1}. {result['filename']}: '{result['prediction']}'")
                else:
                    print(f"{i+1}. {result['filename']}: ERROR - {result.get('error', 'Unknown error')}")
            
            if len(results) > 5:
                print(f"... and {len(results) - 5} more results")
        
        print("\n✅ Inference completed successfully!")
        
    except Exception as e:
        print(f"❌ Inference failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 