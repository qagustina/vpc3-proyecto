#!/usr/bin/env python3
"""
TrOCR Attention Visualization Script

This script generates attention maps to visualize which parts of an image
the TrOCR model focuses on when generating text predictions.

Usage:
    python scripts/attention_viz.py --model trocr --image imgs/date.png --output attention_maps/

Example:
    python scripts/attention_viz.py \
        --model trocr \
        --image imgs/date.png \
        --model-path models/my_trocr_model \
        --output attention_maps/ \
        --save-tokens
"""

import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models import get_model, MODEL_REGISTRY


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate attention maps for TrOCR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model type to use for attention visualization"
    )
    parser.add_argument(
        "--image", 
        type=str,
        required=True,
        help="Path to input image file"
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
        help="Use pretrained HuggingFace model instead of custom trained model"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="attention_maps",
        help="Output directory for attention visualizations (default: attention_maps)"
    )
    parser.add_argument(
        "--save-tokens", 
        action="store_true",
        help="Save token-level attention maps (creates many files)"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=150,
        help="DPI for output images (default: 150)"
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
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    if args.model_path and not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)


def create_attention_overlay(image, attention_weights, alpha=0.6, colormap='hot'):
    """Create attention overlay on original image."""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Resize attention weights to match image size
    from scipy.ndimage import zoom
    h, w = img_array.shape[:2]
    att_h, att_w = attention_weights.shape
    
    zoom_factors = (h / att_h, w / att_w)
    attention_resized = zoom(attention_weights, zoom_factors, order=1)
    
    # Normalize attention weights
    attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min())
    
    # Create colormap
    cmap = plt.cm.get_cmap(colormap)
    attention_colored = cmap(attention_norm)
    
    # Convert to 0-255 range
    attention_colored = (attention_colored[:, :, :3] * 255).astype(np.uint8)
    
    # Blend with original image
    blended = (1 - alpha) * img_array + alpha * attention_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def create_grid_visualization(image, attention_weights, patch_size=16):
    """Create grid visualization showing attention on image patches."""
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Calculate number of patches
    num_patches_h = attention_weights.shape[0]
    num_patches_w = attention_weights.shape[1]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Attention visualization
    ax2.imshow(image)
    
    # Overlay attention as colored rectangles
    patch_h = h / num_patches_h
    patch_w = w / num_patches_w
    
    # Normalize attention weights
    att_norm = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            y = i * patch_h
            x = j * patch_w
            alpha = att_norm[i, j] * 0.7  # Scale alpha
            
            rect = patches.Rectangle((x, y), patch_w, patch_h, 
                                   linewidth=0, facecolor='red', alpha=alpha)
            ax2.add_patch(rect)
    
    ax2.set_title("Attention Map")
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


def save_attention_summary(image_path, prediction, attention_info, output_file):
    """Save attention analysis summary."""
    summary = {
        'image_path': image_path,
        'prediction': prediction,
        'attention_info': attention_info,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    """Main attention visualization function."""
    args = parse_args()
    
    print("🔍 Starting TrOCR Attention Visualization")
    print("=" * 60)
    print(f"Model Type: {args.model}")
    
    model_source = "Using pretrained HuggingFace model" if args.use_pretrained else (args.model_path or "Using base model")
    print(f"Model Path: {model_source}")
    print(f"Input Image: {args.image}")
    print(f"Output Directory: {args.output}")
    print("=" * 60)
    
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
        
        # Check if model supports attention visualization
        if not hasattr(model, 'get_attention_maps'):
            print("❌ This model doesn't support attention visualization")
            sys.exit(1)
        
        # Load image
        image = Image.open(args.image).convert("RGB")
        print(f"📸 Loaded image: {image.size}")
        
        # Get prediction and attention maps
        print("🧠 Generating attention maps...")
        result = model.get_attention_maps(args.image, model_path_to_use)
        
        prediction = result['prediction']
        attention_maps = result['attention_maps']
        tokens = result.get('tokens', [])
        
        print(f"📝 Predicted text: '{prediction}'")
        print(f"🎯 Generated {len(attention_maps)} attention maps")
        
        # Create base filename
        image_name = Path(args.image).stem
        
        # Save overall attention map (averaged across all tokens)
        if attention_maps:
            avg_attention = np.mean(attention_maps, axis=0)
            
            # Create overlay visualization
            overlay_img = create_attention_overlay(image, avg_attention)
            overlay_path = os.path.join(args.output, f"{image_name}_attention_overlay.png")
            overlay_img.save(overlay_path, dpi=(args.dpi, args.dpi))
            print(f"💾 Saved attention overlay: {overlay_path}")
            
            # Create grid visualization
            grid_fig = create_grid_visualization(image, avg_attention)
            grid_path = os.path.join(args.output, f"{image_name}_attention_grid.png")
            grid_fig.savefig(grid_path, dpi=args.dpi, bbox_inches='tight')
            plt.close(grid_fig)
            print(f"💾 Saved attention grid: {grid_path}")
            
            # Save raw attention heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(avg_attention, cmap='hot', interpolation='nearest')
            plt.colorbar(label='Attention Weight')
            plt.title(f'Average Attention Map\nPrediction: "{prediction}"')
            heatmap_path = os.path.join(args.output, f"{image_name}_attention_heatmap.png")
            plt.savefig(heatmap_path, dpi=args.dpi, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved attention heatmap: {heatmap_path}")
        
        # Save token-level attention maps if requested
        if args.save_tokens and tokens:
            token_dir = os.path.join(args.output, f"{image_name}_tokens")
            os.makedirs(token_dir, exist_ok=True)
            
            for i, (token, attention_map) in enumerate(zip(tokens, attention_maps)):
                # Create overlay for this token
                token_overlay = create_attention_overlay(image, attention_map)
                token_path = os.path.join(token_dir, f"token_{i:02d}_{token}.png")
                token_overlay.save(token_path, dpi=(args.dpi, args.dpi))
            
            print(f"💾 Saved {len(tokens)} token attention maps to: {token_dir}")
        
        # Save attention summary
        attention_info = {
            'num_attention_maps': len(attention_maps),
            'tokens': tokens,
            'attention_shape': attention_maps[0].shape if attention_maps else None,
            'avg_attention_stats': {
                'min': float(avg_attention.min()) if attention_maps else None,
                'max': float(avg_attention.max()) if attention_maps else None,
                'mean': float(avg_attention.mean()) if attention_maps else None,
                'std': float(avg_attention.std()) if attention_maps else None
            } if attention_maps else None
        }
        
        summary_path = os.path.join(args.output, f"{image_name}_attention_summary.json")
        save_attention_summary(args.image, prediction, attention_info, summary_path)
        print(f"💾 Saved attention summary: {summary_path}")
        
        print("\n✅ Attention visualization completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Attention visualization failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 