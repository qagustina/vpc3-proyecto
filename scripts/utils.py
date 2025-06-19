"""
Utility functions for OCR data processing and common operations.

This module provides helper functions for data preprocessing, visualization,
and other common tasks in OCR workflows.
"""

import os
import json
import zipfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from typing import Dict, List, Tuple, Optional


def load_coco_annotations(zip_path: str, json_filename: str = 'cocotext.v2.json') -> Dict:
    """Load COCO text annotations from zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        with zipf.open(json_filename) as file:
            return json.load(file)


def get_annotated_image_ids(coco_data: Dict) -> set:
    """Get set of image IDs that have annotations."""
    annotated_ids = set()
    for ann in coco_data['anns'].values():
        annotated_ids.add(ann['image_id'])
    return annotated_ids


def image_id_to_filename(image_id: int, dataset: str = 'train2014') -> str:
    """Convert image ID to COCO filename format."""
    return f'COCO_{dataset}_{image_id:012d}.jpg'


def filename_to_image_id(filename: str) -> int:
    """Convert COCO filename to image ID."""
    return int(filename.split('_')[-1].split('.')[0])


def get_annotations_by_image_id(coco_data: Dict, legibility_filter: str = 'legible') -> Dict:
    """Group annotations by image ID with optional legibility filter."""
    img_id_to_anns = defaultdict(list)
    for ann_id, ann in coco_data['anns'].items():
        if legibility_filter is None or ann['legibility'] == legibility_filter:
            img_id_to_anns[ann['image_id']].append(ann)
    return img_id_to_anns


def extract_images_from_zip(zip_path: str, output_dir: str, target_filenames: set) -> None:
    """Extract specific images from zip file."""
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zip_files = zipf.namelist()
        matched_files = [f for f in zip_files if os.path.basename(f) in target_filenames]

        if not matched_files:
            print(f'No matching files found in {zip_path}')
            return

        for file in tqdm(matched_files, desc=f'Extracting from {os.path.basename(zip_path)}'):
            zipf.extract(file, output_dir)


def create_cropped_dataset(image_dir: str, output_dir: str, img_id_to_anns: Dict) -> None:
    """Create cropped text regions dataset from images and annotations."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing images in '{image_dir}' and saving crops to '{output_dir}'...")

    labels_file_path = os.path.join(output_dir, 'labels.csv')
    labels_data = []

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in tqdm(image_files, desc="Cropping text regions"):
        try:
            image_id = filename_to_image_id(filename)
        except (ValueError, IndexError):
            continue

        if image_id not in img_id_to_anns:
            continue

        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        annotations = img_id_to_anns[image_id]

        for i, ann in enumerate(annotations):
            bbox = [int(p) for p in ann['bbox']]
            text_label = ann['utf8_string']

            if not text_label or len(text_label) < 1:
                continue

            x, y, w, h = bbox

            # Validate bbox
            if (w <= 0 or h <= 0 or x < 0 or y < 0 or 
                (x + w) > image.shape[1] or (y + h) > image.shape[0]):
                continue

            # Crop image
            cropped_image = image[y:y+h, x:x+w]
            crop_filename = f"{image_id}_{i}.png"

            # Save crop and label
            cv2.imwrite(os.path.join(output_dir, crop_filename), cropped_image)
            labels_data.append([crop_filename, text_label])

    # Save labels
    df = pd.DataFrame(labels_data, columns=['filename', 'text'])
    df.to_csv(labels_file_path, index=False, header=False)
    print(f"Saved {len(labels_data)} cropped images and labels to {output_dir}")


def visualize_predictions(images: List[str], predictions: List[str], 
                         ground_truths: List[str] = None, 
                         num_samples: int = 5, 
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    """Visualize OCR predictions with optional ground truth comparison."""
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Load and display image
        try:
            img = Image.open(images[i])
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading image:\n{str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Set title with prediction and ground truth
        title = f"Pred: '{predictions[i]}'"
        if ground_truths:
            title += f"\nGT: '{ground_truths[i]}'"
        
        ax.set_title(title, fontsize=10, wrap=True)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def calculate_dataset_statistics(labels_csv: str) -> Dict:
    """Calculate statistics for OCR dataset."""
    df = pd.read_csv(labels_csv, header=None, names=['filename', 'text'])
    
    # Text length statistics
    text_lengths = df['text'].str.len()
    
    stats = {
        'total_samples': len(df),
        'unique_texts': df['text'].nunique(),
        'text_length_stats': {
            'mean': text_lengths.mean(),
            'median': text_lengths.median(),
            'min': text_lengths.min(),
            'max': text_lengths.max(),
            'std': text_lengths.std()
        },
        'character_distribution': {},
        'word_count_stats': {}
    }
    
    # Character distribution
    all_chars = ''.join(df['text'].astype(str))
    char_counts = pd.Series(list(all_chars)).value_counts()
    stats['character_distribution'] = char_counts.head(20).to_dict()
    
    # Word count statistics
    word_counts = df['text'].str.split().str.len()
    stats['word_count_stats'] = {
        'mean': word_counts.mean(),
        'median': word_counts.median(),
        'min': word_counts.min(),
        'max': word_counts.max()
    }
    
    return stats


def print_dataset_statistics(labels_csv: str) -> None:
    """Print formatted dataset statistics."""
    stats = calculate_dataset_statistics(labels_csv)
    
    print("📊 Dataset Statistics")
    print("=" * 50)
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Unique texts: {stats['unique_texts']:,}")
    
    print("\n📝 Text Length Statistics:")
    tls = stats['text_length_stats']
    print(f"  Mean: {tls['mean']:.2f} characters")
    print(f"  Median: {tls['median']:.2f} characters")
    print(f"  Range: {tls['min']:.0f} - {tls['max']:.0f} characters")
    print(f"  Std Dev: {tls['std']:.2f}")
    
    print("\n📖 Word Count Statistics:")
    wcs = stats['word_count_stats']
    print(f"  Mean: {wcs['mean']:.2f} words")
    print(f"  Median: {wcs['median']:.2f} words")
    print(f"  Range: {wcs['min']:.0f} - {wcs['max']:.0f} words")
    
    print("\n🔤 Top 10 Characters:")
    for char, count in list(stats['character_distribution'].items())[:10]:
        if char == ' ':
            char_display = 'SPACE'
        elif char == '\n':
            char_display = 'NEWLINE'
        elif char == '\t':
            char_display = 'TAB'
        else:
            char_display = char
        print(f"  '{char_display}': {count:,}")


def validate_dataset_structure(dataset_dir: str) -> bool:
    """Validate that dataset directory has proper structure."""
    required_files = ['labels.csv']
    
    for file in required_files:
        file_path = os.path.join(dataset_dir, file)
        if not os.path.exists(file_path):
            print(f"❌ Missing required file: {file_path}")
            return False
    
    # Check if images exist for labels
    labels_df = pd.read_csv(os.path.join(dataset_dir, 'labels.csv'), 
                           header=None, names=['filename', 'text'])
    
    missing_images = []
    for filename in labels_df['filename'].head(10):  # Check first 10
        image_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(image_path):
            missing_images.append(filename)
    
    if missing_images:
        print(f"⚠️  Warning: Some images are missing (checked first 10): {missing_images}")
        return False
    
    print("✅ Dataset structure is valid")
    return True


def split_dataset(labels_csv: str, output_dir: str, 
                 train_ratio: float = 0.7, val_ratio: float = 0.2, 
                 test_ratio: float = 0.1, random_seed: int = 42) -> None:
    """Split dataset into train/val/test sets."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Load data
    df = pd.read_csv(labels_csv, header=None, names=['filename', 'text'])
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split indices
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Create output directories and save splits
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Save labels
        split_df.to_csv(os.path.join(split_dir, 'labels.csv'), 
                       index=False, header=False)
        
        print(f"✅ {split_name}: {len(split_df)} samples")
    
    print(f"📁 Dataset splits saved to: {output_dir}")


def resize_images_in_dataset(dataset_dir: str, target_size: Tuple[int, int] = (224, 224)) -> None:
    """Resize all images in dataset to target size."""
    labels_path = os.path.join(dataset_dir, 'labels.csv')
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(labels_path, header=None, names=['filename', 'text'])
    
    print(f"🔄 Resizing {len(df)} images to {target_size}...")
    
    for filename in tqdm(df['filename'], desc="Resizing images"):
        image_path = os.path.join(dataset_dir, filename)
        
        try:
            # Load image
            image = Image.open(image_path)
            
            # Resize while maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', target_size, (255, 255, 255))
            paste_x = (target_size[0] - image.width) // 2
            paste_y = (target_size[1] - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            # Save
            new_image.save(image_path)
            
        except Exception as e:
            print(f"Error resizing {filename}: {e}")
    
    print("✅ Image resizing completed") 