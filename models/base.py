"""
Base classes and interfaces for OCR models.

This module provides abstract base classes and common functionality
that can be shared across different OCR model implementations.
"""

import os
import torch
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class BaseOCRDataset(Dataset):
    """Base dataset class for OCR training and evaluation."""
    
    def __init__(self, root_dir: str, max_length: int = 64):
        self.root_dir = root_dir
        self.max_length = max_length
        
        # Load labels
        labels_path = os.path.join(root_dir, 'labels.csv')
        df = pd.read_csv(labels_path, header=None, names=['file_name', 'text'])
        
        # Clean data
        df.dropna(inplace=True)
        df = df[df['text'].str.len() > 0]
        df.reset_index(drop=True, inplace=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    @abstractmethod
    def __getitem__(self, idx):
        """Must be implemented by specific model datasets."""
        pass


class BaseOCRModel(ABC):
    """Abstract base class for OCR models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def load_model(self, model_path: Optional[str] = None):
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def train(self, train_dataset_path: str, eval_dataset_path: str, output_dir: str, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def evaluate(self, test_dataset_path: str, model_path: Optional[str] = None) -> Dict:
        """Evaluate the model and return metrics."""
        pass
    
    @abstractmethod
    def infer(self, image_path: str, model_path: Optional[str] = None) -> str:
        """Perform inference on a single image."""
        pass
    
    def infer_batch(self, image_paths: List[str], model_path: Optional[str] = None) -> List[str]:
        """Perform inference on a batch of images."""
        results = []
        for image_path in image_paths:
            result = self.infer(image_path, model_path)
            results.append(result)
        return results


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def character_accuracy(predicted: str, ground_truth: str) -> float:
    """Calculate character-level accuracy."""
    if len(ground_truth) == 0:
        return 1.0 if len(predicted) == 0 else 0.0
    edit_distance = levenshtein_distance(predicted, ground_truth)
    return 1.0 - (edit_distance / len(ground_truth))


def evaluate_predictions(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Calculate evaluation metrics for predictions vs ground truths."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    results = []
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = pred.lower().strip()
        gt_norm = gt.lower().strip()
        
        char_acc = character_accuracy(pred_norm, gt_norm)
        edit_dist = levenshtein_distance(pred_norm, gt_norm)
        
        results.append({
            'prediction': pred,
            'ground_truth': gt,
            'char_accuracy': char_acc,
            'edit_distance': edit_dist
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    metrics = {
        'mean_char_accuracy': results_df['char_accuracy'].mean(),
        'mean_cer': results_df['edit_distance'].mean() / results_df['ground_truth'].str.len().mean(),
        'exact_match_rate': (results_df['edit_distance'] == 0).mean(),
        'total_samples': len(results_df)
    }
    
    return metrics, results_df 