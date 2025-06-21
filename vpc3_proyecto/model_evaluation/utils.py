from jiwer import wer, cer

def compute_metrics(pred,donut_processor):
    # Extraer logits (shape: [batch, seq_len, vocab_size])
    pred_ids = pred.predictions.argmax(-1)

    # Decodificar predicciones y etiquetas
    pred_str = donut_processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = donut_processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    # Limpiar cadenas (opcional, según formato de texto)
    pred_str = [s.strip() for s in pred_str]
    label_str = [s.strip() for s in label_str]

    return {
        "cer": cer(label_str, pred_str),
        "wer": wer(label_str, pred_str)
    }

import torch
from tqdm import tqdm

import json
import os
from tqdm import tqdm
import torch
from jiwer import cer, wer


def save_results(metrics, predictions, references, save_path):
    """Save evaluation results to JSON files"""
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(save_path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions and references
    details_path = os.path.join(save_path, "resulting_metrics.json")
    with open(details_path, "w") as f:
        json.dump({
            "predictions": predictions,
            "references": references
        }, f, indent=2)

    print(f"Results saved to {save_path}")


def manual_evaluate(model, dataset, processor, max_samples=None, results_save_path=None):
    model.eval()
    predictions, references = [], []
    max_samples = max_samples or len(dataset)
    empty_reference_count = 0
    correct_empty_predictions = 0
    error_samples = []
    sample_details = []

    for i in tqdm(range(max_samples)):
        try:
            # 1. Load sample
            sample = dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(model.device)

            # Ensure type matches model precision
            if model.dtype == torch.float16:
                pixel_values = pixel_values.half()
            # 2. Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    max_length=512,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    num_beams=1,
                )

            # 3. Decode prediction
            pred_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            # 4. Safely decode labels
            valid_label_ids = sample["labels"][sample["labels"] != -100]  # Remove padding
            valid_label_ids = valid_label_ids[valid_label_ids < processor.tokenizer.vocab_size]  # Filter invalid IDs
            true_text = processor.decode(valid_label_ids, skip_special_tokens=True)

            # Record sample details
            sample_details.append({
                "id": i,
                "prediction": pred_text.lower().strip() if pred_text else None,
                "ground_truth": true_text.lower().strip() if true_text else None,
                "cer": cer(true_text.lower().strip(), pred_text.lower().strip()) if true_text else float('nan'),
                "wer": wer(true_text.lower().strip(), pred_text.lower().strip()) if true_text else float('nan'),
                "edit_distance" : levenshtein_distance(pred_text.lower().strip(), true_text.lower().strip()),
                "is_empty": not true_text,
                "char_accuracy": character_accuracy(pred_text.lower().strip(), true_text.lower().strip())
            })

            if not true_text:
                    empty_reference_count += 1
                    if not pred_text:
                        correct_empty_predictions += 1
                    continue  # Skip jiwer calculation for empty references

            predictions.append(pred_text.strip())
            references.append(true_text.strip())

        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            error_samples.append({
                "id": i,
                "error": str(e)
            })
            continue

        # Periodic cleanup
        if i % 10 == 0:
            torch.cuda.empty_cache()
    results_df = pd.DataFrame(sample_details)
    metrics = {
        'mean_char_accuracy': results_df['char_accuracy'].mean(),
        'mean_cer': results_df['edit_distance'].mean() / results_df['ground_truth'].str.len().mean(),
        'exact_match_rate': (results_df['edit_distance'] == 0).mean(),
        'total_samples': len(results_df)
    }

    # Save results if path provided
    if results_save_path:
        full_results = {
            "metrics": metrics,
            "sample_details": sample_details,
            "errors": error_samples
        }
        try:
            save_results(full_results,predictions,references, results_save_path)
        except Exception as e:
            print(f"Error saving results: {str(e)}")

    return metrics,results_df
# Usage

import os
import re

import os
import re


def get_last_checkpoint_folder(output_dir):
    """Find the latest checkpoint folder in the output directory"""
    try:
        # List all directories
        all_items = os.listdir(output_dir)
        checkpoints = [
            d for d in all_items
            if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
        ]

        if not checkpoints:
            return None

        # Extract step numbers (si hay muchas epochs) and sort
        def extract_step(d):
            numbers = re.findall(r"\d+", d)
            return int(numbers[-1]) if numbers else -1

        # Sort by step number
        checkpoints.sort(key=extract_step)

        last_checkpoint = checkpoints[-1]
        return os.path.join(output_dir, last_checkpoint)

    except Exception as e:
        print(f"Error finding checkpoints: {str(e)}")
        return None


import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


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
