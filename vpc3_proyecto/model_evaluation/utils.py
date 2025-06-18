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


def manual_evaluate(model, dataset, processor, max_samples=None,save_path=None):
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
                "prediction": pred_text.strip() if pred_text else None,
                "reference": true_text.strip() if true_text else None,
                "is_empty": not true_text
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

    # Calculate standard metrics (only non-empty references)
    metrics = {
        "cer": cer(references, predictions) if references else float('nan'),
        "wer": wer(references, predictions) if references else float('nan'),
        "samples_evaluated": len(references) + empty_reference_count,
        "empty_reference_accuracy": correct_empty_predictions / empty_reference_count if empty_reference_count > 0 else float(
            'nan'),
        "empty_references": empty_reference_count,
        "correct_empty_predictions": correct_empty_predictions
    }
    # Save results if path provided
    if save_path:
        full_results = {
            "metrics": metrics,
            "sample_details": sample_details,
            "errors": error_samples
        }
        save_results(full_results, save_path)

    return metrics
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
