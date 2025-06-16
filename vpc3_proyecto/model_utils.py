import torch
from tqdm import tqdm
def compute_metrics(pred,processor):
    # Extraer logits (shape: [batch, seq_len, vocab_size])
    pred_ids = pred.predictions.argmax(-1)

    # Decodificar predicciones y etiquetas
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)

    # Limpiar cadenas (opcional, según formato de texto)
    pred_str = [s.strip() for s in pred_str]
    label_str = [s.strip() for s in label_str]

    return {
        "cer": cer(label_str, pred_str),
        "wer": wer(label_str, pred_str)
    }

def manual_evaluate(model, dataset, processor, max_samples=None):
    model.eval()
    predictions, references = [], []
    max_samples = max_samples or len(dataset)
    empty_reference_count = 0
    correct_empty_predictions = 0

    for i in tqdm(range(max_samples)):
        try:
            # 1. Load sample
            sample = dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(model.device)

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
            # 4. Handle empty references separately
            if not true_text:
                empty_reference_count += 1
                if not pred_text:
                    correct_empty_predictions += 1
                continue  # Skip jiwer calculation for empty references

            predictions.append(pred_text.strip())
            references.append(true_text.strip())

        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
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

    return metrics

from jiwer import wer, cer

