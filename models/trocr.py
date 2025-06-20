"""
TrOCR model implementation for OCR tasks.

This module contains TrOCR-specific functionality including the model class,
dataset, and configuration specific to the Transformers TrOCR architecture.
"""

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator
)
import evaluate

from .base import BaseOCRModel, BaseOCRDataset, evaluate_predictions


class TrOCRDataset(BaseOCRDataset):
    """Dataset class specifically for TrOCR training and evaluation."""
    
    def __init__(self, root_dir: str, processor, max_length: int = 64):
        super().__init__(root_dir, max_length)
        self.processor = processor

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = str(self.df['text'][idx])

        # Load and process image
        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Process text labels
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        ).input_ids

        # Replace padding tokens with -100 for loss calculation
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels)
        }


class TrOCRModel(BaseOCRModel):
    """TrOCR model implementation using HuggingFace Transformers."""
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten"):
        super().__init__(model_name)
        self.cer_metric = evaluate.load("cer")
    
    def load_model(self, model_path: Optional[str] = None):
        """Load TrOCR model and processor."""
        if model_path and os.path.exists(model_path):
            self.processor = TrOCRProcessor.from_pretrained(model_path, use_fast=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        else:
            self.processor = TrOCRProcessor.from_pretrained(self.model_name, use_fast=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        
        # Configure model
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        self.model.to(self.device)
    
    def train(self, 
              train_dataset_path: str, 
              eval_dataset_path: str, 
              output_dir: str,
              num_epochs: int = 3,
              batch_size: int = 8,
              learning_rate: float = 5e-5,
              save_steps: int = 1000,
              eval_steps: int = 1000,
              logging_steps: int = 50,
              **kwargs):
        """Train the TrOCR model."""
        
        if self.model is None or self.processor is None:
            self.load_model()
        
        # Create datasets
        train_dataset = TrOCRDataset(train_dataset_path, self.processor)
        eval_dataset = TrOCRDataset(eval_dataset_path, self.processor)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=torch.cuda.is_available(),
            output_dir=output_dir,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            report_to="none",
            **kwargs
        )
        
        # Create trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            compute_metrics=self._compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=default_data_collator,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.processor.save_pretrained(output_dir)
        
        return trainer
    
    def _compute_metrics(self, pred):
        """Compute CER metric for evaluation during training."""
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(labels_ids, skip_special_tokens=True)

        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}
    
    def evaluate(self, test_dataset_path: str, model_path: Optional[str] = None) -> Tuple[Dict, pd.DataFrame]:
        """Evaluate the model on test dataset."""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_model()
        
        # Load test data
        test_df = pd.read_csv(os.path.join(test_dataset_path, 'labels.csv'), 
                              header=None, names=['image_path', 'text'])
        
        predictions = []
        ground_truths = []
        
        self.model.eval()
        
        for _, row in test_df.iterrows():
            image_path = os.path.join(test_dataset_path, row['image_path'])
            ground_truth = str(row['text'])
            
            try:
                prediction = self.infer(image_path)
                predictions.append(prediction)
                ground_truths.append(ground_truth)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Use generic evaluation function
        return evaluate_predictions(predictions, ground_truths)
    
    def infer(self, image_path: str, model_path: Optional[str] = None) -> str:
        """Perform inference on a single image."""
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_model()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=128)
        
        prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return prediction
    
    def get_attention_maps(self, image_path: str, model_path: Optional[str] = None) -> Dict:
        """
        Generate attention maps for visualization.
        
        Returns:
            Dict containing:
                - prediction: The predicted text
                - attention_maps: List of 2D attention weight arrays (one per token)
                - tokens: List of generated tokens
        """
        if model_path:
            self.load_model(model_path)
        elif self.model is None:
            self.load_model()
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        self.model.eval()
        
        # Generate with attention output
        with torch.no_grad():
            # Get encoder outputs first
            encoder_outputs = self.model.encoder(pixel_values=pixel_values)
            
            # Generate tokens step by step to collect attention
            generated_ids = []
            attention_weights_list = []
            decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]], device=self.device)
            
            for step in range(128):  # Max length
                # Forward pass through decoder
                decoder_outputs = self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    output_attentions=True,
                    return_dict=True
                )
                
                # Get next token
                logits = decoder_outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1)
                
                # Stop if EOS token
                if next_token_id.item() == self.processor.tokenizer.eos_token_id:
                    break
                
                # Extract cross-attention weights (decoder attending to encoder)
                # decoder_outputs.cross_attentions contains attention weights for each layer
                # We'll use the last layer's attention
                if decoder_outputs.cross_attentions:
                    # Shape: [batch_size, num_heads, sequence_length, encoder_sequence_length]
                    cross_attention = decoder_outputs.cross_attentions[-1]  # Last layer
                    
                    # Average across heads and take the last token's attention
                    token_attention = cross_attention[0, :, -1, :].mean(dim=0)  # [encoder_sequence_length]
                    
                    # Convert to 2D spatial attention map
                    # TrOCR encoder creates a 2D grid of patches from the image
                    attention_2d = self._reshape_attention_to_spatial(token_attention, encoder_outputs.last_hidden_state.shape[1])
                    attention_weights_list.append(attention_2d.cpu().numpy())
                
                # Append token and update input
                generated_ids.append(next_token_id.item())
                decoder_input_ids = torch.cat([decoder_input_ids, next_token_id.unsqueeze(1)], dim=1)
        
        # Decode tokens to text
        prediction = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Decode individual tokens for visualization
        tokens = [self.processor.tokenizer.decode([token_id], skip_special_tokens=True) 
                  for token_id in generated_ids]
        
        return {
            'prediction': prediction,
            'attention_maps': attention_weights_list,
            'tokens': tokens
        }
    
    def _reshape_attention_to_spatial(self, attention_1d: torch.Tensor, num_patches: int) -> torch.Tensor:
        """
        Reshape 1D attention weights to 2D spatial grid.
        
        Args:
            attention_1d: 1D attention weights [num_patches]
            num_patches: Total number of patches
            
        Returns:
            2D attention map
        """
        # For TrOCR, we need to figure out the spatial dimensions
        # The model uses ViT-like patch encoding, typically creating a roughly square grid
        
        # Calculate grid dimensions (assuming roughly square)
        grid_size = int(np.sqrt(num_patches))
        
        # Handle case where not perfectly square
        if grid_size * grid_size != num_patches:
            # Try to find best rectangular dimensions
            for h in range(1, int(np.sqrt(num_patches)) + 1):
                if num_patches % h == 0:
                    w = num_patches // h
                    if abs(h - w) <= 2:  # Prefer roughly square
                        grid_size_h, grid_size_w = h, w
                        break
            else:
                # Fallback to square with padding/truncation
                grid_size_h = grid_size_w = grid_size
                attention_1d = attention_1d[:grid_size * grid_size]
        else:
            grid_size_h = grid_size_w = grid_size
        
        # Reshape to 2D
        attention_2d = attention_1d[:grid_size_h * grid_size_w].reshape(grid_size_h, grid_size_w)
        
        return attention_2d 