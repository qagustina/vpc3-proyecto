"""
OCR Models package.

This package provides implementations of various OCR models and a registry
system for easy model instantiation.
"""

from .base import BaseOCRModel, BaseOCRDataset
from .trocr import TrOCRModel

MODEL_REGISTRY = {
    'trocr': TrOCRModel,
    # 'donut': DonutModel,
}


def get_model(model_type: str, **kwargs) -> BaseOCRModel:
    """Factory function to get model instances."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' not supported. Available: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](**kwargs)


__all__ = ['BaseOCRModel', 'BaseOCRDataset', 'TrOCRModel', 'MODEL_REGISTRY', 'get_model'] 