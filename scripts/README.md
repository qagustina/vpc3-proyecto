# OCR Model Scripts

This directory contains the main scripts for training, evaluating, and performing inference with OCR models. The codebase is designed to be modular and extensible.

## Project Structure

```
src/
├── __init__.py          # Package initialization
├── train.py             # Model training script
├── evaluate.py          # Model evaluation script
├── infer.py             # Inference script for single/batch predictions
├── utils.py             # Utility functions for data processing
└── README.md            # This documentation

models/
└── model.py             # Model implementations and base classes
```

## Quick Start

### 1. Training a Model

Train a TrOCR model on your dataset:

```bash
python src/train.py \
    --model trocr \
    --train-data data/processed/train \
    --val-data data/processed/val \
    --output models/my_trocr_model \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 1e-4
```

### 2. Evaluating a Model

Evaluate a trained model on test data:

```bash
python src/evaluate.py \
    --model trocr \
    --test-data data/processed/test \
    --model-path models/my_trocr_model \
    --output-dir results/evaluation \
    --save-predictions
```

### 3. Inference

#### Single Image
```bash
python src/infer.py \
    --model trocr \
    --image imgs/date.png \
    --model-path models/my_trocr_model
```

#### Batch Inference
```bash
python src/infer.py \
    --model trocr \
    --input-dir path/to/images/ \
    --model-path models/my_trocr_model \
    --output-file predictions.csv
```

## Detailed Usage

### Training (`train.py`)

The training script supports the following arguments:

**Required:**
- `--model`: Model type (`trocr`)
- `--train-data`: Path to training dataset directory
- `--val-data`: Path to validation dataset directory
- `--output`: Output directory for trained model

**Optional:**
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 8)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--save-steps`: Save checkpoint every N steps (default: 1000)
- `--eval-steps`: Evaluate every N steps (default: 1000)
- `--logging-steps`: Log every N steps (default: 50)
- `--base-model`: Base model name/path

**Dataset Structure:**
Your training and validation directories should contain:
- `labels.csv`: CSV file with columns `filename,text`
- Image files referenced in the CSV

### Evaluation (`evaluate.py`)

**Required:**
- `--model`: Model type (`trocr`)
- `--test-data`: Path to test dataset directory

**Optional:**
- `--model-path`: Path to trained model (uses base model if not provided)
- `--output-dir`: Output directory for results (default: `results`)
- `--save-predictions`: Save detailed predictions to CSV
- `--batch-size`: Batch size for evaluation (default: 1)

**Output Files:**
- `metrics.json`: Summary metrics (CER, accuracy, etc.)
- `predictions.csv`: Detailed predictions (if `--save-predictions`)
- `evaluation_report.txt`: Human-readable report
- `evaluation_config.json`: Configuration used

### Inference (`infer.py`)

**Required:**
- `--model`: Model type (`trocr`)
- `--image` OR `--input-dir`: Single image or directory of images

**Optional:**
- `--model-path`: Path to trained model
- `--output-file`: CSV file for batch results
- `--image-extensions`: File extensions to process (default: png, jpg, jpeg, bmp, tiff)
- `--recursive`: Search subdirectories recursively

### Utilities (`utils.py`)

The utilities module provides helper functions for:

- **Data Processing**: COCO annotation loading, dataset creation
- **Visualization**: Prediction visualization, statistics plotting
- **Dataset Management**: Splitting, validation, statistics calculation
- **Image Processing**: Resizing, cropping

## Data Format

### Dataset Structure
```
dataset/
├── labels.csv           # filename,text
├── image1.png
├── image2.png
└── ...
```

### Labels CSV Format
```csv
image1.png,Hello World
image2.png,This is text
image3.png,OCR example
```

## Model Architecture

### TrOCR
- **Base Model**: `microsoft/trocr-base-printed`
- **Architecture**: Vision Encoder + Text Decoder
- **Input**: RGB images (auto-resized)
- **Output**: Text strings

### Adding New Models

To add a new model:

1. Create a new class inheriting from `BaseOCRModel` in `models/model.py`
2. Implement required methods: `load_model`, `train`, `evaluate`, `infer`
3. Add to `MODEL_REGISTRY` in `models/model.py`
4. Update documentation

Example:
```python
class MyOCRModel(BaseOCRModel):
    def load_model(self, model_path=None):
        # Implementation
        pass
    
    def train(self, train_dataset_path, eval_dataset_path, output_dir, **kwargs):
        # Implementation
        pass
    
    def evaluate(self, test_dataset_path, model_path=None):
        # Implementation
        pass
    
    def infer(self, image_path, model_path=None):
        # Implementation
        pass

# Add to registry
MODEL_REGISTRY['mymodel'] = MyOCRModel
```

## Performance Metrics

The evaluation script calculates:
- **Character Error Rate (CER)**: Lower is better
- **Character Accuracy**: Higher is better (1 - CER)
- **Exact Match Rate**: Percentage of perfect predictions
- **Edit Distance Statistics**: Levenshtein distance metrics

## Example Workflows

### Complete Training Pipeline
```bash
# 1. Train model
python src/train.py \
    --model trocr \
    --train-data data/processed/train \
    --val-data data/processed/val \
    --output models/trocr_custom \
    --epochs 10

# 2. Evaluate model
python src/evaluate.py \
    --model trocr \
    --test-data data/processed/test \
    --model-path models/trocr_custom \
    --output-dir results/trocr_custom \
    --save-predictions

# 3. Run inference
python src/infer.py \
    --model trocr \
    --input-dir new_images/ \
    --model-path models/trocr_custom \
    --output-file new_predictions.csv
```

### Data Preparation Example
```python
from src.utils import create_cropped_dataset, print_dataset_statistics

# Create dataset from COCO annotations
# (See notebooks for full COCO processing pipeline)

# Validate dataset
print_dataset_statistics('data/processed/train/labels.csv')
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size
2. **Missing Labels**: Ensure CSV format is correct
3. **Model Loading Issues**: Check model path exists
4. **Import Errors**: Ensure all dependencies installed

### GPU Usage
- Scripts automatically detect and use GPU if available
- For CPU-only usage, the scripts will fall back automatically
- Monitor GPU memory usage with `nvidia-smi`

### Dependencies
Install all requirements:
```bash
pip install -r requirements.txt
```

## Contributing

When adding new models or features:
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Update this README
4. Test thoroughly on sample data

## Future Enhancements

Planned additions:
- Support for PaddleOCR
- Support for EasyOCR
- Multi-language training
- Advanced data augmentation
- Model ensemble support 