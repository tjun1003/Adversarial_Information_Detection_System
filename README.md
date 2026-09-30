# MisRoBAERTa

A hybrid deep learning architecture combining BART and RoBERTa dual-branch encodings with BiLSTM-CNN fusion for text classification.

## Overview

MisRoBAERTa is a sophisticated neural network model that leverages the strengths of two powerful transformer models (BART and RoBERTa) and combines them through a multi-layered BiLSTM-CNN architecture. This dual-branch approach captures both the sequence-to-sequence capabilities of BART and the masked language modeling strengths of RoBERTa, creating rich contextual representations for accurate text classification.

### Model Architecture

```
                    Input Text
                        |
        +---------------+---------------+
        |                               |
    BART Encoder                  RoBERTa Encoder
  (1024-dim embeddings)          (768-dim embeddings)
        |                               |
    3x BiLSTM Layers               3x BiLSTM Layers
    (256 units each)               (256 units each)
        |                               |
    CNN + MaxPool                  CNN + MaxPool
        |                               |
        +---------------+---------------+
                        |
                  Concatenation
                        |
                  3x BiLSTM Layers
                        |
                  CNN + MaxPool
                        |
                   Softmax Output
```

**Key Components:**
- **Dual Encoders**: BART (facebook/bart-large) and RoBERTa (roberta-base) generate complementary sentence embeddings
- **BiLSTM Layers**: Three stacked bidirectional LSTM layers per branch capture sequential dependencies
- **CNN Layers**: Convolutional layers with max pooling extract local patterns and reduce dimensionality
- **Fusion Architecture**: Combined features pass through additional BiLSTM-CNN layers for final classification

## Project Structure

```
misroberta-project/
├── misroberta/
│   ├── __init__.py
│   └── utils.py              # Pure logic functions (data splitting, metrics)
│                              # No heavy dependencies - optimized for CI testing
├── tests/
│   └── test_utils.py         # Unit tests for utility functions
├── train.py                  # Training entry point with model definition
│                              # Contains BART/RoBERTa encoding logic
├── requirements.txt          # Full training dependencies
│                              # (keras, torch, simpletransformers, etc.)
├── requirements-dev.txt      # Lightweight CI dependencies
│                              # (numpy, pandas, sklearn, pytest, ruff)
├── Dockerfile                # Training environment container
├── saved_models/             # Model artifacts, configs, and metrics
└── .github/workflows/
    ├── ci.yml                # Fast CI: linting + unit tests
    └── cd.yml                # Heavy CD: Docker build + training + artifacts
```

## Why Separate CI and CD?

Training deep learning models with transformers is computationally expensive and time-consuming:

- **Heavy Dependencies**: keras, torch, simpletransformers, and sentence-transformers require large downloads (several GB)
- **Model Downloads**: BART-large and RoBERTa-base models need to be fetched from Hugging Face
- **GPU Requirements**: Training requires CUDA-enabled GPU for reasonable performance
- **Long Training Time**: Multiple iterations with early stopping can take hours

Running full training on every code push would:
- Make CI extremely slow (hours instead of minutes)
- Waste computational resources on non-training code changes
- Require expensive GPU runners for all commits

### Our Solution

**CI Pipeline** (Fast - runs on every push/PR):
- Only installs lightweight dependencies (`requirements-dev.txt`)
- Runs unit tests on pure logic functions in `misroberta/utils.py`
- Performs code linting with ruff
- Completes in minutes without GPU

**CD Pipeline** (Heavy - manual or scheduled trigger):
- Builds Docker image with full dependencies
- Runs complete training with GPU on self-hosted runners
- Uploads model artifacts (`.h5`), label mappings, configs, and metrics
- Triggered via `workflow_dispatch` or on a schedule

## Model Configuration

Default hyperparameters in `train.py`:

```python
CONFIG = {
    "epochs_n": 100,              # Maximum epochs (early stopping applies)
    "filters": 64,                # CNN filters
    "units": 128,                 # LSTM units per layer
    "dropout_rate": 0.2,          # LSTM dropout
    "recurrent_dropout_rate": 0.2,
    "batch_size": 1000,
    "patience": 20,               # Early stopping patience
    "test_size": 0.30,            # Train/test split ratio
}
```

## Training Process

The training script performs multiple iterations with the following workflow:

1. **Data Loading**: Reads CSV with `content` and `label` columns
2. **Label Encoding**: Maps text labels to integer IDs
3. **Data Splitting**: Stratified train/test split (70/30 by default)
4. **BART Encoding**: Generates 1024-dim embeddings using `facebook/bart-large`
5. **RoBERTa Encoding**: Generates 768-dim embeddings using `roberta-base`
6. **Model Training**: Dual-branch BiLSTM-CNN with early stopping
7. **Evaluation**: Computes accuracy, precision, recall, confusion matrix
8. **Model Saving**: Exports model, label mapping, config, and metrics with timestamps

### Output Artifacts

After training, the following files are saved to `saved_models/`:

- `model_YYYYMMDD_HHMMSS.h5` - Trained Keras model
- `id2label_YYYYMMDD_HHMMSS.pkl` - Label ID to text mapping
- `config_YYYYMMDD_HHMMSS.json` - Hyperparameters used
- `metrics_YYYYMMDD_HHMMSS.json` - Training metrics per iteration

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (recommended for training)
- Dataset in CSV format with `content` and `label` columns

### Local Setup

**For CI/Development (lightweight):**
```bash
pip install -r requirements-dev.txt
pytest tests/ -v
ruff check .
```

**For Training (full environment):**
```bash
pip install -r requirements.txt
python train.py data/your_dataset.csv 1 3
```

**Command Arguments:**
- `data/your_dataset.csv` - Path to training data
- `1` - Use CUDA (1=yes, 0=no)
- `3` - Number of training iterations

### Docker Training

```bash
# Build the image
docker build -t misroberta-trainer .

# Run training (mount data volume)
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/saved_models:/app/saved_models \
           misroberta-trainer data/dataset.csv 1 3
```

## Metrics and Evaluation

The model reports the following metrics after each iteration:

- **Accuracy**: Overall classification accuracy
- **Precision** (micro & macro): Positive prediction quality
- **Recall** (micro & macro): Actual positive detection rate
- **Classification Report**: Per-class precision, recall, F1-score
- **Confusion Matrix**: Detailed prediction breakdown
- **Execution Time**: Training time per iteration

Average metrics across all iterations are computed at the end.
