# Project Architecture and Code Structure

## Overview

This document explains how different components of the project work together.

## Component Interaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  (CLI Commands or Streamlit Demo)                           │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                     Main Scripts                             │
│  • training/train.py                                         │
│  • evaluation/evaluate.py                                    │
│  • demo/app.py                                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┬──────────┬──────────┐
         ▼                   ▼          ▼          ▼
    ┌────────┐         ┌─────────┐  ┌────────┐  ┌─────────┐
    │ Models │         │  Utils  │  │Attacks │  │ Defense │
    └────────┘         └─────────┘  └────────┘  └─────────┘
         │                   │          │          │
         └───────────────────┴──────────┴──────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   CIFAR-10      │
                    │   Dataset       │
                    └─────────────────┘
```

## Module Descriptions

### 1. Models (`models/`)

**Purpose**: Define neural network architectures

**Key File**: `cnn_model.py`
- `CIFAR10CNN`: Main CNN class
- `get_model()`: Factory function to create models

**Usage**:
```python
from models.cnn_model import get_model
model = get_model(device='cpu')
```

**Architecture Details**:
- Input: 3x32x32 RGB images
- Conv Block 1: 3→32 channels
- Conv Block 2: 32→64 channels  
- Conv Block 3: 64→128 channels
- FC Layers: 2048→512→10
- Output: 10 class logits

### 2. Utils (`utils/`)

**Purpose**: Helper functions for data handling

**Key File**: `data_loader.py`
- `get_cifar10_dataloaders()`: Load and preprocess CIFAR-10
- `denormalize()`: Convert tensors back for visualization
- `CIFAR10_CLASSES`: List of class names

**Data Pipeline**:
1. Download CIFAR-10 (if not present)
2. Apply transformations (crop, flip, normalize)
3. Create DataLoader with batching
4. Return train/test loaders

### 3. Attacks (`attacks/`)

**Purpose**: Implement adversarial attack methods

**Files**:
- `fgsm.py`: Fast Gradient Sign Method
- `pgd.py`: Projected Gradient Descent

**How Attacks Work**:

**FGSM** (Single Step):
```
1. Forward pass: output = model(image)
2. Compute loss: loss = criterion(output, true_label)
3. Backward pass: gradient = ∂loss/∂image
4. Create perturbation: δ = ε × sign(gradient)
5. Generate adversarial: image_adv = image + δ
6. Clip to valid range
```

**PGD** (Iterative):
```
For i in 1 to num_iterations:
  1. Forward pass: output = model(image_adv)
  2. Compute loss: loss = criterion(output, true_label)
  3. Backward pass: gradient = ∂loss/∂image_adv
  4. Update: image_adv = image_adv + α × sign(gradient)
  5. Project back: clip(image_adv - image, -ε, ε)
  6. Clip to valid range
```

### 4. Defense (`defense/`)

**Purpose**: Implement defense mechanisms

**Key File**: `adversarial_training.py`
- `adversarial_training_epoch()`: Train on adversarial examples
- `evaluate_robustness()`: Test against multiple attacks
- `mixed_training_epoch()`: Train on mix of clean and adversarial

**Adversarial Training Process**:
```
For each batch:
  1. Load clean images
  2. Generate adversarial examples (FGSM or PGD)
  3. Train model on adversarial examples
  4. Update model weights
```

### 5. Training (`training/`)

**Purpose**: Training scripts and loops

**Key File**: `train.py`
- `train_epoch()`: Train for one epoch
- `evaluate()`: Evaluate on test set
- `train_model()`: Complete training pipeline
- `plot_training_history()`: Visualize training progress

**Training Flow**:
```
1. Load data (CIFAR-10)
2. Create model
3. Setup optimizer and scheduler
4. For each epoch:
   a. Train on training set
   b. Evaluate on test set
   c. Update learning rate
   d. Save best model
5. Plot training curves
```

### 6. Evaluation (`evaluation/`)

**Purpose**: Evaluate and visualize model robustness

**Key File**: `evaluate.py`
- `evaluate_model_comprehensive()`: Test across epsilon values
- `plot_robustness_curve()`: Visualize accuracy vs epsilon
- `visualize_adversarial_examples()`: Show attack examples
- `compare_models()`: Standard vs robust model comparison

**Evaluation Metrics**:
- Clean Accuracy: Performance on original images
- FGSM Accuracy: Performance under FGSM attack
- PGD Accuracy: Performance under PGD attack
- Attack Success Rate: % of successful attacks

### 7. Demo (`demo/`)

**Purpose**: Interactive web interface

**Key File**: `app.py`
- Streamlit-based web application
- Real-time attack generation
- Visual comparison of clean vs adversarial images
- Model selection and parameter tuning

**Demo Features**:
- Select images from test set
- Choose attack type (FGSM/PGD)
- Adjust attack strength
- Compare standard and robust models
- See predictions and confidence scores

## Data Flow Examples

### Example 1: Standard Training

```
1. User runs: python training/train.py --epochs 50
2. train.py calls get_cifar10_dataloaders()
3. utils/data_loader.py downloads and loads CIFAR-10
4. train.py creates model with get_model()
5. For each epoch:
   - train_epoch() processes all training batches
   - evaluate() tests on validation set
6. Best model saved to checkpoints/
7. Training plots generated
```

### Example 2: Adversarial Training

```
1. User runs: python training/train.py --epochs 50 --adversarial
2. train.py enables adversarial mode
3. For each training batch:
   - Generate adversarial examples with fgsm_attack()
   - Train model on adversarial images
   - Update weights
4. Model learns to handle adversarial perturbations
5. Robust model saved to checkpoints/
```

### Example 3: Evaluation

```
1. User runs: python evaluation/evaluate.py --model-path checkpoints/model.pth
2. evaluate.py loads trained model
3. For each epsilon value:
   - Test clean accuracy
   - Generate FGSM attacks with fgsm_attack()
   - Generate PGD attacks with pgd_attack()
   - Measure accuracy on adversarial examples
4. Generate visualization plots
5. Save results
```

### Example 4: Interactive Demo

```
1. User runs: streamlit run demo/app.py
2. Browser opens to localhost:8501
3. User selects image and attack parameters
4. app.py generates adversarial example in real-time
5. Displays:
   - Original image with prediction
   - Adversarial image with prediction
   - Confidence scores
   - Attack success indicator
```

## Code Dependencies

### Import Hierarchy

```
demo/app.py
├── models.cnn_model
├── utils.data_loader
├── attacks.fgsm
└── attacks.pgd

evaluation/evaluate.py
├── models.cnn_model
├── utils.data_loader
├── attacks.fgsm
└── attacks.pgd

training/train.py
├── models.cnn_model
├── utils.data_loader
└── attacks.fgsm (if --adversarial)

defense/adversarial_training.py
├── attacks.fgsm
└── attacks.pgd
```

## Key Design Decisions

### 1. Modularity
- Each component is independent and reusable
- Clear separation of concerns
- Easy to add new attacks or defenses

### 2. Device Agnostic
- All code works on both CPU and GPU
- Automatic device detection
- No hard-coded device assumptions

### 3. Beginner Friendly
- Extensive comments explaining concepts
- Multiple documentation levels
- Example scripts for learning
- Clear error messages

### 4. Extensibility
- Easy to add new model architectures
- Simple to implement new attacks
- Flexible defense mechanisms
- Pluggable components

## Common Workflows

### Workflow 1: Train and Evaluate Standard Model
```bash
# Train
python training/train.py --epochs 50

# Evaluate
python evaluation/evaluate.py \
  --model-path checkpoints/best_model_standard.pth \
  --visualize
```

### Workflow 2: Train Robust Model and Compare
```bash
# Train standard
python training/train.py --epochs 50

# Train robust
python training/train.py --epochs 50 --adversarial

# Compare both
python evaluation/evaluate.py \
  --model-path checkpoints/best_model_standard.pth

python evaluation/evaluate.py \
  --model-path checkpoints/best_model_adversarial.pth
```

### Workflow 3: Quick Testing
```bash
# Test components
python example_workflow.py

# Quick train (5 epochs)
python training/train.py --epochs 5

# Demo
streamlit run demo/app.py
```

## Performance Considerations

### Memory Usage
- Batch size affects GPU memory
- Default: 128 (adjust for your GPU)
- Reduce if out-of-memory errors

### Training Speed
- GPU: ~10 minutes for 50 epochs
- CPU: ~2 hours for 50 epochs
- Adversarial training: ~2x slower

### Disk Space
- CIFAR-10 dataset: ~170 MB
- Model checkpoints: ~5 MB each
- Training plots: ~1 MB each

## Extending the Project

### Adding a New Attack
1. Create file in `attacks/` (e.g., `cw_attack.py`)
2. Implement attack function with same interface
3. Import in evaluation and demo scripts
4. Add documentation

### Adding a New Defense
1. Create file in `defense/` (e.g., `input_transformation.py`)
2. Implement defense mechanism
3. Integrate with training loop
4. Test effectiveness

### Supporting New Dataset
1. Modify `utils/data_loader.py`
2. Update model input/output sizes
3. Adjust normalization values
4. Update class names

## Troubleshooting Guide

### Issue: Import Errors
**Solution**: Ensure you're in the project root directory

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size (`--batch-size 64`)

### Issue: Slow Training
**Solution**: Use GPU or reduce epochs for testing

### Issue: Model Not Loading
**Solution**: Check path and ensure model architecture matches

## Summary

This project provides a complete, modular framework for studying adversarial robustness. Each component is designed to work independently while integrating seamlessly with others. The clear structure makes it easy to understand, extend, and experiment with different approaches to adversarial machine learning.
