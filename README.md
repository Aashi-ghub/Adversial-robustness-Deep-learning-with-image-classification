# Adversarial Robustness in Deep Learning for Image Classification

A comprehensive PyTorch project demonstrating adversarial attacks and defenses for image classification on the CIFAR-10 dataset. This project is beginner-friendly with extensive comments and visualizations.

## 🎯 Project Overview

This project explores adversarial robustness in deep learning by:
- Training CNN models on CIFAR-10 dataset
- Implementing adversarial attacks (FGSM and PGD)
- Demonstrating adversarial training as a defense mechanism
- Evaluating model robustness with visualizations
- Providing an interactive Streamlit demo

## 📁 Project Structure

```
.
├── models/                     # Neural network architectures
│   ├── __init__.py
│   └── cnn_model.py           # CNN model for CIFAR-10
├── training/                   # Training scripts
│   └── train.py               # Main training script
├── attacks/                    # Adversarial attack implementations
│   ├── __init__.py
│   ├── fgsm.py                # Fast Gradient Sign Method
│   └── pgd.py                 # Projected Gradient Descent
├── defense/                    # Defense mechanisms
│   ├── __init__.py
│   └── adversarial_training.py # Adversarial training defense
├── evaluation/                 # Evaluation and visualization
│   └── evaluate.py            # Comprehensive evaluation script
├── demo/                       # Interactive demo
│   └── app.py                 # Streamlit web application
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── data_loader.py         # Data loading and preprocessing
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Aashi-ghub/Adversial-robustness-Deep-learning-with-image-classification.git
cd Adversial-robustness-Deep-learning-with-image-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Usage

### 1. Training a Standard Model

Train a CNN model on CIFAR-10 with standard training:

```bash
python training/train.py --epochs 50 --batch-size 128 --lr 0.001
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 128)
- `--lr`: Learning rate (default: 0.001)
- `--save-dir`: Directory to save checkpoints (default: checkpoints)

### 2. Training a Robust Model (Adversarial Training)

Train a model with adversarial training for improved robustness:

```bash
python training/train.py --epochs 50 --adversarial --batch-size 128 --lr 0.001
```

The `--adversarial` flag enables adversarial training, which makes the model more robust to attacks.

### 3. Evaluating Model Robustness

Evaluate a trained model against adversarial attacks:

```bash
python evaluation/evaluate.py --model-path checkpoints/best_model_standard.pth --visualize
```

**Arguments:**
- `--model-path`: Path to the trained model checkpoint
- `--batch-size`: Batch size for evaluation (default: 100)
- `--visualize`: Generate visualization plots

This will:
- Test the model on clean and adversarial examples
- Generate robustness curves showing accuracy vs attack strength
- Create visualizations of adversarial examples (if `--visualize` is used)

### 4. Running the Interactive Demo

Launch the Streamlit web application for interactive exploration:

```bash
streamlit run demo/app.py
```

This opens a web interface where you can:
- Select different models (standard vs robust)
- Choose attack types (FGSM or PGD)
- Adjust attack strength (epsilon)
- Visualize attacks on individual images
- Compare predictions on clean vs adversarial images

## 🔬 Key Concepts

### Adversarial Attacks

**FGSM (Fast Gradient Sign Method)**
- Single-step attack
- Fast to compute
- Formula: `x_adv = x + ε × sign(∇_x L(θ, x, y))`
- Used for: Quick evaluation and adversarial training

**PGD (Projected Gradient Descent)**
- Iterative attack
- Stronger than FGSM
- Takes multiple small steps with projection
- Used for: Robust evaluation

### Defense Mechanism

**Adversarial Training**
- Train the model on adversarial examples
- Most effective defense method
- Trade-off: May slightly reduce clean accuracy
- Significantly improves robustness

### Model Architecture

**CIFAR-10 CNN**
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch normalization and dropout for regularization
- 2 Fully connected layers
- ~500K parameters

## 📊 Expected Results

### Standard Training
- Clean Accuracy: ~75-80%
- FGSM Accuracy (ε=0.03): ~20-30%
- PGD Accuracy (ε=0.03): ~10-20%

### Adversarial Training
- Clean Accuracy: ~70-75%
- FGSM Accuracy (ε=0.03): ~50-60%
- PGD Accuracy (ε=0.03): ~45-55%

*Note: Adversarial training improves robustness significantly while maintaining reasonable clean accuracy.*

## 🎨 Visualizations

The project generates several types of visualizations:

1. **Training History Plots**: Loss and accuracy curves during training
2. **Robustness Curves**: Accuracy vs epsilon for different attacks
3. **Adversarial Examples**: Side-by-side comparison of clean and adversarial images
4. **Model Comparison**: Standard vs adversarially trained models

## 🔧 Customization

### Modifying the Model

Edit `models/cnn_model.py` to change the architecture:
- Adjust number of layers
- Change filter sizes
- Modify dropout rates

### Implementing New Attacks

Add new attack methods in the `attacks/` directory:
1. Create a new Python file (e.g., `custom_attack.py`)
2. Implement the attack function
3. Follow the same interface as FGSM and PGD

### Adding Defense Mechanisms

Implement new defenses in the `defense/` directory:
- Input preprocessing
- Gradient masking
- Defensive distillation

## 📖 Learning Resources

### Papers
- **FGSM**: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
- **PGD**: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)

### Key Concepts
- **Adversarial Examples**: Inputs designed to cause misclassification
- **L∞ Norm**: Maximum change to any pixel (epsilon)
- **Robustness**: Model's ability to maintain accuracy under attack
- **Clean Accuracy**: Performance on unmodified images

## 🐛 Troubleshooting

### Out of Memory Errors
- Reduce batch size: `--batch-size 64`
- Use CPU instead of GPU for smaller datasets

### Slow Training
- Enable GPU if available
- Reduce number of epochs for initial testing
- Use smaller batch size with fewer workers

### Model Not Loading
- Ensure checkpoint file exists in the specified path
- Check that model architecture matches the saved checkpoint

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional attack methods (C&W, DeepFool)
- More defense mechanisms
- Support for other datasets (ImageNet, MNIST)
- Model architectures (ResNet, VGG)
- Performance optimizations

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- CIFAR-10 dataset by Krizhevsky et al.
- PyTorch framework
- Research papers on adversarial robustness
- Open source community

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes to understand adversarial robustness in deep learning. The concepts demonstrated here are crucial for building secure and reliable AI systems.