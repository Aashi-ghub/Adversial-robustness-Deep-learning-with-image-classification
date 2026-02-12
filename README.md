# Adversarial Robustness in Deep Learning for Image Classification

A comprehensive implementation of adversarial robustness techniques for deep learning image classification models. This project demonstrates how to train robust models, generate adversarial examples, and evaluate model robustness against various attacks.

## 🎯 Features

- **Multiple Attack Methods**: FGSM and PGD adversarial attacks
- **Adversarial Training**: Defense mechanism to improve model robustness
- **Comprehensive Evaluation**: Test models against various attack strengths
- **Visualization Demo**: See adversarial examples and their effects
- **Multiple Datasets**: Support for MNIST, CIFAR-10, and CIFAR-100

## 📁 Project Structure

```
adversarial-robustness/
│
├── models/
│   └── cnn.py                 # CNN model architecture
│
├── training/
│   ├── data_loader.py         # Data loading utilities
│   └── train.py               # Standard training script
│
├── attacks/
│   ├── fgsm.py               # Fast Gradient Sign Method attack
│   └── pgd.py                # Projected Gradient Descent attack
│
├── defense/
│   └── adv_training.py       # Adversarial training defense
│
├── evaluation/
│   └── robustness_eval.py    # Robustness evaluation metrics
│
├── demo/
│   └── app.py                # Demo and visualization application
│
├── checkpoints/              # Model checkpoints directory
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

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

### Usage

#### 1. Standard Training

Train a model using standard (clean) training:

```bash
python main.py --mode train --dataset cifar10 --epochs 10 --lr 0.01
```

#### 2. Adversarial Training

Train a robust model using adversarial training:

```bash
python main.py --mode adv-train --dataset cifar10 --epochs 10 --attack pgd --epsilon 0.3
```

#### 3. Evaluate Model Robustness

Evaluate a trained model against adversarial attacks:

```bash
python main.py --mode evaluate --load-path checkpoints/model.pth --dataset cifar10
```

#### 4. Run Demo

Generate and visualize adversarial examples:

```bash
python main.py --mode demo --load-path checkpoints/model.pth --dataset cifar10
```

## 📊 Command Line Arguments

| Argument | Description | Default | Choices |
|----------|-------------|---------|---------|
| `--mode` | Operating mode | `train` | `train`, `adv-train`, `evaluate`, `demo` |
| `--dataset` | Dataset to use | `cifar10` | `mnist`, `cifar10`, `cifar100` |
| `--epochs` | Number of training epochs | `10` | - |
| `--batch-size` | Batch size for training | `128` | - |
| `--lr` | Learning rate | `0.01` | - |
| `--attack` | Attack method for adversarial training | `pgd` | `fgsm`, `pgd` |
| `--epsilon` | Perturbation magnitude | `0.3` | - |
| `--save-path` | Path to save model | `checkpoints/model.pth` | - |
| `--load-path` | Path to load pretrained model | `None` | - |
| `--device` | Device to use | `cuda` | `cuda`, `cpu` |

## 🔬 Attack Methods

### FGSM (Fast Gradient Sign Method)
A simple one-step attack that perturbs the input in the direction of the gradient:
```
x_adv = x + ε * sign(∇_x L(θ, x, y))
```

### PGD (Projected Gradient Descent)
An iterative version of FGSM that takes multiple small steps:
```
x_adv^(t+1) = Π_{x+S} (x_adv^(t) + α * sign(∇_x L(θ, x_adv^(t), y)))
```

## 🛡️ Defense: Adversarial Training

Adversarial training improves model robustness by training on both clean and adversarial examples:
```
min_θ E_{(x,y)~D} [max_{||δ||≤ε} L(θ, x+δ, y)]
```

## 📈 Evaluation Metrics

The evaluation module provides:
- Clean accuracy (performance on original images)
- Adversarial accuracy (performance under attack)
- Robustness curves across different perturbation magnitudes
- Comparison between different attack methods

## 🎨 Visualization

The demo application generates visualizations showing:
- Original images
- Perturbations added by attacks
- Resulting adversarial examples
- Differences between original and adversarial images
- Model predictions on clean vs adversarial inputs

## 📚 Example Workflow

Complete workflow for training and evaluating a robust model:

```bash
# 1. Train a standard model
python main.py --mode train --dataset cifar10 --epochs 20 --save-path checkpoints/clean_model.pth

# 2. Evaluate standard model's robustness
python main.py --mode evaluate --load-path checkpoints/clean_model.pth --dataset cifar10

# 3. Train an adversarially robust model
python main.py --mode adv-train --dataset cifar10 --epochs 20 --attack pgd --epsilon 0.3 --save-path checkpoints/robust_model.pth

# 4. Evaluate robust model
python main.py --mode evaluate --load-path checkpoints/robust_model.pth --dataset cifar10

# 5. Generate visualizations
python main.py --mode demo --load-path checkpoints/robust_model.pth --dataset cifar10
```

## 🔬 Research Background

### Adversarial Examples
Adversarial examples are inputs intentionally designed to cause machine learning models to make mistakes. Even small, imperceptible perturbations can fool state-of-the-art deep learning models.

### Why This Matters
- **Security**: Adversarial attacks pose security risks in real-world applications
- **Robustness**: Understanding vulnerabilities helps build more reliable models
- **Interpretability**: Studying adversarial examples reveals how models make decisions

## 📖 References

- Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (FGSM)
- Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (PGD)
- Adversarial training and robustness evaluation methodologies

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Add new attack or defense methods
- Improve documentation

## 📝 License

This project is available for educational and research purposes.

## 👤 Author

Created by [Aashi-ghub](https://github.com/Aashi-ghub)

## 🙏 Acknowledgments

This project implements techniques from adversarial robustness research and is intended for educational purposes to understand and improve deep learning security.