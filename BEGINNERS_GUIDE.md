# Beginner's Guide to Adversarial Robustness

## What are Adversarial Attacks?

Adversarial attacks are small, carefully crafted changes to input images that are nearly invisible to humans but can fool neural networks into making wrong predictions.

### Example:
- **Original Image**: A photo of a cat → Model predicts: "Cat" ✓
- **Adversarial Image**: Same photo with tiny perturbations → Model predicts: "Dog" ✗
- **To Human Eyes**: Both images look identical!

## Understanding the Code

### 1. Model Architecture (`models/cnn_model.py`)

The CNN model has:
- **3 Convolutional Layers**: Extract features from images
- **Batch Normalization**: Stabilizes training
- **Max Pooling**: Reduces spatial dimensions
- **Fully Connected Layers**: Make final classification

```python
# Simple usage:
from models.cnn_model import get_model
model = get_model()
```

### 2. Data Loading (`utils/data_loader.py`)

CIFAR-10 dataset contains:
- 60,000 images (50,000 training, 10,000 test)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Image size: 32×32 pixels, RGB

```python
# Load data:
from utils.data_loader import get_cifar10_dataloaders
train_loader, test_loader, _, _ = get_cifar10_dataloaders()
```

### 3. FGSM Attack (`attacks/fgsm.py`)

**How it works:**
1. Feed image to model → get prediction
2. Calculate loss (how wrong is the prediction)
3. Calculate gradient (how to change image to increase loss)
4. Add small perturbation in gradient direction

**Formula**: `adversarial_image = clean_image + ε × sign(gradient)`

**Epsilon (ε)**: Controls attack strength
- ε = 0.01: Weak attack, barely visible
- ε = 0.03: Medium attack (common choice)
- ε = 0.1: Strong attack, may be visible

```python
# Usage:
from attacks.fgsm import fgsm_attack
adv_images = fgsm_attack(model, images, labels, epsilon=0.03)
```

### 4. PGD Attack (`attacks/pgd.py`)

**More powerful than FGSM:**
- Takes multiple small steps instead of one big step
- Projects back to valid range after each step
- More effective at fooling models

**Think of it as:**
- FGSM: Jump in one direction
- PGD: Take many small steps in the right direction

```python
# Usage:
from attacks.pgd import pgd_attack
adv_images = pgd_attack(model, images, labels, epsilon=0.03, num_iter=10)
```

### 5. Adversarial Training (`defense/adversarial_training.py`)

**The Defense:**
- Instead of training only on clean images, also train on adversarial examples
- Model learns to recognize both clean and perturbed images
- Significantly improves robustness

**Trade-off:**
- ✓ Better defense against attacks
- ✗ Slightly lower accuracy on clean images
- ⏱️ Takes longer to train

## Step-by-Step Tutorial

### Step 1: Install Dependencies
```bash
pip install torch torchvision numpy matplotlib streamlit
```

### Step 2: Test Components
```bash
# Test the model
python models/cnn_model.py

# Test attacks (demonstration)
python example_workflow.py
```

### Step 3: Train a Standard Model
```bash
# Quick test (5 epochs, ~5 minutes on CPU)
python training/train.py --epochs 5

# Full training (50 epochs, ~2 hours on CPU, 10 min on GPU)
python training/train.py --epochs 50
```

**What happens during training:**
1. Load CIFAR-10 dataset
2. For each epoch:
   - Show model all training images
   - Adjust weights to reduce errors
   - Test on validation set
3. Save best model to `checkpoints/`

### Step 4: Train a Robust Model
```bash
# With adversarial training
python training/train.py --epochs 50 --adversarial
```

**Difference:**
- Standard: Trains on clean images only
- Adversarial: Trains on mixture of clean + adversarial images

### Step 5: Evaluate Models
```bash
# Evaluate standard model
python evaluation/evaluate.py --model-path checkpoints/best_model_standard.pth --visualize

# Evaluate robust model
python evaluation/evaluate.py --model-path checkpoints/best_model_adversarial.pth --visualize
```

**Output:**
- Accuracy on clean images
- Accuracy under FGSM attack
- Accuracy under PGD attack
- Visualization plots

### Step 6: Interactive Demo
```bash
streamlit run demo/app.py
```

**In the demo you can:**
- Select different images
- Choose attack type (FGSM/PGD)
- Adjust attack strength
- See predictions change in real-time
- Compare standard vs robust models

## Key Concepts Explained

### Epsilon (ε)
- Maximum change allowed per pixel
- Larger ε = stronger attack, more visible
- Typically use 0.01 to 0.1 for normalized images

### L∞ Norm (L-infinity)
- Measures maximum pixel change
- Ensures all pixels change by at most ε
- Keeps perturbations small

### Gradient
- Shows how to change input to change output
- Points in direction of steepest increase
- Used by attacks to maximize error

### Clean Accuracy vs Robust Accuracy
- **Clean Accuracy**: Performance on normal images
- **Robust Accuracy**: Performance under attack
- Goal: High accuracy on both!

## Expected Results

### Standard Training
```
Clean Accuracy:     75-80%  ✓ Good
FGSM Accuracy:      20-30%  ✗ Poor
PGD Accuracy:       10-20%  ✗ Very Poor
```

### Adversarial Training
```
Clean Accuracy:     70-75%  ✓ Decent
FGSM Accuracy:      50-60%  ✓ Much better!
PGD Accuracy:       45-55%  ✓ Much better!
```

## Common Questions

**Q: Why do adversarial examples work?**
A: Neural networks learn patterns but not true understanding. Small changes can exploit these patterns.

**Q: Can humans see adversarial perturbations?**
A: Usually no - they're designed to be imperceptible (< 3% pixel change).

**Q: Is adversarial training the best defense?**
A: It's one of the most effective, but perfect defense is still an open problem.

**Q: Why does robustness matter?**
A: Important for security-critical applications (autonomous vehicles, medical diagnosis, authentication).

**Q: Can I use this on other datasets?**
A: Yes! You'd need to modify the data loader and possibly the model architecture.

## Troubleshooting

### "Out of memory" error
```bash
# Reduce batch size
python training/train.py --batch-size 64
```

### Training is slow
```bash
# Use GPU if available (30x faster)
# Check: torch.cuda.is_available()

# Or reduce epochs for testing
python training/train.py --epochs 5
```

### Model not loading
```bash
# Make sure checkpoint exists
ls checkpoints/

# Check path is correct
python evaluation/evaluate.py --model-path checkpoints/best_model_standard.pth
```

## Further Learning

### Papers to Read
1. **FGSM**: "Explaining and Harnessing Adversarial Examples" (Goodfellow et al., 2015)
2. **PGD**: "Towards Deep Learning Models Resistant to Adversarial Attacks" (Madry et al., 2018)

### Next Steps
- Try different model architectures (ResNet, VGG)
- Implement other attacks (C&W, DeepFool)
- Test on other datasets (ImageNet, MNIST)
- Explore certified defenses

## Tips for Success

1. **Start Small**: Train for 5 epochs first to ensure everything works
2. **Use GPU**: Training on GPU is 30x faster
3. **Save Checkpoints**: Don't lose your trained models
4. **Visualize**: Use the demo to understand what's happening
5. **Experiment**: Try different epsilon values and attack parameters

## Getting Help

If you encounter issues:
1. Check the error message carefully
2. Verify installation of dependencies
3. Ensure correct file paths
4. Try the example workflow script first
5. Open an issue on GitHub with details

Happy learning! 🎓🛡️
