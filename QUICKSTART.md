# Quick Start Guide

## First Time Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test the data loader:
```bash
python utils/data_loader.py
```

3. Test the model:
```bash
python models/cnn_model.py
```

## Quick Training (5 epochs for testing)

Train a standard model quickly:
```bash
python training/train.py --epochs 5 --batch-size 128
```

Train an adversarially robust model:
```bash
python training/train.py --epochs 5 --batch-size 128 --adversarial
```

## Demo the Attacks

Test FGSM attack:
```bash
python attacks/fgsm.py
```

Test PGD attack:
```bash
python attacks/pgd.py
```

## Launch the Web Demo

```bash
streamlit run demo/app.py
```

Then open your browser to http://localhost:8501

## Full Training (Recommended)

For best results, train for 50 epochs:

Standard model:
```bash
python training/train.py --epochs 50 --batch-size 128 --lr 0.001
```

Robust model:
```bash
python training/train.py --epochs 50 --batch-size 128 --lr 0.001 --adversarial
```

## Evaluation

After training, evaluate the model:
```bash
python evaluation/evaluate.py --model-path checkpoints/best_model_standard.pth --visualize
```

## Tips

- Training on GPU is much faster (30x speedup)
- Start with 5 epochs to ensure everything works
- Adversarial training takes ~2x longer than standard training
- The demo works even without trained models (uses random weights)
