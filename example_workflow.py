"""
Example Script: Complete Workflow Demo

This script demonstrates a complete workflow without requiring 
actual training or downloading data. It shows how all components work together.
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.cnn_model import get_model
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def main():
    print("="*60)
    print("Adversarial Robustness Demo - Complete Workflow")
    print("="*60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n1. Device: {device}")
    
    # Create model
    print("\n2. Creating CNN model...")
    model = get_model(device=device)
    model.eval()
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data (simulating CIFAR-10)
    print("\n3. Creating dummy CIFAR-10 images...")
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32).to(device)
    labels = torch.randint(0, 10, (batch_size,)).to(device)
    print(f"   Batch size: {batch_size}")
    print(f"   Image shape: {images.shape}")
    
    # Clean prediction
    print("\n4. Making clean predictions...")
    with torch.no_grad():
        clean_outputs = model(images)
        _, clean_preds = clean_outputs.max(1)
    
    print(f"   True labels: {labels.tolist()}")
    print(f"   Predictions: {clean_preds.tolist()}")
    
    # FGSM Attack
    print("\n5. Generating FGSM adversarial examples...")
    epsilon = 0.03
    fgsm_images = fgsm_attack(model, images, labels, epsilon=epsilon)
    
    with torch.no_grad():
        fgsm_outputs = model(fgsm_images)
        _, fgsm_preds = fgsm_outputs.max(1)
    
    print(f"   Epsilon: {epsilon}")
    print(f"   Predictions: {fgsm_preds.tolist()}")
    
    # Calculate perturbation
    perturbation = (fgsm_images - images).abs().mean().item()
    print(f"   Avg perturbation: {perturbation:.6f}")
    
    # PGD Attack
    print("\n6. Generating PGD adversarial examples...")
    num_iter = 10
    pgd_images = pgd_attack(model, images, labels, epsilon=epsilon, num_iter=num_iter)
    
    with torch.no_grad():
        pgd_outputs = model(pgd_images)
        _, pgd_preds = pgd_outputs.max(1)
    
    print(f"   Epsilon: {epsilon}")
    print(f"   Iterations: {num_iter}")
    print(f"   Predictions: {pgd_preds.tolist()}")
    
    # Compare attacks
    print("\n7. Attack Comparison:")
    print(f"   Original:     {labels.tolist()}")
    print(f"   Clean Pred:   {clean_preds.tolist()}")
    print(f"   FGSM Pred:    {fgsm_preds.tolist()}")
    print(f"   PGD Pred:     {pgd_preds.tolist()}")
    
    # Summary
    print("\n8. Summary:")
    
    # For demonstration, count mismatches (with random model, this is random)
    clean_acc = (clean_preds == labels).sum().item() / batch_size * 100
    fgsm_acc = (fgsm_preds == labels).sum().item() / batch_size * 100
    pgd_acc = (pgd_preds == labels).sum().item() / batch_size * 100
    
    print(f"   Clean Accuracy: {clean_acc:.1f}%")
    print(f"   FGSM Accuracy:  {fgsm_acc:.1f}%")
    print(f"   PGD Accuracy:   {pgd_acc:.1f}%")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    
    print("\nNote: This demo uses a randomly initialized model.")
    print("For realistic results, train a model using:")
    print("  python training/train.py --epochs 50")
    print("\nThen evaluate it using:")
    print("  python evaluation/evaluate.py --model-path checkpoints/best_model_standard.pth")
    print("\nOr launch the interactive demo:")
    print("  streamlit run demo/app.py")


if __name__ == "__main__":
    main()
