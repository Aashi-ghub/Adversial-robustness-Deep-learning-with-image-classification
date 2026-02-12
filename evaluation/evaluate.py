"""
Evaluation Script with Visualization

This script evaluates trained models on clean and adversarial examples,
and generates visualizations to compare robustness.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import get_model
from utils.data_loader import get_cifar10_dataloaders, denormalize, CIFAR10_CLASSES
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def evaluate_model_comprehensive(model, test_loader, device, epsilon_values=[0.0, 0.01, 0.03, 0.05, 0.1]):
    """
    Comprehensive evaluation of model robustness across different epsilon values.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        epsilon_values (list): List of epsilon values to test
    
    Returns:
        dict: Results for each attack type and epsilon value
    """
    model.eval()
    
    results = {
        'epsilon': epsilon_values,
        'clean': [],
        'fgsm': [],
        'pgd': []
    }
    
    for epsilon in epsilon_values:
        print(f"\n{'='*60}")
        print(f"Evaluating with epsilon = {epsilon}")
        print(f"{'='*60}")
        
        clean_correct = 0
        fgsm_correct = 0
        pgd_correct = 0
        total = 0
        
        for images, labels in tqdm(test_loader, desc=f"ε={epsilon}"):
            images, labels = images.to(device), labels.to(device)
            
            # Clean accuracy
            if epsilon == 0.0:
                with torch.no_grad():
                    outputs = model(images)
                    _, pred = outputs.max(1)
                    clean_correct += pred.eq(labels).sum().item()
            
            # FGSM attack
            if epsilon > 0:
                fgsm_images = fgsm_attack(model, images, labels, epsilon=epsilon)
                with torch.no_grad():
                    outputs = model(fgsm_images)
                    _, pred = outputs.max(1)
                    fgsm_correct += pred.eq(labels).sum().item()
            
            # PGD attack
            if epsilon > 0:
                pgd_images = pgd_attack(model, images, labels, epsilon=epsilon)
                with torch.no_grad():
                    outputs = model(pgd_images)
                    _, pred = outputs.max(1)
                    pgd_correct += pred.eq(labels).sum().item()
            
            total += labels.size(0)
        
        # Store results
        if epsilon == 0.0:
            clean_acc = 100. * clean_correct / total
            results['clean'].append(clean_acc)
            results['fgsm'].append(clean_acc)
            results['pgd'].append(clean_acc)
            print(f"Clean Accuracy: {clean_acc:.2f}%")
        else:
            fgsm_acc = 100. * fgsm_correct / total
            pgd_acc = 100. * pgd_correct / total
            results['clean'].append(results['clean'][0])  # Same as epsilon=0
            results['fgsm'].append(fgsm_acc)
            results['pgd'].append(pgd_acc)
            print(f"FGSM Accuracy: {fgsm_acc:.2f}%")
            print(f"PGD Accuracy: {pgd_acc:.2f}%")
    
    return results


def plot_robustness_curve(results, save_path='robustness_curve.png', title='Model Robustness'):
    """
    Plot robustness curve showing accuracy vs epsilon.
    
    Args:
        results (dict): Results from evaluate_model_comprehensive
        save_path (str): Path to save the plot
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['epsilon'], results['clean'], 'g-o', label='Clean', linewidth=2, markersize=8)
    plt.plot(results['epsilon'], results['fgsm'], 'b-s', label='FGSM Attack', linewidth=2, markersize=8)
    plt.plot(results['epsilon'], results['pgd'], 'r-^', label='PGD Attack', linewidth=2, markersize=8)
    
    plt.xlabel('Epsilon (Attack Strength)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nRobustness curve saved to {save_path}")
    plt.close()


def visualize_adversarial_examples(model, test_loader, device, epsilon=0.03, num_examples=5, save_path='adversarial_examples.png'):
    """
    Visualize clean images alongside their adversarial counterparts.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to use
        epsilon (float): Attack strength
        num_examples (int): Number of examples to visualize
        save_path (str): Path to save the plot
    """
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images, labels = images[:num_examples].to(device), labels[:num_examples].to(device)
    
    # Generate adversarial examples
    fgsm_images = fgsm_attack(model, images, labels, epsilon=epsilon)
    pgd_images = pgd_attack(model, images, labels, epsilon=epsilon)
    
    # Get predictions
    with torch.no_grad():
        clean_outputs = model(images)
        fgsm_outputs = model(fgsm_images)
        pgd_outputs = model(pgd_images)
        
        _, clean_preds = clean_outputs.max(1)
        _, fgsm_preds = fgsm_outputs.max(1)
        _, pgd_preds = pgd_outputs.max(1)
    
    # Denormalize for visualization
    images_vis = denormalize(images.cpu()).numpy()
    fgsm_vis = denormalize(fgsm_images.cpu()).numpy()
    pgd_vis = denormalize(pgd_images.cpu()).numpy()
    
    # Clip to valid range [0, 1]
    images_vis = np.clip(images_vis, 0, 1)
    fgsm_vis = np.clip(fgsm_vis, 0, 1)
    pgd_vis = np.clip(pgd_vis, 0, 1)
    
    # Plot
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
    
    for i in range(num_examples):
        # Clean image
        axes[i, 0].imshow(np.transpose(images_vis[i], (1, 2, 0)))
        axes[i, 0].set_title(f'Clean\nTrue: {CIFAR10_CLASSES[labels[i]]}\nPred: {CIFAR10_CLASSES[clean_preds[i]]}', fontsize=10)
        axes[i, 0].axis('off')
        
        # FGSM adversarial
        axes[i, 1].imshow(np.transpose(fgsm_vis[i], (1, 2, 0)))
        color = 'green' if fgsm_preds[i] == labels[i] else 'red'
        axes[i, 1].set_title(f'FGSM (ε={epsilon})\nTrue: {CIFAR10_CLASSES[labels[i]]}\nPred: {CIFAR10_CLASSES[fgsm_preds[i]]}', 
                            fontsize=10, color=color)
        axes[i, 1].axis('off')
        
        # PGD adversarial
        axes[i, 2].imshow(np.transpose(pgd_vis[i], (1, 2, 0)))
        color = 'green' if pgd_preds[i] == labels[i] else 'red'
        axes[i, 2].set_title(f'PGD (ε={epsilon})\nTrue: {CIFAR10_CLASSES[labels[i]]}\nPred: {CIFAR10_CLASSES[pgd_preds[i]]}', 
                            fontsize=10, color=color)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Adversarial examples visualization saved to {save_path}")
    plt.close()


def compare_models(standard_model_path, robust_model_path, test_loader, device, epsilon_values=[0.0, 0.01, 0.03, 0.05, 0.1]):
    """
    Compare standard and adversarially trained models.
    
    Args:
        standard_model_path (str): Path to standard trained model
        robust_model_path (str): Path to adversarially trained model
        test_loader: DataLoader for test data
        device: Device to use
        epsilon_values (list): Epsilon values to test
    """
    # Load models
    print("Loading standard model...")
    standard_model = get_model(device=device)
    checkpoint = torch.load(standard_model_path, map_location=device)
    standard_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Loading robust model...")
    robust_model = get_model(device=device)
    checkpoint = torch.load(robust_model_path, map_location=device)
    robust_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate both models
    print("\n" + "="*60)
    print("Evaluating Standard Model")
    print("="*60)
    standard_results = evaluate_model_comprehensive(standard_model, test_loader, device, epsilon_values)
    
    print("\n" + "="*60)
    print("Evaluating Robust Model")
    print("="*60)
    robust_results = evaluate_model_comprehensive(robust_model, test_loader, device, epsilon_values)
    
    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Standard model
    ax1.plot(standard_results['epsilon'], standard_results['clean'], 'g-o', label='Clean', linewidth=2)
    ax1.plot(standard_results['epsilon'], standard_results['fgsm'], 'b-s', label='FGSM', linewidth=2)
    ax1.plot(standard_results['epsilon'], standard_results['pgd'], 'r-^', label='PGD', linewidth=2)
    ax1.set_xlabel('Epsilon', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Standard Training', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Robust model
    ax2.plot(robust_results['epsilon'], robust_results['clean'], 'g-o', label='Clean', linewidth=2)
    ax2.plot(robust_results['epsilon'], robust_results['fgsm'], 'b-s', label='FGSM', linewidth=2)
    ax2.plot(robust_results['epsilon'], robust_results['pgd'], 'r-^', label='PGD', linewidth=2)
    ax2.set_xlabel('Epsilon', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Adversarial Training', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nModel comparison plot saved to model_comparison.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model robustness')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    _, test_loader, _, _ = get_cifar10_dataloaders(batch_size=args.batch_size, num_workers=2)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = get_model(device=device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded (trained for {checkpoint['epoch']+1} epochs)")
    
    # Evaluate
    results = evaluate_model_comprehensive(model, test_loader, device)
    
    # Plot results
    plot_robustness_curve(results)
    
    # Visualize adversarial examples if requested
    if args.visualize:
        visualize_adversarial_examples(model, test_loader, device)
