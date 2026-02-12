"""
Demo Application for Adversarial Robustness Visualization
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import get_model
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def visualize_adversarial_examples(model, images, labels, attack_method='fgsm', 
                                   epsilon=0.3, device='cuda', class_names=None):
    """
    Visualize original and adversarial examples side by side
    
    Args:
        model: Neural network model
        images: Input images (batch)
        labels: True labels
        attack_method: Attack method ('fgsm' or 'pgd')
        epsilon: Perturbation magnitude
        device: Device to run on
        class_names: List of class names for labels
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)
    
    # Generate adversarial examples
    if attack_method == 'fgsm':
        adv_images, perturbations = fgsm_attack(model, images, labels, epsilon, device)
    elif attack_method == 'pgd':
        adv_images, perturbations = pgd_attack(model, images, labels, epsilon, 
                                               alpha=0.01, num_iter=40, device=device)
    else:
        raise ValueError(f"Unknown attack method: {attack_method}")
    
    # Get predictions
    with torch.no_grad():
        clean_outputs = model(images)
        adv_outputs = model(adv_images)
        _, clean_preds = clean_outputs.max(1)
        _, adv_preds = adv_outputs.max(1)
    
    # Convert to numpy for visualization
    images_np = images.cpu().numpy()
    adv_images_np = adv_images.cpu().numpy()
    perturbations_np = perturbations.cpu().numpy()
    
    # Visualize
    num_samples = min(5, len(images))
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = np.transpose(images_np[i], (1, 2, 0))
        if img.shape[2] == 1:  # Grayscale
            axes[i, 0].imshow(img.squeeze(), cmap='gray')
        else:
            axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original\nTrue: {labels[i].item()}\nPred: {clean_preds[i].item()}')
        axes[i, 0].axis('off')
        
        # Perturbation
        pert = np.transpose(perturbations_np[i], (1, 2, 0))
        if pert.shape[2] == 1:
            axes[i, 1].imshow(pert.squeeze(), cmap='seismic', vmin=-epsilon, vmax=epsilon)
        else:
            axes[i, 1].imshow(pert)
        axes[i, 1].set_title(f'Perturbation\n(ε={epsilon})')
        axes[i, 1].axis('off')
        
        # Adversarial image
        adv_img = np.transpose(adv_images_np[i], (1, 2, 0))
        if adv_img.shape[2] == 1:
            axes[i, 2].imshow(adv_img.squeeze(), cmap='gray')
        else:
            axes[i, 2].imshow(adv_img)
        axes[i, 2].set_title(f'Adversarial\nPred: {adv_preds[i].item()}')
        axes[i, 2].axis('off')
        
        # Difference (magnified)
        diff = adv_img - img
        if diff.shape[2] == 1:
            axes[i, 3].imshow(diff.squeeze() * 10, cmap='seismic')
        else:
            axes[i, 3].imshow((diff * 10).clip(0, 1))
        axes[i, 3].set_title('Difference (10x)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('adversarial_examples.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'adversarial_examples.png'")
    plt.close()


def demo_robustness_comparison(model_clean, model_adv, test_loader, device='cuda'):
    """
    Compare robustness of clean-trained vs adversarially-trained models
    
    Args:
        model_clean: Clean-trained model
        model_adv: Adversarially-trained model
        test_loader: Test data loader
        device: Device to run on
    """
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    clean_model_accuracies = []
    adv_model_accuracies = []
    
    print("\nComparing model robustness...")
    
    for eps in epsilons:
        print(f"\nEvaluating at ε={eps}")
        
        # Evaluate both models
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            if eps > 0:
                # Generate adversarial examples
                adv_data, _ = fgsm_attack(model_clean, data, target, eps, device)
            else:
                adv_data = data
            
            # Test clean model
            with torch.no_grad():
                output_clean = model_clean(adv_data)
                _, pred_clean = output_clean.max(1)
                clean_correct += pred_clean.eq(target).sum().item()
                
                # Test adversarial model
                output_adv = model_adv(adv_data)
                _, pred_adv = output_adv.max(1)
                adv_correct += pred_adv.eq(target).sum().item()
            
            total += target.size(0)
        
        clean_acc = 100. * clean_correct / total
        adv_acc = 100. * adv_correct / total
        
        clean_model_accuracies.append(clean_acc)
        adv_model_accuracies.append(adv_acc)
        
        print(f"Clean Model: {clean_acc:.2f}% | Robust Model: {adv_acc:.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, clean_model_accuracies, 'o-', label='Clean Training', linewidth=2)
    plt.plot(epsilons, adv_model_accuracies, 's-', label='Adversarial Training', linewidth=2)
    plt.xlabel('Perturbation Magnitude (ε)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Robustness Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('robustness_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'robustness_comparison.png'")
    plt.close()


def run_demo(model_path, dataset='cifar10', device='cuda'):
    """
    Run complete demo
    
    Args:
        model_path: Path to trained model checkpoint
        dataset: Dataset name
        device: Device to run on
    """
    print("="*60)
    print("ADVERSARIAL ROBUSTNESS DEMO")
    print("="*60)
    
    # Load model
    from models.cnn import get_model
    from training.data_loader import get_data_loaders
    
    model = get_model(dataset, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel loaded from: {model_path}")
    
    # Load data
    _, test_loader = get_data_loaders(dataset, batch_size=16)
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Visualize FGSM attack
    print("\nGenerating FGSM adversarial examples...")
    visualize_adversarial_examples(model, images, labels, 'fgsm', 0.3, device)
    
    print("\nDemo complete!")
