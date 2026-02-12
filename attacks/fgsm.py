"""
Fast Gradient Sign Method (FGSM) Attack

FGSM is one of the simplest and fastest adversarial attack methods.
It generates adversarial examples by taking a single step in the direction
of the gradient of the loss with respect to the input.

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
"""

import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon=0.03):
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM).
    
    The attack perturbs the input by adding epsilon * sign(gradient) to each pixel.
    The perturbation is bounded by epsilon to keep changes imperceptible.
    
    Formula: x_adv = x + epsilon * sign(∇_x L(θ, x, y))
    
    Args:
        model: Neural network model
        images (torch.Tensor): Clean input images of shape (batch_size, C, H, W)
        labels (torch.Tensor): True labels of shape (batch_size,)
        epsilon (float): Maximum perturbation magnitude (L-infinity norm)
                        Common values: 0.01-0.1 for normalized images
    
    Returns:
        torch.Tensor: Adversarial examples of the same shape as input images
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Make a copy of images and enable gradient tracking
    images = images.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(images)
    
    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Get the sign of the gradient
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create adversarial examples
    # Perturb the image by epsilon in the direction of the gradient sign
    adversarial_images = images + epsilon * sign_data_grad
    
    # Clamp to maintain valid pixel range [0, 1] after normalization
    # Note: CIFAR-10 is normalized, so we need to clamp to maintain valid range
    adversarial_images = torch.clamp(adversarial_images, -3, 3)
    
    return adversarial_images.detach()


def fgsm_attack_targeted(model, images, target_labels, epsilon=0.03):
    """
    Generate targeted adversarial examples using FGSM.
    
    Instead of maximizing the loss for the true label, this minimizes
    the loss for a target label to make the model misclassify as the target.
    
    Formula: x_adv = x - epsilon * sign(∇_x L(θ, x, y_target))
    
    Args:
        model: Neural network model
        images (torch.Tensor): Clean input images
        target_labels (torch.Tensor): Target labels to misclassify as
        epsilon (float): Maximum perturbation magnitude
    
    Returns:
        torch.Tensor: Adversarial examples
    """
    model.eval()
    
    images = images.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(images)
    
    # Calculate loss with respect to target label
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, target_labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradient sign
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create adversarial examples (subtract to minimize loss for target)
    adversarial_images = images - epsilon * sign_data_grad
    
    # Clamp to valid range
    adversarial_images = torch.clamp(adversarial_images, -3, 3)
    
    return adversarial_images.detach()


def test_attack_success_rate(model, test_loader, epsilon=0.03, device='cpu'):
    """
    Test the success rate of FGSM attack on a dataset.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        epsilon (float): Attack strength
        device (str): Device to run on
    
    Returns:
        dict: Dictionary containing attack statistics
    """
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    print(f"\nTesting FGSM attack with epsilon={epsilon}...")
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Test on clean images
        with torch.no_grad():
            clean_outputs = model(images)
            _, clean_pred = clean_outputs.max(1)
            clean_correct += clean_pred.eq(labels).sum().item()
        
        # Generate adversarial examples
        adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        
        # Test on adversarial images
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_pred = adv_outputs.max(1)
            adv_correct += adv_pred.eq(labels).sum().item()
        
        total += labels.size(0)
    
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    attack_success_rate = 100. * (clean_correct - adv_correct) / clean_correct
    
    results = {
        'epsilon': epsilon,
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': attack_success_rate,
        'total_samples': total
    }
    
    print(f"\nAttack Results:")
    print(f"  Clean Accuracy: {clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"  Attack Success Rate: {attack_success_rate:.2f}%")
    
    return results


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.cnn_model import get_model
    from utils.data_loader import get_cifar10_dataloaders
    
    print("FGSM Attack Demo")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a small batch for demo
    _, test_loader, _, _ = get_cifar10_dataloaders(batch_size=100, num_workers=0)
    
    # Create model (in practice, load a trained model)
    model = get_model(device=device)
    print("Note: Using untrained model for demo. Load a trained model for real evaluation.")
    
    # Test different epsilon values
    epsilons = [0.0, 0.01, 0.03, 0.05, 0.1]
    
    for eps in epsilons:
        results = test_attack_success_rate(model, test_loader, epsilon=eps, device=device)
        print()
