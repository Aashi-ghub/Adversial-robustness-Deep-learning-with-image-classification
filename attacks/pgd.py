"""
Projected Gradient Descent (PGD) Attack

PGD is a stronger iterative adversarial attack that applies FGSM multiple times
with small step sizes and projects the result back to the epsilon-ball after each step.
It is considered one of the strongest first-order adversarial attacks.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
"""

import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, num_iter=10, random_start=True):
    """
    Generate adversarial examples using Projected Gradient Descent (PGD).
    
    PGD is an iterative attack that:
    1. Optionally starts from a random point in the epsilon-ball
    2. Takes multiple small steps (alpha) in the gradient direction
    3. Projects back to the epsilon-ball after each step
    
    Args:
        model: Neural network model
        images (torch.Tensor): Clean input images of shape (batch_size, C, H, W)
        labels (torch.Tensor): True labels of shape (batch_size,)
        epsilon (float): Maximum perturbation magnitude (L-infinity norm)
        alpha (float): Step size for each iteration (typically epsilon/num_iter)
        num_iter (int): Number of iterations
        random_start (bool): Whether to start from a random point in epsilon-ball
    
    Returns:
        torch.Tensor: Adversarial examples of the same shape as input images
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Clone images for perturbation
    adv_images = images.clone().detach()
    
    # Random initialization within epsilon-ball
    if random_start:
        # Add uniform random noise in [-epsilon, epsilon]
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        # Clamp to valid range
        adv_images = torch.clamp(adv_images, -3, 3)
    
    # Iterative attack
    for i in range(num_iter):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        grad_sign = adv_images.grad.sign()
        
        # Update adversarial images
        # Take a step in the direction of the gradient
        adv_images = adv_images.detach() + alpha * grad_sign
        
        # Project back to epsilon-ball around original images
        # Ensure perturbation stays within [-epsilon, epsilon]
        perturbation = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = images + perturbation
        
        # Clamp to maintain valid range
        adv_images = torch.clamp(adv_images, -3, 3)
    
    return adv_images.detach()


def pgd_attack_targeted(model, images, target_labels, epsilon=0.03, alpha=0.01, num_iter=10, random_start=True):
    """
    Generate targeted adversarial examples using PGD.
    
    This variant tries to make the model misclassify as a specific target class.
    
    Args:
        model: Neural network model
        images (torch.Tensor): Clean input images
        target_labels (torch.Tensor): Target labels to misclassify as
        epsilon (float): Maximum perturbation magnitude
        alpha (float): Step size for each iteration
        num_iter (int): Number of iterations
        random_start (bool): Whether to start from a random point
    
    Returns:
        torch.Tensor: Adversarial examples
    """
    model.eval()
    
    adv_images = images.clone().detach()
    
    # Random initialization
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, -3, 3)
    
    # Iterative attack
    for i in range(num_iter):
        adv_images.requires_grad = True
        
        outputs = model(adv_images)
        
        # Loss with respect to target label
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, target_labels)
        
        model.zero_grad()
        loss.backward()
        
        grad_sign = adv_images.grad.sign()
        
        # Step in opposite direction to minimize loss for target
        adv_images = adv_images.detach() - alpha * grad_sign
        
        # Project back to epsilon-ball
        perturbation = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = images + perturbation
        
        # Clamp to valid range
        adv_images = torch.clamp(adv_images, -3, 3)
    
    return adv_images.detach()


def test_attack_success_rate(model, test_loader, epsilon=0.03, alpha=0.01, num_iter=10, device='cpu'):
    """
    Test the success rate of PGD attack on a dataset.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        epsilon (float): Maximum perturbation
        alpha (float): Step size
        num_iter (int): Number of iterations
        device (str): Device to run on
    
    Returns:
        dict: Dictionary containing attack statistics
    """
    model.eval()
    
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    print(f"\nTesting PGD attack with epsilon={epsilon}, alpha={alpha}, num_iter={num_iter}...")
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Test on clean images
        with torch.no_grad():
            clean_outputs = model(images)
            _, clean_pred = clean_outputs.max(1)
            clean_correct += clean_pred.eq(labels).sum().item()
        
        # Generate adversarial examples
        adv_images = pgd_attack(model, images, labels, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
        
        # Test on adversarial images
        with torch.no_grad():
            adv_outputs = model(adv_images)
            _, adv_pred = adv_outputs.max(1)
            adv_correct += adv_pred.eq(labels).sum().item()
        
        total += labels.size(0)
    
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    attack_success_rate = 100. * (clean_correct - adv_correct) / clean_correct if clean_correct > 0 else 0
    
    results = {
        'epsilon': epsilon,
        'alpha': alpha,
        'num_iter': num_iter,
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


def compare_with_fgsm(model, test_loader, epsilon=0.03, device='cpu'):
    """
    Compare PGD attack strength with FGSM.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        epsilon (float): Attack strength
        device (str): Device to run on
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from attacks.fgsm import fgsm_attack
    
    model.eval()
    
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0
    
    print(f"\nComparing FGSM vs PGD (epsilon={epsilon})...")
    
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Clean accuracy
        with torch.no_grad():
            outputs = model(images)
            _, pred = outputs.max(1)
            clean_correct += pred.eq(labels).sum().item()
        
        # FGSM attack
        fgsm_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        with torch.no_grad():
            outputs = model(fgsm_images)
            _, pred = outputs.max(1)
            fgsm_correct += pred.eq(labels).sum().item()
        
        # PGD attack
        pgd_images = pgd_attack(model, images, labels, epsilon=epsilon)
        with torch.no_grad():
            outputs = model(pgd_images)
            _, pred = outputs.max(1)
            pgd_correct += pred.eq(labels).sum().item()
        
        total += labels.size(0)
    
    clean_acc = 100. * clean_correct / total
    fgsm_acc = 100. * fgsm_correct / total
    pgd_acc = 100. * pgd_correct / total
    
    print(f"\nComparison Results:")
    print(f"  Clean Accuracy: {clean_acc:.2f}%")
    print(f"  FGSM Accuracy: {fgsm_acc:.2f}%")
    print(f"  PGD Accuracy: {pgd_acc:.2f}%")
    print(f"\nPGD is typically stronger (lower accuracy) than FGSM")


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models.cnn_model import get_model
    from utils.data_loader import get_cifar10_dataloaders
    
    print("PGD Attack Demo")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load a small batch for demo
    _, test_loader, _, _ = get_cifar10_dataloaders(batch_size=100, num_workers=0)
    
    # Create model (in practice, load a trained model)
    model = get_model(device=device)
    print("Note: Using untrained model for demo. Load a trained model for real evaluation.")
    
    # Test PGD attack
    results = test_attack_success_rate(model, test_loader, epsilon=0.03, device=device)
