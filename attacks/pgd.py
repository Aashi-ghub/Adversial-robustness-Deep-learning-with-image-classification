"""
Projected Gradient Descent (PGD) Attack
"""
import torch
import torch.nn as nn


def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, num_iter=40, device='cuda'):
    """
    Perform PGD attack on given images
    
    PGD is an iterative version of FGSM that takes multiple small steps
    and projects back to the epsilon ball after each step.
    
    Args:
        model: Neural network model
        images: Input images (batch)
        labels: True labels for images
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        alpha: Step size for each iteration
        num_iter: Number of iterations
        device: Device to perform attack on
        
    Returns:
        adv_images: Adversarial images
        perturbations: Perturbations added to original images
    """
    images = images.to(device)
    labels = labels.to(device)
    
    # Start with random perturbation within epsilon ball
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(num_iter):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial images
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            
            # Project back to epsilon ball
            perturbations = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = images + perturbations
            
            # Clamp to valid range [0, 1]
            adv_images = torch.clamp(adv_images, 0, 1)
    
    perturbations = adv_images - images
    return adv_images.detach(), perturbations.detach()


def pgd_attack_l2(model, images, labels, epsilon=1.0, alpha=0.1, num_iter=40, device='cuda'):
    """
    Perform PGD attack with L2 norm constraint
    
    Args:
        model: Neural network model
        images: Input images (batch)
        labels: True labels for images
        epsilon: Maximum perturbation magnitude (L2 norm)
        alpha: Step size for each iteration
        num_iter: Number of iterations
        device: Device to perform attack on
        
    Returns:
        adv_images: Adversarial images
    """
    images = images.to(device)
    labels = labels.to(device)
    
    # Start with random perturbation
    adv_images = images.clone().detach()
    
    criterion = nn.CrossEntropyLoss()
    
    for i in range(num_iter):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial images
        with torch.no_grad():
            # Gradient step
            grad = adv_images.grad
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1, keepdim=True)
            grad_norm = grad_norm.view(-1, 1, 1, 1)
            normalized_grad = grad / (grad_norm + 1e-8)
            
            adv_images = adv_images + alpha * normalized_grad
            
            # Project back to L2 ball
            perturbations = adv_images - images
            perturbation_norm = torch.norm(perturbations.view(perturbations.shape[0], -1), 
                                          dim=1, keepdim=True)
            perturbation_norm = perturbation_norm.view(-1, 1, 1, 1)
            
            factor = torch.min(torch.ones_like(perturbation_norm), 
                              epsilon / (perturbation_norm + 1e-8))
            perturbations = perturbations * factor
            adv_images = images + perturbations
            
            # Clamp to valid range [0, 1]
            adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach()
