"""
Fast Gradient Sign Method (FGSM) Attack
"""
import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon=0.3, device='cuda'):
    """
    Perform FGSM attack on given images
    
    FGSM is a simple yet effective adversarial attack that perturbs
    the input in the direction of the gradient of the loss with respect
    to the input.
    
    Args:
        model: Neural network model
        images: Input images (batch)
        labels: True labels for images
        epsilon: Perturbation magnitude
        device: Device to perform attack on
        
    Returns:
        adv_images: Adversarial images
        perturbations: Perturbations added to original images
    """
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    
    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect gradients
    data_grad = images.grad.data
    
    # Generate adversarial examples
    sign_data_grad = data_grad.sign()
    perturbations = epsilon * sign_data_grad
    adv_images = images + perturbations
    
    # Clamp to valid range [0, 1]
    adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach(), perturbations.detach()


def fgsm_attack_untargeted(model, images, epsilon=0.3, device='cuda'):
    """
    Perform untargeted FGSM attack (maximize loss for true label)
    
    Args:
        model: Neural network model
        images: Input images (batch)
        epsilon: Perturbation magnitude
        device: Device to perform attack on
        
    Returns:
        adv_images: Adversarial images
    """
    images = images.to(device)
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    
    # Get predicted labels
    _, labels = torch.max(outputs, 1)
    
    # Calculate loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect gradients
    data_grad = images.grad.data
    
    # Generate adversarial examples
    sign_data_grad = data_grad.sign()
    adv_images = images + epsilon * sign_data_grad
    
    # Clamp to valid range [0, 1]
    adv_images = torch.clamp(adv_images, 0, 1)
    
    return adv_images.detach()
