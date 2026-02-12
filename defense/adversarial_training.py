"""
Adversarial Training Defense

Adversarial training is a defense technique that trains the model on
adversarial examples to improve robustness. It's one of the most effective
defense methods against adversarial attacks.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks", ICLR 2018
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def adversarial_training_epoch(model, train_loader, criterion, optimizer, device, 
                               attack_type='pgd', epsilon=0.03, alpha=0.01, num_iter=7):
    """
    Train the model for one epoch using adversarial training.
    
    In adversarial training, we generate adversarial examples on-the-fly
    during training and train the model to correctly classify them.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        attack_type (str): Type of attack ('fgsm' or 'pgd')
        epsilon (float): Maximum perturbation for attacks
        alpha (float): Step size for PGD
        num_iter (int): Number of iterations for PGD
    
    Returns:
        tuple: (average_loss, clean_accuracy, adversarial_accuracy)
    """
    model.train()
    
    running_loss = 0.0
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Adversarial Training")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        model.eval()  # Set to eval mode for attack generation
        if attack_type == 'fgsm':
            adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        elif attack_type == 'pgd':
            adv_images = pgd_attack(model, images, labels, epsilon=epsilon, 
                                   alpha=alpha, num_iter=num_iter)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        model.train()  # Set back to train mode
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass on adversarial examples
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Check accuracy on adversarial examples
        _, predicted = outputs.max(1)
        total += labels.size(0)
        adv_correct += predicted.eq(labels).sum().item()
        
        # Also check clean accuracy
        with torch.no_grad():
            clean_outputs = model(images)
            _, clean_pred = clean_outputs.max(1)
            clean_correct += clean_pred.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'clean_acc': f'{100.*clean_correct/total:.2f}%',
            'adv_acc': f'{100.*adv_correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    
    return avg_loss, clean_acc, adv_acc


def mixed_training_epoch(model, train_loader, criterion, optimizer, device,
                         mix_ratio=0.5, attack_type='pgd', epsilon=0.03):
    """
    Train with a mix of clean and adversarial examples.
    
    This approach balances clean accuracy and adversarial robustness by
    training on both clean and adversarial examples in each batch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        mix_ratio (float): Ratio of adversarial examples (0.0 to 1.0)
        attack_type (str): Type of attack
        epsilon (float): Attack strength
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Mixed Training")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Split batch into clean and adversarial
        num_adv = int(batch_size * mix_ratio)
        
        if num_adv > 0:
            # Generate adversarial examples for part of the batch
            model.eval()
            adv_images = images[:num_adv].clone()
            adv_labels = labels[:num_adv].clone()
            
            if attack_type == 'fgsm':
                adv_images = fgsm_attack(model, adv_images, adv_labels, epsilon=epsilon)
            else:
                adv_images = pgd_attack(model, adv_images, adv_labels, epsilon=epsilon)
            
            # Combine clean and adversarial
            mixed_images = torch.cat([images[num_adv:], adv_images], dim=0)
            mixed_labels = torch.cat([labels[num_adv:], adv_labels], dim=0)
        else:
            mixed_images = images
            mixed_labels = labels
        
        model.train()
        
        # Training step
        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += mixed_labels.size(0)
        correct += predicted.eq(mixed_labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    acc = 100. * correct / total
    
    return avg_loss, acc


def evaluate_robustness(model, test_loader, device, epsilon=0.03):
    """
    Evaluate model robustness against multiple attacks.
    
    Tests the model against:
    - Clean examples
    - FGSM attack
    - PGD attack
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        epsilon (float): Attack strength
    
    Returns:
        dict: Accuracy results for each scenario
    """
    model.eval()
    
    clean_correct = 0
    fgsm_correct = 0
    pgd_correct = 0
    total = 0
    
    print(f"\nEvaluating robustness with epsilon={epsilon}...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            
            # Clean accuracy
            outputs = model(images)
            _, pred = outputs.max(1)
            clean_correct += pred.eq(labels).sum().item()
            
            total += labels.size(0)
    
    # FGSM attack
    print("Testing FGSM attack...")
    for images, labels in tqdm(test_loader, desc="FGSM"):
        images, labels = images.to(device), labels.to(device)
        adv_images = fgsm_attack(model, images, labels, epsilon=epsilon)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, pred = outputs.max(1)
            fgsm_correct += pred.eq(labels).sum().item()
    
    # PGD attack
    print("Testing PGD attack...")
    for images, labels in tqdm(test_loader, desc="PGD"):
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(model, images, labels, epsilon=epsilon)
        
        with torch.no_grad():
            outputs = model(adv_images)
            _, pred = outputs.max(1)
            pgd_correct += pred.eq(labels).sum().item()
    
    results = {
        'clean_accuracy': 100. * clean_correct / total,
        'fgsm_accuracy': 100. * fgsm_correct / total,
        'pgd_accuracy': 100. * pgd_correct / total
    }
    
    print(f"\nRobustness Evaluation Results:")
    print(f"  Clean Accuracy: {results['clean_accuracy']:.2f}%")
    print(f"  FGSM Robustness: {results['fgsm_accuracy']:.2f}%")
    print(f"  PGD Robustness: {results['pgd_accuracy']:.2f}%")
    
    return results


if __name__ == "__main__":
    from models.cnn_model import get_model
    from utils.data_loader import get_cifar10_dataloaders
    
    print("Adversarial Training Demo")
    print("=" * 60)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data (small batch for demo)
    train_loader, test_loader, _, _ = get_cifar10_dataloaders(
        batch_size=64,
        num_workers=0
    )
    
    # Create model
    model = get_model(device=device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Demo: One epoch of adversarial training
    print("\nRunning one epoch of adversarial training (demo)...")
    loss, clean_acc, adv_acc = adversarial_training_epoch(
        model, train_loader, criterion, optimizer, device,
        attack_type='fgsm', epsilon=0.03
    )
    
    print(f"\nEpoch Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Clean Accuracy: {clean_acc:.2f}%")
    print(f"  Adversarial Accuracy: {adv_acc:.2f}%")
