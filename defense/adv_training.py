"""
Adversarial Training Defense
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def adversarial_train_epoch(model, train_loader, optimizer, criterion, device, 
                            attack_method='pgd', epsilon=0.3):
    """
    Train model for one epoch using adversarial training
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to train on
        attack_method: Attack method to use ('fgsm' or 'pgd')
        epsilon: Perturbation magnitude
        
    Returns:
        avg_loss: Average training loss
        clean_accuracy: Accuracy on clean examples
        adv_accuracy: Accuracy on adversarial examples
    """
    model.train()
    total_loss = 0
    clean_correct = 0
    adv_correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Adversarial Training')):
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        if attack_method == 'fgsm':
            adv_data, _ = fgsm_attack(model, data, target, epsilon, device)
        elif attack_method == 'pgd':
            adv_data, _ = pgd_attack(model, data, target, epsilon, alpha=0.01, 
                                     num_iter=10, device=device)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        # Train on both clean and adversarial examples
        optimizer.zero_grad()
        
        # Forward pass on clean data
        clean_output = model(data)
        clean_loss = criterion(clean_output, target)
        
        # Forward pass on adversarial data
        adv_output = model(adv_data)
        adv_loss = criterion(adv_output, target)
        
        # Combined loss
        loss = (clean_loss + adv_loss) / 2
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        total_loss += loss.item()
        _, clean_pred = clean_output.max(1)
        _, adv_pred = adv_output.max(1)
        total += target.size(0)
        clean_correct += clean_pred.eq(target).sum().item()
        adv_correct += adv_pred.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    clean_accuracy = 100. * clean_correct / total
    adv_accuracy = 100. * adv_correct / total
    
    return avg_loss, clean_accuracy, adv_accuracy


def adversarial_train(model, train_loader, test_loader, epochs=10, lr=0.01,
                     device='cuda', attack_method='pgd', epsilon=0.3,
                     save_path='checkpoints/adv_model.pth'):
    """
    Complete adversarial training pipeline
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        attack_method: Attack method to use ('fgsm' or 'pgd')
        epsilon: Perturbation magnitude
        save_path: Path to save best model
        
    Returns:
        model: Adversarially trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_adv_acc = 0
    
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        
        train_loss, clean_acc, adv_acc = adversarial_train_epoch(
            model, train_loader, optimizer, criterion, device, attack_method, epsilon
        )
        
        print(f'Train Loss: {train_loss:.3f}')
        print(f'Clean Acc: {clean_acc:.2f}% | Adv Acc: {adv_acc:.2f}%')
        
        # Save checkpoint if best adversarial accuracy
        if adv_acc > best_adv_acc:
            print(f'Saving model (adv accuracy improved: {best_adv_acc:.2f}% -> {adv_acc:.2f}%)')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'clean_accuracy': clean_acc,
                'adv_accuracy': adv_acc,
            }, save_path)
            best_adv_acc = adv_acc
        
        scheduler.step()
    
    print(f'\nAdversarial training complete. Best adversarial accuracy: {best_adv_acc:.2f}%')
    return model
