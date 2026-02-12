"""
Robustness Evaluation Module
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack


def evaluate_clean(model, test_loader, device='cuda'):
    """
    Evaluate model on clean (non-adversarial) examples
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        accuracy: Clean accuracy
        avg_loss: Average loss
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating Clean'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def evaluate_adversarial(model, test_loader, attack_method='fgsm', 
                         epsilon=0.3, device='cuda'):
    """
    Evaluate model on adversarial examples
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        attack_method: Attack method ('fgsm' or 'pgd')
        epsilon: Perturbation magnitude
        device: Device to evaluate on
        
    Returns:
        accuracy: Adversarial accuracy
        avg_loss: Average loss on adversarial examples
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(test_loader, desc=f'Evaluating {attack_method.upper()}'):
        data, target = data.to(device), target.to(device)
        
        # Generate adversarial examples
        if attack_method == 'fgsm':
            adv_data, _ = fgsm_attack(model, data, target, epsilon, device)
        elif attack_method == 'pgd':
            adv_data, _ = pgd_attack(model, data, target, epsilon, 
                                     alpha=0.01, num_iter=40, device=device)
        else:
            raise ValueError(f"Unknown attack method: {attack_method}")
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            output = model(adv_data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def comprehensive_evaluation(model, test_loader, device='cuda', 
                            epsilons=[0.0, 0.1, 0.2, 0.3]):
    """
    Comprehensive robustness evaluation across multiple attack strengths
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        device: Device to evaluate on
        epsilons: List of perturbation magnitudes to test
        
    Returns:
        results: Dictionary containing evaluation results
    """
    results = {
        'clean': {},
        'fgsm': {},
        'pgd': {}
    }
    
    # Clean accuracy
    print("\n=== Evaluating Clean Accuracy ===")
    clean_acc, clean_loss = evaluate_clean(model, test_loader, device)
    results['clean']['accuracy'] = clean_acc
    results['clean']['loss'] = clean_loss
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    # FGSM accuracy across different epsilons
    print("\n=== Evaluating FGSM Robustness ===")
    for eps in epsilons:
        if eps == 0.0:
            continue
        print(f"\nEpsilon: {eps}")
        fgsm_acc, fgsm_loss = evaluate_adversarial(
            model, test_loader, 'fgsm', eps, device
        )
        results['fgsm'][eps] = {'accuracy': fgsm_acc, 'loss': fgsm_loss}
        print(f"FGSM Accuracy (ε={eps}): {fgsm_acc:.2f}%")
    
    # PGD accuracy across different epsilons
    print("\n=== Evaluating PGD Robustness ===")
    for eps in epsilons:
        if eps == 0.0:
            continue
        print(f"\nEpsilon: {eps}")
        pgd_acc, pgd_loss = evaluate_adversarial(
            model, test_loader, 'pgd', eps, device
        )
        results['pgd'][eps] = {'accuracy': pgd_acc, 'loss': pgd_loss}
        print(f"PGD Accuracy (ε={eps}): {pgd_acc:.2f}%")
    
    return results


def print_evaluation_summary(results):
    """
    Print formatted summary of evaluation results
    
    Args:
        results: Dictionary containing evaluation results
    """
    print("\n" + "="*60)
    print("ROBUSTNESS EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nClean Accuracy: {results['clean']['accuracy']:.2f}%")
    
    print("\nFGSM Attack Results:")
    print("-" * 40)
    for eps, metrics in results['fgsm'].items():
        print(f"  ε={eps}: {metrics['accuracy']:.2f}%")
    
    print("\nPGD Attack Results:")
    print("-" * 40)
    for eps, metrics in results['pgd'].items():
        print(f"  ε={eps}: {metrics['accuracy']:.2f}%")
    
    print("\n" + "="*60)
