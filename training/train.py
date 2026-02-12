"""
Training Script for CIFAR-10 Classification

This script trains a CNN model on the CIFAR-10 dataset with options for:
- Standard training
- Adversarial training (for robustness)
- Learning rate scheduling
- Model checkpointing
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import get_model
from utils.data_loader import get_cifar10_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, adversarial=False, attack_fn=None):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        adversarial (bool): Whether to use adversarial training
        attack_fn: Function to generate adversarial examples
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples if adversarial training is enabled
        if adversarial and attack_fn is not None:
            images = attack_fn(model, images, labels)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    return test_loss, test_acc


def train_model(num_epochs=50, batch_size=128, learning_rate=0.001, 
                adversarial=False, save_dir='checkpoints', device=None):
    """
    Complete training pipeline.
    
    Args:
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Initial learning rate
        adversarial (bool): Whether to use adversarial training
        save_dir (str): Directory to save model checkpoints
        device (str): Device to train on (None for auto-detect)
    
    Returns:
        dict: Training history with losses and accuracies
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader, _, _ = get_cifar10_dataloaders(
        batch_size=batch_size,
        num_workers=2
    )
    
    # Create model
    print("\nCreating model...")
    model = get_model(device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Adversarial training setup
    attack_fn = None
    if adversarial:
        print("\nAdversarial training enabled!")
        # Import FGSM attack for adversarial training
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from attacks.fgsm import fgsm_attack
        attack_fn = lambda model, images, labels: fgsm_attack(model, images, labels, epsilon=0.03)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            adversarial=adversarial, attack_fn=attack_fn
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_type = "adversarial" if adversarial else "standard"
            save_path = os.path.join(save_dir, f'best_model_{model_type}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, save_path)
            print(f"✓ Saved best model (acc: {best_acc:.2f}%) to {save_path}")
    
    print(f"\n{'='*60}")
    print(f"Training completed! Best test accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")
    
    return history


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history.
    
    Args:
        history (dict): Training history dictionary
        save_path (str): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    ax1.plot(history['test_loss'], label='Test Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    ax2.plot(history['test_acc'], label='Test Accuracy', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--adversarial', action='store_true', help='Use adversarial training')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Train the model
    history = train_model(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        adversarial=args.adversarial,
        save_dir=args.save_dir
    )
    
    # Plot training history
    model_type = "adversarial" if args.adversarial else "standard"
    plot_training_history(history, save_path=f'training_history_{model_type}.png')
