"""
Standard Training Script
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train model for one epoch
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss criterion
        device: Device to train on
        
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc='Training')):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def test(model, test_loader, criterion, device):
    """
    Evaluate model on test set
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss criterion
        device: Device to evaluate on
        
    Returns:
        avg_loss: Average test loss
        accuracy: Test accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train(model, train_loader, test_loader, epochs=10, lr=0.01, 
          device='cuda', save_path='checkpoints/model.pth'):
    """
    Complete training pipeline
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model
        
    Returns:
        model: Trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
        
        # Save checkpoint if best accuracy
        if test_acc > best_acc:
            print(f'Saving model (accuracy improved: {best_acc:.2f}% -> {test_acc:.2f}%)')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': test_acc,
            }, save_path)
            best_acc = test_acc
        
        scheduler.step()
    
    print(f'\nTraining complete. Best accuracy: {best_acc:.2f}%')
    return model
