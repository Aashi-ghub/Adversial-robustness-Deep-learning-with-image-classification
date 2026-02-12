"""
CNN Model for Image Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for image classification
    Suitable for datasets like MNIST, CIFAR-10
    """
    def __init__(self, input_channels=3, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_model(dataset='cifar10', device='cuda'):
    """
    Factory function to get appropriate model for dataset
    
    Args:
        dataset: Name of the dataset ('mnist', 'cifar10', etc.)
        device: Device to load model on
        
    Returns:
        model: CNN model
    """
    if dataset.lower() == 'mnist':
        model = SimpleCNN(input_channels=1, num_classes=10)
    elif dataset.lower() == 'cifar10':
        model = SimpleCNN(input_channels=3, num_classes=10)
    elif dataset.lower() == 'cifar100':
        model = SimpleCNN(input_channels=3, num_classes=100)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return model.to(device)
