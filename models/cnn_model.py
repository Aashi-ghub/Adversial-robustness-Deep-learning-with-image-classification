"""
CNN Model for CIFAR-10 Image Classification

This module defines a Convolutional Neural Network (CNN) architecture
optimized for the CIFAR-10 dataset (32x32 RGB images, 10 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    A simple CNN architecture for CIFAR-10 classification.
    
    Architecture:
    - 3 Convolutional blocks with increasing filters (32, 64, 128)
    - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool
    - 2 Fully connected layers
    - Dropout for regularization
    
    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        dropout_rate (float): Dropout probability (default: 0.5)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CIFAR10CNN, self).__init__()
        
        # First convolutional block
        # Input: 3x32x32 -> Output: 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        # Input: 32x16x16 -> Output: 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        # Input: 64x8x8 -> Output: 128x4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        # After flattening: 128*4*4 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # First block: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second block: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third block: Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        
        # First FC layer with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Output layer (logits)
        x = self.fc2(x)
        
        return x


def get_model(num_classes=10, dropout_rate=0.5, device='cpu'):
    """
    Factory function to create and initialize a CIFAR10CNN model.
    
    Args:
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout probability
        device (str): Device to place the model on ('cpu' or 'cuda')
    
    Returns:
        CIFAR10CNN: Initialized model on the specified device
    """
    model = CIFAR10CNN(num_classes=num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing CIFAR10CNN model...")
    model = get_model()
    
    # Create dummy input (batch_size=4, channels=3, height=32, width=32)
    dummy_input = torch.randn(4, 3, 32, 32)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
