"""
Data Loading and Preprocessing Utilities

This module provides functions to load and preprocess the CIFAR-10 dataset
with appropriate transformations for training and testing.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_cifar10_dataloaders(batch_size=128, num_workers=2, data_dir='./data'):
    """
    Load CIFAR-10 dataset with appropriate transformations.
    
    The function applies:
    - Training set: Data augmentation (random crop, horizontal flip) + normalization
    - Test set: Only normalization
    
    CIFAR-10 normalization values:
    - Mean: [0.4914, 0.4822, 0.4465]
    - Std: [0.2470, 0.2435, 0.2616]
    
    Args:
        batch_size (int): Number of samples per batch
        num_workers (int): Number of worker threads for data loading
        data_dir (str): Directory to store/load the dataset
    
    Returns:
        tuple: (train_loader, test_loader, trainset, testset)
            - train_loader: DataLoader for training data
            - test_loader: DataLoader for test data
            - trainset: Training dataset
            - testset: Test dataset
    """
    
    # Normalization parameters for CIFAR-10
    # These are computed from the training set
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Training transformations with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Random crop with padding
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ToTensor(),                  # Convert to tensor
        transforms.Normalize(mean, std)         # Normalize
    ])
    
    # Test transformations (only normalization, no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load training dataset
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    # Load test dataset
    testset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    # Create data loaders
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speed up data transfer to GPU
    )
    
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"CIFAR-10 dataset loaded successfully!")
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, test_loader, trainset, testset


def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    """
    Denormalize a normalized tensor for visualization.
    
    Args:
        tensor (torch.Tensor): Normalized tensor of shape (C, H, W) or (B, C, H, W)
        mean (list): Mean values used for normalization
        std (list): Std values used for normalization
    
    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    # Handle both single image and batch
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    return tensor * std + mean


def get_class_name(label):
    """
    Get the class name for a given label.
    
    Args:
        label (int): Class label (0-9)
    
    Returns:
        str: Class name
    """
    return CIFAR10_CLASSES[label]


if __name__ == "__main__":
    # Test the data loading
    print("Testing CIFAR-10 data loading...")
    
    train_loader, test_loader, trainset, testset = get_cifar10_dataloaders(
        batch_size=4,
        num_workers=0
    )
    
    # Get a batch from training data
    images, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")
    
    print(f"\nClass labels in batch: {labels.tolist()}")
    print(f"Class names: {[get_class_name(label.item()) for label in labels]}")
