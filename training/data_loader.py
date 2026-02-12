"""
Data Loader for Image Classification Datasets
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(dataset='cifar10', batch_size=128, data_dir='./data', num_workers=2):
    """
    Get train and test data loaders for specified dataset
    
    Args:
        dataset: Name of the dataset ('mnist', 'cifar10', 'cifar100')
        batch_size: Batch size for data loaders
        data_dir: Directory to download/load data
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    dataset = dataset.lower()
    
    if dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        
    elif dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader
