import torch
from torchvision import datasets, transforms
from typing import Tuple, Optional
from configs.config import DataConfig

def get_data_transforms(train: bool = True) -> transforms.Compose:
    """Create and return the data transformation pipeline.
    
    Args:
        train (bool): Whether to use training transforms
        
    Returns:
        transforms.Compose: The composed transformation pipeline
    """
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def load_dataset(config: DataConfig, train: bool = True) -> datasets.CIFAR10:
    """Load the CIFAR-10 dataset.
    
    Args:
        config (DataConfig): Data configuration
        train (bool): Whether to load training set
        
    Returns:
        datasets.CIFAR10: The loaded dataset
    """
    transform = get_data_transforms(train)
    return datasets.CIFAR10(
        root=config.data_dir,
        train=train,
        download=config.download,
        transform=transform
    )

def get_data_loader(
    dataset: datasets.CIFAR10,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset (datasets.CIFAR10): The dataset to create loader for
        batch_size (int): Batch size for the loader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes
        
    Returns:
        torch.utils.data.DataLoader: The created data loader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    ) 