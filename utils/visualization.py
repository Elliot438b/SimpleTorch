import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from torchvision import datasets

def get_class_names() -> Tuple[str, ...]:
    """Get the class names for CIFAR-10 dataset.
    
    Returns:
        Tuple[str, ...]: Tuple of class names
    """
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def display_dataset_info(trainset: datasets.CIFAR10) -> None:
    """Display basic information about the dataset.
    
    Args:
        trainset (datasets.CIFAR10): The dataset to display information about
    """
    print(f"\nDataset size: {len(trainset)} images")
    print(f"Image size: {trainset[0][0].shape}")
    print(f"Number of classes: {len(get_class_names())}")

def plot_sample_images(
    trainset: datasets.CIFAR10,
    classes: Tuple[str, ...],
    num_samples: int = 5,
    save_path: str = 'cifar10_samples.png'
) -> None:
    """Plot and save sample images from the dataset.
    
    Args:
        trainset (datasets.CIFAR10): The dataset to sample from
        classes (Tuple[str, ...]): The class names
        num_samples (int): Number of samples to display
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        img, label = trainset[i]
        img = img / 2 + 0.5  # Denormalize
        img = img.numpy()
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(f'Class: {classes[label]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)

def plot_class_distribution(
    trainset: datasets.CIFAR10,
    classes: Tuple[str, ...],
    save_path: str = 'class_distribution.png'
) -> None:
    """Plot and save the class distribution of the dataset.
    
    Args:
        trainset (datasets.CIFAR10): The dataset to analyze
        classes (Tuple[str, ...]): The class names
        save_path (str): Path to save the plot
    """
    class_counts = torch.zeros(10)
    for _, label in trainset:
        class_counts[label] += 1
    
    plt.figure(figsize=(10, 5))
    plt.bar(classes, class_counts)
    plt.title('Number of samples per class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)

def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: str = 'training_history.png'
) -> None:
    """Plot and save training history.
    
    Args:
        train_losses (List[float]): List of training losses
        val_losses (Optional[List[float]]): List of validation losses
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path) 