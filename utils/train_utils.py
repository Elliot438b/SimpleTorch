import torch
import torch.nn as nn
from typing import Tuple, Optional
from configs.config import TrainingConfig

def setup_device(config: TrainingConfig) -> torch.device:
    """Set up and return the appropriate device.
    
    Args:
        config (TrainingConfig): Training configuration
        
    Returns:
        torch.device: The device to be used for training
    """
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def train_one_epoch(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int
) -> float:
    """Train the model for one epoch.
    
    Args:
        model (nn.Module): The neural network model
        trainloader (torch.utils.data.DataLoader): The training data loader
        criterion (nn.CrossEntropyLoss): The loss function
        optimizer (torch.optim.Optimizer): The optimizer
        device (torch.device): The device to train on
        epoch (int): Current epoch number
        num_epochs (int): Total number of epochs
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Batch [{i + 1}/{len(trainloader)}], '
                  f'Loss: {loss.item():.4f}')
    
    return running_loss / len(trainloader)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
) -> None:
    """Save a model checkpoint.
    
    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer state
        epoch (int): Current epoch number
        loss (float): Current loss value
        path (str): Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path) 