from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    learning_rate: float = 0.01
    num_epochs: int = 1
    device: str = "cuda"  # or "cpu"
    seed: int = 42

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    input_channels: int = 3
    conv1_channels: int = 16
    conv2_channels: int = 32
    num_classes: int = 10
    kernel_size: int = 3
    padding: int = 1

@dataclass
class DataConfig:
    """Dataset and data loading configuration."""
    data_dir: str = "./data"
    image_size: Tuple[int, int] = (32, 32)
    train_transform: bool = True
    download: bool = True

@dataclass
class Config:
    """Main configuration class combining all configs."""
    training: TrainingConfig = TrainingConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig() 