# SimpleTorch

A PyTorch implementation of CIFAR-10 image classification using a simple CNN architecture.

## Project Structure

```
SimpleTorch/
├── README.md          # Project documentation
├── requirements.txt   # Project dependencies
├── main.py           # Main training script
├── model.py          # CNN model definition
└── data/             # Dataset directory
```

## Environment Setup

1. Create a new conda environment:
```bash
conda create -n simpletorch python=3.9
conda activate simpletorch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

The CNN model consists of:
- Input: 3x32x32 RGB images
- Conv1: 3->32 channels, 3x3 kernel
- Conv2: 32->64 channels, 3x3 kernel
- Conv3: 64->64 channels, 3x3 kernel
- FC1: 64*4*4 -> 512 neurons
- FC2: 512 -> 10 classes (CIFAR-10 categories)

## Usage

1. Run training:
```bash
python main.py
```

2. The script will:
   - Download CIFAR-10 dataset
   - Train the model
   - Save the best model checkpoint
   - Display training progress

## Dataset

CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes:
- 50,000 training images
- 10,000 test images

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Requirements

- Python 3.9
- PyTorch 2.6.0
- torchvision
- numpy
- matplotlib 