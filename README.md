# SimpleTorch

A PyTorch implementation of CIFAR-10 image classification using a simple CNN architecture.

## Project Structure

```
SimpleTorch/
├── README.md          # Project documentation
├── requirements.txt   # Project dependencies
├── main.py           # Main training script
├── model.py          # CNN model definition
├── show_dataset.py   # Dataset visualization script
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
- Conv1: 3->16 channels, 3x3 kernel
- Conv2: 16->32 channels, 3x3 kernel
- FC1: 32*8*8 -> 10 classes

## Dataset and Data Loading

### CIFAR-10 Dataset Overview
- Dataset size: 50,000 images
- Image size: 3x32x32 (RGB images)
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

### Data Preprocessing
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Dataset Visualization
Run the following command to view dataset samples and distribution:
```bash
python show_dataset.py
```
This will generate two visualization files:
- `cifar10_samples.png`: Sample images from the dataset
- `class_distribution.png`: Class distribution visualization

## Usage

1. Run training:
```bash
python main.py
```

2. The script will:
   - Download CIFAR-10 dataset
   - Train the model
   - Display training progress

## Requirements

- Python 3.9
- PyTorch 2.6.0
- torchvision
- numpy
- matplotlib

## Future Improvements

1. Add validation set evaluation
2. Implement model checkpointing
3. Add training visualization
4. Optimize training strategies
5. Add test set evaluation

## License

MIT License 