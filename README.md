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

## 数据集的选择与加载

### CIFAR-10 数据集概述
- 数据集大小：50,000 张图片
- 图片大小：3x32x32（RGB图像）
- 类别数：10个（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）

### 数据预处理
```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### 数据集可视化
运行以下命令查看数据集样本和分布：
```bash
python show_dataset.py
```
这将生成两个可视化文件：
- `cifar10_samples.png`：数据集样本图片
- `class_distribution.png`：类别分布

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