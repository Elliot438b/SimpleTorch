# SimpleTorch

一个简单的深度学习项目，使用 PyTorch 框架实现 MNIST 手写数字识别。

## 功能特点

- 基本的神经网络模型实现
- MNIST 数据集加载和预处理
- 模型训练和评估
- 可视化工具

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）

## 安装依赖

```bash
# 创建并激活 conda 环境
conda create -n simpletorch python=3.9
conda activate simpletorch

# 安装依赖包
pip install -r requirements.txt
```

## 使用方法

1. 训练模型：
```bash
python main.py
```

2. 运行可视化测试：
```bash
python testPlt.py
```

## 项目结构

```
SimpleTorch/
├── README.md
├── requirements.txt
├── main.py          # 主程序入口
├── model.py         # 模型定义
├── testPlt.py       # 可视化测试
└── data/            # 数据集目录
```

## 注意事项

- 首次运行时会自动下载 MNIST 数据集
- 确保有足够的磁盘空间存储数据集
- 如果下载失败，可以手动下载数据集并放入 data 目录 