# SimpleTorch

一个使用简单 CNN 架构实现 CIFAR-10 图像分类的 PyTorch 项目。

## 项目结构

```
SimpleTorch/
├── README.md          # 项目文档
├── requirements.txt   # 项目依赖
├── main.py           # 主训练脚本
├── model.py          # CNN 模型定义
├── configs/          # 配置文件
│   ├── __init__.py
│   └── config.py     # 配置类
├── utils/            # 工具函数
│   ├── __init__.py
│   ├── data_utils.py # 数据加载和处理
│   ├── train_utils.py # 训练工具
│   └── visualization.py # 可视化工具
├── tests/            # 测试文件
│   └── __init__.py
├── checkpoints/      # 模型检查点
└── data/            # 数据集目录
```

## 环境配置

1. 创建新的 conda 环境：
```bash
conda create -n simpletorch python=3.9
conda activate simpletorch
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 模型架构

CNN 模型结构：
- 输入：3x32x32 RGB 图像
- 卷积层1：3->16 通道，3x3 卷积核
- 卷积层2：16->32 通道，3x3 卷积核
- 全连接层：32*8*8 -> 10 个类别

## 配置系统

项目使用模块化的配置系统：
- `TrainingConfig`：训练参数（批次大小、学习率等）
- `ModelConfig`：模型架构参数
- `DataConfig`：数据集和数据加载参数

配置示例：
```python
from configs.config import Config

config = Config()
config.training.batch_size = 32
config.training.learning_rate = 0.01
```

## 数据集和数据加载

### CIFAR-10 数据集概述
- 数据集大小：50,000 张图像
- 图像尺寸：3x32x32（RGB 图像）
- 类别数：10（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）

### 数据预处理
```python
from utils.data_utils import get_data_transforms, load_dataset

transform = get_data_transforms(train=True)
dataset = load_dataset(config.data)
```

### 数据集可视化
运行以下命令查看数据集样本和分布：
```bash
python show_dataset.py
```
这将生成两个可视化文件：
- `cifar10_samples.png`：数据集样本图像
- `class_distribution.png`：类别分布可视化

## 训练

1. 运行训练：
```bash
python main.py
```

2. 脚本将执行以下操作：
   - 下载 CIFAR-10 数据集
   - 训练模型
   - 显示训练进度
   - 保存模型检查点
   - 生成训练可视化

## 可视化工具

项目包含多个可视化工具：
- 数据集样本可视化
- 类别分布绘图
- 训练历史绘图
- 模型预测可视化

## 环境要求

- Python 3.9
- PyTorch 2.6.0
- torchvision
- numpy
- matplotlib

## 未来改进计划

1. 添加验证集评估
2. 实现模型检查点保存
3. 添加训练可视化
4. 优化训练策略
5. 添加测试集评估
6. 添加单元测试
7. 添加 CI/CD 流程
8. 添加性能分析工具

## 许可证

MIT 许可证 