import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def show_dataset():
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练集
    print("Loading CIFAR-10 dataset...")
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # 定义类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 显示数据集信息
    print(f"\nDataset size: {len(trainset)} images")
    print(f"Image size: {trainset[0][0].shape}")
    print(f"Number of classes: {len(classes)}")
    
    # 显示样本图片
    plt.figure(figsize=(15, 5))
    for i in range(5):
        img, label = trainset[i]
        img = img / 2 + 0.5
        img = img.numpy()
        plt.subplot(1, 5, i + 1)
        plt.imshow(img.transpose(1, 2, 0))
        plt.title(f'Class: {classes[label]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cifar10_samples.png')
    
    # 显示每个类别的样本数量
    class_counts = torch.zeros(10)
    for _, label in trainset:
        class_counts[label] += 1
    
    plt.figure(figsize=(10, 5))
    plt.bar(classes, class_counts)
    plt.title('Number of samples per class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    print("\nVisualization files saved:")
    print("- cifar10_samples.png")
    print("- class_distribution.png")

if __name__ == '__main__':
    show_dataset() 