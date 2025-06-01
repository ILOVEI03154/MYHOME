"""
DAY 41 简单CNN

本节重点：
1. 数据增强
2. CNN结构定义
3. Batch Normalization
4. 特征图可视化
5. 学习率调度器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10数据集的类别
classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

#====================== 1. 数据加载与增强 ======================

def load_data(batch_size=64, is_train=True):
    """
    加载CIFAR-10数据集，并应用数据增强
    """
    if is_train:
        # 训练集使用数据增强
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomRotation(10),      # 随机旋转
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # 随机仿射变换
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # 测试集只需要标准化
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=is_train,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2
    )
    
    return dataloader

#====================== 2. CNN模型定义 ======================

class SimpleNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleNet, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 保存特征图用于可视化
        self.feature_maps = []
        
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        self.feature_maps.append(x)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        self.feature_maps.append(x)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        self.feature_maps.append(x)
        
        # 全连接层
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

#====================== 3. 训练函数 ======================

def train(model, train_loader, optimizer, epoch, history):
    """
    训练一个epoch
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    epoch_loss = train_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    
    return epoch_loss, epoch_acc

#====================== 4. 测试函数 ======================

def test(model, test_loader, history):
    """
    在测试集上评估模型
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    history['test_loss'].append(test_loss)
    history['test_acc'].append(accuracy)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

#====================== 5. 可视化函数 ======================

def plot_training_history(history):
    """
    绘制训练历史曲线
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['test_loss'], 'r-', label='测试损失')
    plt.title('训练和测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    plt.plot(epochs, history['test_acc'], 'r-', label='测试准确率')
    plt.title('训练和测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, test_loader):
    """
    可视化特征图
    """
    # 获取一个批次的数据
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    
    # 获取特征图
    with torch.no_grad():
        _ = model(images[0:1].to(device))
    
    # 显示原始图像和每层的特征图
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, 4, 1)
    img = images[0] / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示每层的特征图
    for i, feature_map in enumerate(model.feature_maps, 2):
        plt.subplot(1, 4, i)
        # 选择第一个样本的第一个特征图
        plt.imshow(feature_map[0, 0].cpu(), cmap='viridis')
        plt.title(f'层 {i-1} 特征图')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#====================== 6. 主函数 ======================

def main():
    # 超参数设置
    batch_size = 64
    epochs = 4
    lr = 0.01
    dropout_rate = 0.5
    
    # 初始化训练历史记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # 加载数据
    print("正在加载训练集...")
    train_loader = load_data(batch_size, is_train=True)
    print("正在加载测试集...")
    test_loader = load_data(batch_size, is_train=False)
    
    # 创建模型
    model = SimpleNet(dropout_rate=dropout_rate).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    # 训练和测试
    print(f"开始训练，使用设备: {device}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, epoch, history)
        test_loss, test_acc = test(model, test_loader, history)
        scheduler.step()
    
    # 可视化训练过程
    print("训练完成，绘制训练历史...")
    plot_training_history(history)
    
    # 可视化特征图
    print("可视化特征图...")
    visualize_feature_maps(model, test_loader)

if __name__ == '__main__':
    main()

"""
学习要点：

1. 数据增强
- transforms.RandomHorizontalFlip(): 随机水平翻转
- transforms.RandomRotation(): 随机旋转
- transforms.RandomAffine(): 随机仿射变换
- transforms.ColorJitter(): 颜色抖动

2. CNN结构
- 常见流程：输入 → 卷积层 → BN → 激活函数 → 池化层
- 特征提取：多个卷积块串联
- 分类器：Flatten后接全连接层

3. Batch Normalization
- 在卷积层后添加
- 训练时计算并更新均值和方差
- 测试时使用训练阶段的统计量

4. 特征图
- 保存每层的特征图用于可视化
- 观察模型学习到的特征
- 帮助理解模型的工作原理

5. 学习率调度器
- 使用StepLR按步长降低学习率
- 帮助模型更好地收敛
- 避免学习率过大或过小
"""