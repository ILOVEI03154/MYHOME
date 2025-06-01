"""
DAY 41 实验：比较不同的调度器和CNN结构

本文件提供了多种CNN结构和学习率调度器的实现，
用于比较不同配置下的训练效果。
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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR

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
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

#====================== 2. 不同的CNN模型结构 ======================

class BasicCNN(nn.Module):
    """基础CNN模型：3个卷积层"""
    def __init__(self, use_bn=True, dropout_rate=0.5):
        super(BasicCNN, self).__init__()
        self.use_bn = use_bn
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 全连接层
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class DeepCNN(nn.Module):
    """更深的CNN模型：5个卷积层"""
    def __init__(self, use_bn=True, dropout_rate=0.5):
        super(DeepCNN, self).__init__()
        self.use_bn = use_bn
        
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_bn else nn.Identity()
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_bn else nn.Identity()
        
        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256) if use_bn else nn.Identity()
        
        # 第五个卷积块
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512) if use_bn else nn.Identity()
        
        # 全连接层
        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 第四个卷积块
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # 第五个卷积块
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # 全连接层
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCNN(nn.Module):
    """带有残差连接的CNN模型"""
    def __init__(self, dropout_rate=0.5):
        super(ResNetCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 全连接层（修正输入维度：256 * 8 * 8）
        self.fc = nn.Linear(256 * 8 * 8, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32
        out = self.layer1(out)                  # 32x32
        out = self.layer2(out)                  # 16x16
        out = self.layer3(out)                  # 8x8
        out = torch.flatten(out, 1)             # 256*8*8
        out = self.dropout(out)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

#====================== 3. 训练函数 ======================

def train(model, train_loader, optimizer, scheduler, epoch, history):
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
    
    # 如果使用ReduceLROnPlateau，需要在每个epoch结束时更新
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(train_loss)
    elif isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
        pass  # OneCycleLR在每个batch后更新，不在epoch结束时更新
    else:
        scheduler.step()
    
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

def plot_training_history(history, title):
    """
    绘制训练历史曲线
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['test_loss'], 'r-', label='测试损失')
    plt.title(f'{title} - 训练和测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
    plt.plot(epochs, history['test_acc'], 'r-', label='测试准确率')
    plt.title(f'{title} - 训练和测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_models(histories, titles):
    """
    比较不同模型的训练历史
    """
    epochs = range(1, len(histories[0]['train_loss']) + 1)
    
    plt.figure(figsize=(12, 8))
    
    # 比较训练损失
    plt.subplot(2, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(epochs, history['train_loss'], label=titles[i])
    plt.title('训练损失比较')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 比较测试损失
    plt.subplot(2, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(epochs, history['test_loss'], label=titles[i])
    plt.title('测试损失比较')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 比较训练准确率
    plt.subplot(2, 2, 3)
    for i, history in enumerate(histories):
        plt.plot(epochs, history['train_acc'], label=titles[i])
    plt.title('训练准确率比较')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    
    # 比较测试准确率
    plt.subplot(2, 2, 4)
    for i, history in enumerate(histories):
        plt.plot(epochs, history['test_acc'], label=titles[i])
    plt.title('测试准确率比较')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

#====================== 6. 实验函数 ======================

def run_experiment(model_type, scheduler_type, epochs=10, batch_size=64, lr=0.01):
    """
    运行一个实验：训练指定的模型和调度器
    """
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
    if model_type == 'basic':
        model = BasicCNN(use_bn=True).to(device)
        model_name = "基础CNN"
    elif model_type == 'deep':
        model = DeepCNN(use_bn=True).to(device)
        model_name = "深层CNN"
    elif model_type == 'resnet':
        model = ResNetCNN().to(device)
        model_name = "残差CNN"
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
    
    # 创建优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # 创建学习率调度器
    if scheduler_type == 'step':
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        scheduler_name = "StepLR"
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        scheduler_name = "ReduceLROnPlateau"
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        scheduler_name = "CosineAnnealingLR"
    elif scheduler_type == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=lr*10, epochs=epochs, steps_per_epoch=len(train_loader))
        scheduler_name = "OneCycleLR"
    else:
        raise ValueError(f"未知的调度器类型: {scheduler_type}")
    
    # 训练和测试
    print(f"开始训练 {model_name} 使用 {scheduler_name}，设备: {device}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, scheduler, epoch, history)
        test_loss, test_acc = test(model, test_loader, history)
    
    # 返回训练历史和实验标题
    return history, f"{model_name} + {scheduler_name}"

#====================== 7. 主函数 ======================

def main():
    # 超参数设置
    epochs = 10
    batch_size = 64
    lr = 0.01
    
    # 运行不同的实验
    experiments = [
        # 比较不同的CNN结构（使用相同的调度器）
        ('basic', 'cosine'),
        ('deep', 'cosine'),
        ('resnet', 'cosine'),
        
        # 比较不同的调度器（使用相同的CNN结构）
        # ('basic', 'step'),
        # ('basic', 'plateau'),
        # ('basic', 'cosine'),
        # ('basic', 'onecycle'),
    ]
    
    histories = []
    titles = []
    
    for model_type, scheduler_type in experiments:
        history, title = run_experiment(model_type, scheduler_type, epochs, batch_size, lr)
        histories.append(history)
        titles.append(title)
    
    # 比较不同实验的结果
    compare_models(histories, titles)

if __name__ == '__main__':
    main()

"""
实验说明：

1. 模型结构比较
- BasicCNN: 3个卷积层的基础模型
- DeepCNN: 5个卷积层的深层模型
- ResNetCNN: 带有残差连接的模型

2. 学习率调度器比较
- StepLR: 按步长降低学习率
- ReduceLROnPlateau: 当指标不再改善时降低学习率
- CosineAnnealingLR: 余弦退火调整学习率
- OneCycleLR: 一个周期的学习率策略

3. 如何使用
- 默认比较不同的CNN结构（使用相同的余弦退火调度器）
- 取消注释相应的代码可以比较不同的调度器（使用相同的基础CNN）
- 可以修改epochs、batch_size和lr参数来调整训练过程

4. 观察重点
- 不同模型的收敛速度
- 最终的测试准确率
- 是否出现过拟合
- 学习率调度器对训练过程的影响
"""