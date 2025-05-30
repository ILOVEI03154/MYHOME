"""
DAY 40 训练和测试的规范写法

本节介绍深度学习中训练和测试的规范写法，包括：
1. 训练和测试函数的封装
2. 展平操作
3. dropout的使用
4. 训练过程可视化
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

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#====================== 1. 数据加载 ======================

def load_data(batch_size=64, is_train=True):
    """
    加载CIFAR-10数据集
    Args:
        batch_size: 批次大小
        is_train: 是否为训练集
    Returns:
        dataloader: 数据加载器
    """
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
        shuffle=is_train,  # 训练集打乱，测试集不打乱
        num_workers=2
    )
    
    return dataloader

#====================== 2. 模型定义 ======================

class SimpleNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SimpleNet, self).__init__()
        # 修改第一层卷积的输入通道为3（彩色图像）
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout_rate)  # 2D dropout用于卷积层
        self.dropout2 = nn.Dropout(dropout_rate)    # 1D dropout用于全连接层
        
        # 展平后的特征图大小计算：
        # 原始图像: 32x32
        # conv1: (32-3+1)x(32-3+1) = 30x30
        # maxpool: 15x15
        # conv2: (15-3+1)x(15-3+1) = 13x13
        # maxpool: 6x6
        # 因此全连接层输入大小为: 64 * 6 * 6
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # 训练时随机丢弃，测试时自动关闭
        
        # 展平操作：保留batch_size维度，其余维度展平
        x = torch.flatten(x, 1)  # 等价于 x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

#====================== 3. 训练函数 ======================

def train(model, train_loader, optimizer, epoch, history):
    """
    训练一个epoch
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前epoch数
        history: 记录训练历史的字典
    Returns:
        epoch_loss: 当前epoch的平均损失
        epoch_acc: 当前epoch的准确率
    """
    model.train()  # 设置为训练模式，启用dropout
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()  # 清空梯度
        output = model(data)   # 前向传播
        loss = F.nll_loss(output, target)  # 计算损失
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
        
        train_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]  # 获取最大概率的索引
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\t'
                  f'Accuracy: {100. * correct / total:.2f}%')
    
    # 计算epoch的平均损失和准确率
    epoch_loss = train_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # 记录训练历史
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    
    return epoch_loss, epoch_acc

#====================== 4. 测试函数 ======================

def test(model, test_loader, history):
    """
    在测试集上评估模型
    Args:
        model: 模型
        test_loader: 测试数据加载器
        history: 记录训练历史的字典
    Returns:
        test_loss: 测试集上的平均损失
        accuracy: 测试集上的准确率
    """
    model.eval()  # 设置为评估模式，关闭dropout
    test_loss = 0
    correct = 0
    
    with torch.no_grad():  # 测试时不需要计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # 记录测试历史
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
    Args:
        history: 包含训练和测试历史数据的字典
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建一个包含两个子图的图表
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, history['test_loss'], 'r-', label='测试损失')
    plt.title('训练和测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
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

def visualize_predictions(model, test_loader, num_samples=5):
    """
    可视化模型预测结果
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        num_samples: 要显示的样本数量
    """
    model.eval()
    
    # 获取一批数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    # 显示图像和预测结果
    fig = plt.figure(figsize=(12, 3))
    for idx in range(num_samples):
        ax = fig.add_subplot(1, num_samples, idx + 1)
        img = images[idx] / 2 + 0.5  # 反标准化
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(f'预测: {classes[predicted[idx]]}\n实际: {classes[labels[idx]]}',
                    color=('green' if predicted[idx] == labels[idx] else 'red'))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#====================== 6. 主函数 ======================

# CIFAR-10数据集的类别
classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

def main():
    # 超参数设置
    batch_size = 64
    epochs = 9
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
    
    # 训练和测试
    print(f"开始训练，使用设备: {device}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, epoch, history)
        test_loss, test_acc = test(model, test_loader, history)
    
    # 可视化训练过程
    print("训练完成，绘制训练历史...")
    plot_training_history(history)
    
    # 可视化预测结果
    print("可视化模型预测结果...")
    visualize_predictions(model, test_loader)

if __name__ == '__main__':
    main()

"""
重点说明：

1. 训练和测试的区别：
   - 训练时：model.train()，启用dropout
   - 测试时：model.eval()，关闭dropout
   
2. 展平操作：
   - torch.flatten(x, 1) 或 x.view(x.size(0), -1)
   - 保留第一维度（batch_size），其余维度展平
   
3. dropout的使用：
   - 训练阶段：随机丢弃神经元，防止过拟合
   - 测试阶段：自动关闭dropout，使用完整网络
   
4. 规范写法的优点：
   - 代码结构清晰，便于维护
   - 功能模块化，易于复用
   - 训练过程可控，便于调试
   - 适用于不同的数据集和模型
"""