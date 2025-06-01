"""
DAY 41 简单CNN - 基础示例

这个文件提供了一个简单的CNN示例，展示了如何：
1. 修改CNN结构
2. 使用不同的学习率调度器
3. 观察训练效果的变化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子和设备
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#====================== 1. 数据加载 ======================

def load_data(batch_size=64, use_augmentation=True):
    """
    加载CIFAR-10数据集
    参数：
        batch_size: 批次大小
        use_augmentation: 是否使用数据增强
    """
    if use_augmentation:
        # 使用数据增强
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),    # 随机水平翻转
            transforms.RandomRotation(10),        # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # 不使用数据增强
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # 测试集转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据集
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

#====================== 2. 模型定义 ======================

class SimpleCNN(nn.Module):
    """
    简单的CNN模型，可以通过参数调整结构
    参数：
        num_conv_layers: 卷积层数量
        channels: 每层的通道数列表
        use_bn: 是否使用Batch Normalization
        dropout_rate: Dropout比率
    """
    def __init__(self, num_conv_layers=2, channels=[16, 32, 64], use_bn=False, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        self.use_bn = use_bn
        
        # 创建卷积层
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # 输入图像是RGB三通道
        
        for i in range(num_conv_layers):
            # 添加卷积层
            self.conv_layers.append(
                nn.Conv2d(in_channels, channels[i], kernel_size=3, padding=1)
            )
            
            # 添加BN层（如果使用）
            if use_bn:
                self.conv_layers.append(nn.BatchNorm2d(channels[i]))
            
            in_channels = channels[i]
        
        # 计算全连接层的输入维度
        # 假设输入图像是32x32，每经过一次池化层大小减半
        final_size = 32 // (2 ** num_conv_layers)
        fc_input = channels[-1] * final_size * final_size
        
        # 全连接层
        self.fc1 = nn.Linear(fc_input, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 保存特征图用于可视化
        self.feature_maps = []
        
        # 通过卷积层
        for i in range(0, len(self.conv_layers), 2 if self.use_bn else 1):
            x = self.conv_layers[i](x)  # 卷积
            if self.use_bn:
                x = self.conv_layers[i+1](x)  # BN
            x = F.relu(x)  # 激活函数
            x = F.max_pool2d(x, 2)  # 池化
            self.feature_maps.append(x)  # 保存特征图
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

#====================== 3. 训练函数 ======================

def train_model(model, train_loader, test_loader, optimizer, scheduler, epochs=5):
    """
    训练模型并记录历史
    """
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(1, epochs + 1):
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
                      f'Accuracy: {100. * correct / total:.2f}%\t'
                      f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # 记录训练指标
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(100. * correct / total)
        
        # 测试
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
        
        # 记录测试指标
        history['test_loss'].append(test_loss)
        history['test_acc'].append(accuracy)
        
        print(f'Epoch {epoch}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # 更新学习率
        scheduler.step()
    
    return history

#====================== 4. 可视化函数 ======================

def plot_history(history, title):
    """
    绘制训练历史
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['test_loss'], label='测试损失')
    plt.title(f'{title} - 损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['test_acc'], label='测试准确率')
    plt.title(f'{title} - 准确率曲线')
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
    
    # 显示原始图像和特征图
    plt.figure(figsize=(15, 5))
    
    # 显示原始图像
    plt.subplot(1, len(model.feature_maps) + 1, 1)
    img = images[0] / 2 + 0.5  # 反归一化
    plt.imshow(img.permute(1, 2, 0))
    plt.title('原始图像')
    plt.axis('off')
    
    # 显示每层的特征图
    for i, feature_map in enumerate(model.feature_maps, 1):
        plt.subplot(1, len(model.feature_maps) + 1, i + 1)
        # 显示第一个特征图
        plt.imshow(feature_map[0, 0].cpu(), cmap='viridis')
        plt.title(f'特征图 {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

#====================== 5. 主函数 ======================

def main():
    # 基础配置
    batch_size = 64
    epochs = 5
    lr = 0.01
    
    # 加载数据
    print("加载数据...")
    train_loader, test_loader = load_data(batch_size, use_augmentation=True)
    
    # 实验1：基础CNN（3层）
    print("\n实验1：训练基础CNN（3层）...")
    model1 = SimpleCNN(num_conv_layers=3, channels=[32, 64, 128], use_bn=True).to(device)
    optimizer1 = optim.SGD(model1.parameters(), lr=lr, momentum=0.9)
    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.1)
    history1 = train_model(model1, train_loader, test_loader, optimizer1, scheduler1, epochs)
    plot_history(history1, "基础CNN（3层）+ StepLR")
    visualize_feature_maps(model1, test_loader)
    
    # 实验2：深层CNN（4层）
    print("\n实验2：训练深层CNN（4层）...")
    model2 = SimpleCNN(num_conv_layers=4, channels=[32, 64, 128, 256], use_bn=True).to(device)
    optimizer2 = optim.SGD(model2.parameters(), lr=lr, momentum=0.9)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=epochs)
    history2 = train_model(model2, train_loader, test_loader, optimizer2, scheduler2, epochs)
    plot_history(history2, "深层CNN（4层）+ CosineAnnealingLR")
    visualize_feature_maps(model2, test_loader)

if __name__ == '__main__':
    main()

"""
学习要点：

1. CNN结构修改
- 可以通过修改num_conv_layers和channels参数来改变网络深度和宽度
- use_bn参数控制是否使用Batch Normalization
- dropout_rate参数调整Dropout比率

2. 学习率调度器选择
- StepLR：按固定步长降低学习率
- CosineAnnealingLR：余弦周期调整学习率

3. 观察重点
- 不同深度的网络收敛速度
- 是否出现过拟合（训练准确率高但测试准确率低）
- 特征图的变化

4. 实验建议
- 尝试不同的网络深度（修改num_conv_layers和channels）
- 对比有无Batch Normalization的效果（修改use_bn）
- 测试不同的学习率调度策略
- 观察数据增强的影响（修改use_augmentation）
"""