"""
DAY 39 图像数据与显存

本节主要介绍深度学习中的图像数据处理和显存管理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
 # 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 设置随机种子确保结果可复现
torch.manual_seed(42)

#====================== 1. 图像数据的格式 ======================
"""
1.1 图像数据与结构化数据的区别：
- 结构化数据（表格数据）形状：(样本数, 特征数)，如(1000, 5)
- 图像数据需要保留空间信息，形状更复杂：(通道数, 高度, 宽度)

1.2 图像数据的两种主要格式：
- 灰度图像：单通道，如MNIST数据集 (1, 28, 28)
- 彩色图像：三通道(RGB)，如CIFAR-10数据集 (3, 32, 32)
"""

# 定义数据处理步骤
transforms = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化处理
])

# 加载CIFAR-10数据集作为示例
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#====================== 2. 模型的定义 ======================
"""
为了演示显存占用，我们定义一个简单的CNN模型
"""

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层：输入3通道，输出6通道，卷积核5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 第二个卷积层：输入6通道，输出16通道，卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积层 -> ReLU -> 最大池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 将特征图展平
        x = x.view(-1, 16 * 5 * 5)
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#====================== 3. 显存占用分析 ======================
"""
3.1 模型参数与梯度参数
- 每个参数需要存储值和梯度
- 使用float32类型，每个数占4字节
"""
model = SimpleCNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型总参数量：{total_params}")
print(f"参数占用显存：{total_params * 4 / 1024 / 1024:.2f} MB")

"""
3.2 优化器参数
- 如Adam优化器会为每个参数存储额外状态（如动量）
- 通常是参数量的2-3倍
"""
optimizer = torch.optim.Adam(model.parameters())
print(f"优化器额外占用显存：{total_params * 8 / 1024 / 1024:.2f} MB")

"""
3.3 数据批量所占显存
- 与batch_size成正比
- 需要考虑输入数据和中间特征图
"""
# 计算单个CIFAR-10图像占用
single_image_size = 3 * 32 * 32 * 4  # 通道*高*宽*字节数
print(f"单张图像占用：{single_image_size / 1024:.2f} KB")
print(f"batch_size=4时占用：{single_image_size * 4 / 1024:.2f} KB")
print(f"batch_size=64时占用：{single_image_size * 64 / 1024 / 1024:.2f} MB")

"""
3.4 神经元输出中间状态
- 前向传播时的特征图
- 反向传播需要的中间结果
- 通常比输入数据大很多
"""

#====================== 4. batch_size与训练的关系 ======================
"""
4.1 batch_size的影响：
- 较大的batch_size：
  * 计算效率更高
  * 梯度估计更准确
  * 需要更多显存
  * 可能导致泛化性能下降
  
- 较小的batch_size：
  * 训练更慢
  * 梯度估计噪声大
  * 需要更少显存
  * 可能有更好的泛化性能
  
4.2 选择合适的batch_size：
- 从小值开始（如16）
- 逐渐增加直到接近显存限制
- 通常设置为显存上限的80%
- 需要在训练效率和模型性能之间权衡
"""

# 展示一张样例图片
def show_sample_image():
    sample_idx = torch.randint(0, len(trainset), size=(1,)).item()
    image, label = trainset[sample_idx]
    
    print(f"图片形状: {image.shape}")
    print(f"类别: {classes[label]}")
    
    # 显示图片
    img = image / 2 + 0.5     # 反标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'类别: {classes[label]}')
    plt.show()

# 显示样例图片
show_sample_image()

"""
总结：
1. 图像数据需要特殊的预处理和格式转换
2. 显存管理是深度学习中的重要问题
3. batch_size的选择需要综合考虑多个因素
4. 合理的显存管理可以提高训练效率
"""