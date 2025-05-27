import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision  # 添加torchvision导入
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 尝试导入tqdm库，如果不可用则使用简单的打印
try:
    from tqdm import tqdm
    has_tqdm = True
except ImportError:
    has_tqdm = False
    print("提示: 安装tqdm库可以显示进度条 (pip install tqdm)")

print("开始执行脚本...")
print("检查CUDA是否可用:", "可用" if torch.cuda.is_available() else "不可用")

# 定义转换器：将图片转换为张量，并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("开始下载CIFAR-10数据集（如果需要）...")
# 下载并加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
print("CIFAR-10数据集加载完成！")

# CIFAR-10数据集的类别
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 定义一个函数来显示和保存图片
def imshow(img, filename=None):
    # 反归一化
    img = img / 2 + 0.5
    # 将张量转换为numpy数组
    npimg = img.numpy()
    # 转置维度，从(channels, height, width)变为(height, width, channels)
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # 不显示坐标轴
    
    # 保存图片到文件
    if filename:
        plt.savefig(filename)
        print(f"图片已保存到: {filename}")
    
    # 尝试显示图片
    try:
        plt.show()
    except Exception as e:
        print(f"显示图片时出错: {e}")

# 创建保存图片的目录
output_dir = './cifar_images'
os.makedirs(output_dir, exist_ok=True)
print(f"图片将保存在: {output_dir}")

# 获取数据集中的第一张图片
image, label = trainset[0]

print(f'这是一张{classes[label]}的图片')
# 显示并保存图片
imshow(image, f'{output_dir}/cifar_single_image.png')

# 演示Dataset类的特殊方法
print(f'\nCIFAR-10数据集的大小: {len(trainset)}')  # 使用__len__方法
print(f'图片的形状: {image.shape}')  # 形状应该是[3, 32, 32]，表示3通道，32x32像素

# 演示DataLoader的使用
print("创建DataLoader...")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=0)  # 减少worker数量避免潜在问题

# 获取一个批次的图片并显示
print("从DataLoader获取一批数据...")
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 显示一个批次的图片
print('\n显示一个批次(4张)的图片:')
imshow(torchvision.utils.make_grid(images), f'{output_dir}/cifar_batch_images.png')
print('这些图片的类别是: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

print("\n脚本执行完成！")