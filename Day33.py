# ------------------- 数据准备部分 -------------------
# 我们使用有 4 个特征、3 个类别的鸢尾花数据集作为本次训练的数据
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 

# 加载鸢尾花数据集
# load_iris() 是 sklearn 提供的函数，用于加载鸢尾花数据集
iris = load_iris()
# 提取数据集中的特征数据
X = iris.data  
# 提取数据集中的标签数据
y = iris.target  

# 将数据集划分为训练集和测试集
# test_size=0.2 表示将 20% 的数据作为测试集，random_state=42 用于保证结果可重复
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印训练集和测试集的特征和标签的尺寸，方便我们确认数据划分是否正确
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# 对数据进行归一化处理
# 神经网络对输入数据的尺度比较敏感，归一化是常见的处理方法
from sklearn.preprocessing import MinMaxScaler
# 创建 MinMaxScaler 实例，用于将数据缩放到 [0, 1] 区间
scaler = MinMaxScaler()
# 对训练集进行拟合和转换
X_train = scaler.fit_transform(X_train)
# 使用训练集的缩放规则对测试集进行转换
X_test = scaler.transform(X_test)

# 将数据转换为 PyTorch 张量，因为 PyTorch 模型使用张量进行训练
# y_train 和 y_test 是整数标签，需要转换为 long 类型
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# ------------------- 模型架构定义部分 -------------------
# 定义一个简单的全连接神经网络模型（Multi-Layer Perceptron，多层感知机）
# 包含一个输入层、一个隐藏层和一个输出层
class MLP(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的构造函数
        super(MLP, self).__init__()
        # 定义输入层到隐藏层的线性变换
        # 输入特征有 4 个，隐藏层有 10 个神经元
        self.fc1 = nn.Linear(4, 10)  
        # 定义激活函数 ReLU，增加模型的非线性能力
        self.relu = nn.ReLU()   
        # 定义隐藏层到输出层的线性变换
        # 隐藏层有 10 个神经元，输出有 3 个类别
        self.fc2 = nn.Linear(10,3) 
        # 输出层不需要激活函数，因为后续使用的交叉熵损失函数内部包含了 softmax 函数

    def forward(self, x):
        # 输入数据经过输入层到隐藏层的线性变换
        out = self.fc1(x)
        # 经过 ReLU 激活函数处理
        out = self.relu(out)
        # 经过隐藏层到输出层的线性变换
        out = self.fc2(out)
        return out

# ------------------- 模型训练部分 -------------------
# 实例化模型
model = MLP()

# 定义损失函数和优化器
# 对于分类问题，我们使用交叉熵损失函数
criterion = nn.CrossEntropyLoss() 

# 使用随机梯度下降（SGD）优化器，学习率设置为 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01) 

# 开始训练模型
# 定义训练的轮数
num_epochs = 20000 
# 用于存储每一轮训练的损失值
losses = []

for epoch in range(num_epochs):
    # 前向传播
    # 调用模型的 forward 方法，计算模型的输出
    outputs = model.forward(X_train) 
    # 计算损失值，outputs 是模型的输出，y_train 是真实标签
    loss = criterion(outputs, y_train) 

    # 反向传播和优化
    # 清空优化器中的梯度信息，因为 PyTorch 会累加梯度
    optimizer.zero_grad() 
    # 反向传播计算梯度
    loss.backward() 
    # 根据计算得到的梯度更新模型的参数
    optimizer.step() 

    # 记录当前轮的损失值
    # loss.item() 用于将损失值从张量转换为 Python 标量
    losses.append(loss.item()) 

    # 每训练 10000 轮，打印一次训练信息
    if (epoch + 1) % 10000 == 0: 
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}') 

# ------------------- 可视化结果部分 -------------------
# 导入 matplotlib 库用于可视化
import matplotlib.pyplot as plt

# 绘制训练损失曲线
plt.plot(range(num_epochs), losses)
# 设置 x 轴标签
plt.xlabel('Epoch')
# 设置 y 轴标签
plt.ylabel('Loss')
# 设置图表标题
plt.title('Training Loss')
# 显示图表
plt.show()
