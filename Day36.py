# ------------------- 导入工具库 -------------------
# pandas: 用于读取和处理表格数据（类似Excel的操作）
# numpy: 用于高效数值计算（比如矩阵运算）
# torch: PyTorch深度学习框架（核心库，用于搭建和训练神经网络）
# sklearn: 机器学习工具库（这里用数据划分、标准化等）
# time: 用于记录训练时间
# matplotlib: 可视化图表（绘制损失、准确率曲线）
# tqdm: 进度条工具（让训练过程更直观）
# imblearn: 处理数据不平衡（这里暂时未用，但先导入）
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

# ------------------- 设备配置（GPU/CPU） -------------------
# 检查是否有可用的GPU：如果有则用GPU加速训练（速度更快），否则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")  # 打印当前使用的设备（确认是否启用GPU）
 
# ------------------- 加载并清洗数据 -------------------
# 加载信贷预测数据集（假设data.csv在当前目录下）
# 数据包含用户信息（如收入、工作年限）和标签（是否违约：Credit Default）
data = pd.read_csv(r'python60-days-challenge\1_python-learning-library\data.csv')
 
# 丢弃无用的Id列（Id是用户唯一标识，与信贷违约无关）
data = data.drop(['Id'], axis=1)  # axis=1表示按列删除
 
# 区分连续特征（数值型）和离散特征（文本型/类别型）
# 连续特征：比如年龄、收入（可以取任意数值）
# 离散特征：比如职业、教育程度（只能取有限的类别）
continuous_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()  # 数值列
discrete_features = data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()  # 非数值列
 
# 离散特征用众数（出现次数最多的值）填充缺失值
# 例：如果"职业"列有缺失，用出现最多的职业填充
for feature in discrete_features:
    if data[feature].isnull().sum() > 0:  # 检查是否有缺失值
        mode_value = data[feature].mode()[0]  # 计算众数
        data[feature].fillna(mode_value, inplace=True)  # 填充缺失值
 
# 连续特征用中位数（中间位置的数）填充缺失值
# 例：如果"收入"列有缺失，用所有收入的中间值填充（比平均数更抗异常值）
for feature in continuous_features:
    if data[feature].isnull().sum() > 0:
        median_value = data[feature].median()  # 计算中位数
        data[feature].fillna(median_value, inplace=True)
 
# ------------------- 离散特征编码（转成数值） -------------------
# 有顺序的离散特征（比如"工作年限"有"1年"<"2年"<"10+年"）用标签编码（转成数字）
mappings = {
    "Years in current job": {
        "10+ years": 10,  # "10+年"对应数字10（最大）
        "2 years": 2,     # "2年"对应数字2
        "3 years": 3,
        "< 1 year": 0,    # "<1年"对应数字0（最小）
        "5 years": 5,
        "1 year": 1,
        "4 years": 4,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9
    },
    "Home Ownership": {  # 房屋所有权（有顺序：租房 < 房贷 < 有房贷 < 自有房？）
        "Home Mortgage": 0,  # 房贷
        "Rent": 1,           # 租房
        "Own Home": 2,       # 自有房
        "Have Mortgage": 3   # 有房贷（可能顺序需要根据业务调整）
    },
    "Term": {  # 贷款期限（短期 < 长期）
        "Short Term": 0,  # 短期
        "Long Term": 1    # 长期
    }
}
 
# 使用映射字典将文本转成数字（标签编码）
data["Years in current job"] = data["Years in current job"].map(mappings["Years in current job"])
data["Home Ownership"] = data["Home Ownership"].map(mappings["Home Ownership"])
data["Term"] = data["Term"].map(mappings["Term"])
 
# 无顺序的离散特征（比如"贷款用途"：购车/教育/装修，彼此无大小关系）用独热编码
# 独热编码：将1列转成N列（N是类别数），每列用0/1表示是否属于该类别
data = pd.get_dummies(data, columns=['Purpose'])  # 对"Purpose"列做独热编码
 
# 独热编码后会生成新列（比如Purpose_购车、Purpose_教育），需要将这些列的类型从bool转成int（0/1）
list_final = []  # 存储新生成的列名
data2 = pd.read_csv(r'python60-days-challenge\1_python-learning-library\data.csv')  # 重新读取原始数据（对比列名）
for i in data.columns:
    if i not in data2.columns:  # 原始数据没有的列，就是新生成的独热列
        list_final.append(i)
for i in list_final:
    data[i] = data[i].astype(int)  # 将bool型（True/False）转成int（1/0）
 
# ------------------- 分离特征和标签 -------------------
X = data.drop(['Credit Default'], axis=1)  # 特征数据（所有列，除了标签列）
y = data['Credit Default']  # 标签数据（0=未违约，1=违约）
 
# 划分训练集（80%）和测试集（20%）：训练集用来学习规律，测试集验证模型效果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # random_state固定随机种子，保证结果可复现
 
# 特征标准化（将特征缩放到0-1区间，避免大数值特征"欺负"小数值特征）
scaler = MinMaxScaler()  # 创建MinMaxScaler（最小-最大标准化）
X_train = scaler.fit_transform(X_train)  # 用训练集拟合标准化参数并转换
X_test = scaler.transform(X_test)  # 用训练集的参数转换测试集（保证数据分布一致）
 
# 将数据转成PyTorch张量（神经网络只能处理张量数据），并移动到GPU（如果有）
# FloatTensor：32位浮点数（特征数据）
# LongTensor：64位整数（标签数据，分类任务需要）
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train.values).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test.values).to(device)
 
# ------------------- 定义神经网络模型 -------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()  # 调用父类构造函数（必须）
        # 全连接层1：输入30个特征（根据数据预处理后的列数确定），输出64个神经元
        self.fc1 = nn.Linear(30, 64)
        self.relu = nn.ReLU()  # 激活函数（引入非线性，让模型能学习复杂规律）
        self.dropout = nn.Dropout(0.3)  # Dropout层（随机丢弃30%的神经元，防止过拟合）
        # 全连接层2：输入64个神经元，输出32个神经元
        self.fc2 = nn.Linear(64, 32)
        # 全连接层3：输入32个神经元，输出2个类别（0=未违约，1=违约）
        self.fc3 = nn.Linear(32, 2)
 
    def forward(self, x):
        # 前向传播：数据从输入层→隐藏层→输出层的计算流程
        x = self.fc1(x)    # 输入层→隐藏层1：30→64
        x = self.relu(x)   # 激活函数（过滤负数值）
        x = self.dropout(x)  # 应用Dropout（防止过拟合）
        x = self.fc2(x)    # 隐藏层1→隐藏层2：64→32
        x = self.relu(x)   # 激活函数
        x = self.fc3(x)    # 隐藏层2→输出层：32→2（输出未归一化的分数）
        return x
 
# ------------------- 初始化模型、损失函数、优化器 -------------------
model = MLP().to(device)  # 实例化模型并移动到GPU
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（适合分类任务）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器（比SGD更智能，自动调整学习率）
 
# ------------------- 模型训练 -------------------
num_epochs = 20000  # 训练轮数（完整遍历训练集的次数）
losses = []         # 记录每200轮的训练损失
accuracies = []     # 记录每200轮的测试准确率
epochs = []         # 记录对应的轮数
start_time = time.time()  # 记录训练开始时间
 
# 创建tqdm进度条（可视化训练进度）
with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
    for epoch in range(num_epochs):
        # 前向传播：模型根据输入数据计算预测值
        outputs = model(X_train)  # 模型输出（形状：[训练样本数, 2]，表示每个样本属于2个类别的分数）
        loss = criterion(outputs, y_train)  # 计算损失（预测值与真实标签的差异，越小越好）
 
        # 反向传播和参数更新
        optimizer.zero_grad()  # 清空历史梯度（避免梯度累加）
        loss.backward()        # 反向传播计算梯度（自动求导）
        optimizer.step()       # 根据梯度更新模型参数（优化器核心操作）
 
        # 每200轮记录一次损失和准确率（避免记录太频繁影响速度）
        if (epoch + 1) % 200 == 0:
            losses.append(loss.item())  # loss.item()：将张量转成Python数值
            epochs.append(epoch + 1)    # 记录当前轮数
 
            # 在测试集上评估模型（不更新参数，只看效果）
            model.eval()  # 切换到评估模式（关闭Dropout，保证结果稳定）
            with torch.no_grad():  # 禁用梯度计算（节省内存，加速推理）
                test_outputs = model(X_test)  # 测试集预测值
                _, predicted = torch.max(test_outputs, 1)  # 取分数最高的类别作为预测结果
                correct = (predicted == y_test).sum().item()  # 计算预测正确的样本数
                accuracy = correct / y_test.size(0)  # 准确率 = 正确数 / 总样本数
                accuracies.append(accuracy)  # 记录准确率
 
            # 更新进度条显示的信息（当前损失和准确率）
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Accuracy': f'{accuracy * 100:.2f}%'})
 
        # 每1000轮更新一次进度条（避免进度条刷新太频繁）
        if (epoch + 1) % 1000 == 0:
            pbar.update(1000)  # 进度条前进1000步
 
    # 确保进度条最终显示100%（防止最后一轮未更新）
    if pbar.n < num_epochs:
        pbar.update(num_epochs - pbar.n)
 
# 计算总训练时间并打印
time_all = time.time() - start_time
print(f'Training time: {time_all:.2f} seconds')
 
# ------------------- 可视化训练结果 -------------------
# 创建双y轴图表（损失和准确率在同一张图显示）
fig, ax1 = plt.subplots(figsize=(10, 6))
 
# 绘制损失曲线（红色）
color = 'tab:red'
ax1.set_xlabel('Epoch')  # x轴：训练轮数
ax1.set_ylabel('Loss', color=color)  # y轴：损失值（红色）
ax1.plot(epochs, losses, color=color)  # 绘制损失随轮数变化的曲线
ax1.tick_params(axis='y', labelcolor=color)  # y轴标签颜色与曲线一致
 
# 创建第二个y轴（蓝色），用于绘制准确率
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)  # y轴：准确率（蓝色）
ax2.plot(epochs, accuracies, color=color)  # 绘制准确率随轮数变化的曲线
ax2.tick_params(axis='y', labelcolor=color)
 
plt.title('Training Loss and Accuracy over Epochs')  # 图表标题
plt.grid(True)  # 显示网格线（方便观察趋势）
plt.show()  # 显示图表
 
# ------------------- 最终测试集评估 -------------------
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    outputs = model(X_test)  # 测试集预测值
    _, predicted = torch.max(outputs, 1)  # 取预测类别（0或1）
    correct = (predicted == y_test).sum().item()  # 正确预测的样本数
    accuracy = correct / y_test.size(0)  # 计算准确率
    print(f'测试集准确率: {accuracy * 100:.2f}%')  # 打印准确率（百分比形式）