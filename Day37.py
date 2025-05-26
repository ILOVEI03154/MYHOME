# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import time
# import matplotlib.pyplot as plt
# from tqdm import tqdm  # 导入tqdm库用于进度条显示
# import warnings
# warnings.filterwarnings("ignore")  # 忽略警告信息

# # 设置GPU设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# # 加载鸢尾花数据集
# iris = load_iris()
# X = iris.data  # 特征数据
# y = iris.target  # 标签数据

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 归一化数据
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # 将数据转换为PyTorch张量并移至GPU
# X_train = torch.FloatTensor(X_train).to(device)
# y_train = torch.LongTensor(y_train).to(device)
# X_test = torch.FloatTensor(X_test).to(device)
# y_test = torch.LongTensor(y_test).to(device)

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

# # 实例化模型并移至GPU
# model = MLP().to(device)

# # 分类问题使用交叉熵损失函数
# criterion = nn.CrossEntropyLoss()

# # 使用随机梯度下降优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 训练模型
# num_epochs = 20000  # 训练的轮数

# # 用于存储每100个epoch的损失值和对应的epoch数
# losses = []
# epochs = []

# start_time = time.time()  # 记录开始时间

# # 创建tqdm进度条
# with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
#     # 训练模型
#     for epoch in range(num_epochs):
#         # 前向传播
#         outputs = model(X_train)  # 隐式调用forward函数
#         loss = criterion(outputs, y_train)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # 记录损失值并更新进度条
#         if (epoch + 1) % 200 == 0:
#             losses.append(loss.item())
#             epochs.append(epoch + 1)
#             # 更新进度条的描述信息
#             pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

#         # 每1000个epoch更新一次进度条
#         if (epoch + 1) % 1000 == 0:
#             pbar.update(1000)  # 更新进度条

#     # 确保进度条达到100%
#     if pbar.n < num_epochs:
#         pbar.update(num_epochs - pbar.n)  # 计算剩余的进度并更新

# time_all = time.time() - start_time  # 计算训练时间
# print(f'Training time: {time_all:.2f} seconds')

# # 可视化损失曲线
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss over Epochs')
# plt.grid(True)
# plt.show()

# # 在测试集上评估模型，此时model内部已经是训练好的参数了
# # 评估模型
# model.eval() # 设置模型为评估模式
# with torch.no_grad(): # torch.no_grad()的作用是禁用梯度计算，可以提高模型推理速度
#     outputs = model(X_test)  # 对测试数据进行前向传播，获得预测结果
#     _, predicted = torch.max(outputs, 1) # torch.max(outputs, 1)返回每行的最大值和对应的索引
#     #这个函数返回2个值，分别是最大值和对应索引，参数1是在第1维度（行）上找最大值，_ 是Python的约定，表示忽略这个返回值，所以这个写法是找到每一行最大值的下标
#     # 此时outputs是一个tensor，p每一行是一个样本，每一行有3个值，分别是属于3个类别的概率，取最大值的下标就是预测的类别


#     # predicted == y_test判断预测值和真实值是否相等，返回一个tensor，1表示相等，0表示不等，然后求和，再除以y_test.size(0)得到准确率
#     # 因为这个时候数据是tensor，所以需要用item()方法将tensor转化为Python的标量
#     # 之所以不用sklearn的accuracy_score函数，是因为这个函数是在CPU上运行的，需要将数据转移到CPU上，这样会慢一些
#     # size(0)获取第0维的长度，即样本数量

#     correct = (predicted == y_test).sum().item() # 计算预测正确的样本数
#     accuracy = correct / y_test.size(0)
#     print(f'测试集准确率: {accuracy * 100:.2f}%')

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import time
# import matplotlib.pyplot as plt
# from tqdm import tqdm  # 导入tqdm库用于进度条显示
# import warnings
# warnings.filterwarnings("ignore")  # 忽略警告信息

# # 设置GPU设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# print(f"使用设备: {device}")

# # 加载鸢尾花数据集
# iris = load_iris()
# X = iris.data  # 特征数据
# y = iris.target
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # 将数据转换为PyTorch张量并移至GPU
# X_train = torch.FloatTensor(X_train).to(device)
# y_train = torch.LongTensor(y_train).to(device)  
# X_test = torch.FloatTensor(X_test).to(device)
# y_test = torch.LongTensor(y_test).to(device)

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
#         self.relu = nn.ReLU()   
#         self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

# # 实例化模型并移至GPU
# model = MLP().to(device)
# criterion = nn.CrossEntropyLoss()  # 分类问题使用交叉熵损失函数
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

# # 训练模型
# num_epochs = 20000  # 训练的轮数
# # 用于存储每100个epoch的损失值和对应的epoch数
# train_losses = []  # 存储训练集损失
# test_losses = []  # 新增：存储测试集损失
# epochs = []
# start_time = time.time()  # 记录开始时间

# # 创建tqdm进度条
# with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
#     # 训练模型
#     for epoch in range(num_epochs):
#         # 前向传播
#         outputs = model(X_train)  # 隐式调用forward函数
#         train_loss = criterion(outputs, y_train)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()

#         # 记录损失值并更新进度条
#         if (epoch + 1) % 200 == 0:
#             # 计算测试集损失，新增代码
#             model.eval()  # 设置模型为评估模式
#             with torch.no_grad():  # torch.no_grad()的作用是禁用梯度计算，可以提高模型推理速度
#                 test_outputs = model(X_test)
#                 test_loss = criterion(test_outputs, y_test)
#             model.train()  # 恢复模型为训练模式

#             train_losses.append(train_loss.item())  # 存储训练集损失
#             test_losses.append(test_loss.item())  # 存储测试集损失
#             epochs.append(epoch + 1)  # 存储对应的epoch数

#             # 更新进度条的描述信息
#             pbar.set_postfix({'Train Loss': f'{train_loss.item():.4f}', 'Test Loss': f'{test_loss.item():.4f}'})

#         # 每1000个epoch更新一次进度条
#         if (epoch + 1) % 1000 == 0:
#             pbar.update(1000)  # 更新进度条

#     # 确保进度条达到100%
#     if pbar.n < num_epochs:
#         pbar.update(num_epochs - pbar.n)  # 计算剩余的进度并更新

# time_all = time.time() - start_time  # 计算训练时间
# print(f'Training time: {time_all:.2f} seconds')

# # 可视化损失曲线
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_losses, label='Train Loss')  # 原始代码已有
# plt.plot(epochs, test_losses, label='Test Loss')  # 新增：测试集损失曲线
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Loss over Epochs')
# plt.legend()  # 新增：显示图例
# plt.grid(True)
# plt.show()

# # 在测试集上评估模型，此时model内部已经是训练好的参数了
# # 评估模型
# model.eval()  # 设置模型为评估模式
# with torch.no_grad():  # torch.no_grad()的作用是禁用梯度计算，可以提高模型推理速度
#     outputs = model(X_test)  # 对测试数据进行前向传播，获得预测结果 
#     _, predicted = torch.max(outputs, 1)  # torch.max(outputs, 1)返回每行的最大值和对应的索引
#     correct = (predicted == y_test).sum().item()  # 计算预测正确的样本数
#     accuracy = correct / y_test.size(0)  # 计算准确率
#     print(f'测试集准确率: {accuracy * 100:.2f}%')



# # 模型加载与保存
# # 仅保存模型参数
# torch.save(model.state_dict(), 'model_weights.pth')  # 保存模型参数到文件
# print("模型参数已保存到 'model_weights.pth'")

# # 加载模型参数
# model = MLP().to(device)  # 重新实例化模型
# model.load_state_dict(torch.load('model_weights.pth'))  # 加载模型参数


# # 保存模型+权重
# # 保存模型结构及参数
# # 加载时无需提前定义模型类
# # 文件体积大，依赖训练时的代码环境（如自定义层可能报错）
# torch.save(model,'model.pth')  # 保存模型结构及参数到文件
# print("模型结构及参数已保存到'model.pth'")

# # 加载模型结构及参数
# model = torch.load('model.pth')  # 加载模型结构及参数
# model.eval()  # 设置模型为评估模式

# 保存训练状态（断点续训）
# - 原理：保存模型参数、优化器状态（学习率、动量）、训练轮次、损失值等完整训练状态，用于中断后继续训练。
# - 适用场景：长时间训练任务（如分布式训练、算力中断）。







# 早停策略
# - 正常情况：训练集和测试集损失同步下降，最终趋于稳定。

# - 过拟合：训练集损失持续下降，但测试集损失在某一时刻开始上升（或不再下降）。

# 如果可以监控验证集的指标不再变好，此时提前终止训练，避免模型对训练集过度拟合。----监控的对象是验证集的指标。
# 这种策略叫早停法。

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import time
# import matplotlib.pyplot as plt
# from tqdm import tqdm  # 导入tqdm库用于进度条显示
# import warnings
# warnings.filterwarnings("ignore")  # 忽略警告信息

# # 设置GPU设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# # 加载鸢尾花数据集
# iris = load_iris()
# X = iris.data  # 特征数据
# y = iris.target  # 标签数据

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 归一化数据
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # 将数据转换为PyTorch张量并移至GPU
# X_train = torch.FloatTensor(X_train).to(device)
# y_train = torch.LongTensor(y_train).to(device)
# X_test = torch.FloatTensor(X_test).to(device)
# y_test = torch.LongTensor(y_test).to(device)

# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(4, 10)  # 输入层到隐藏层
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(10, 3)  # 隐藏层到输出层

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

# # 实例化模型并移至GPU
# model = MLP().to(device)

# # 分类问题使用交叉熵损失函数
# criterion = nn.CrossEntropyLoss()

# # 使用随机梯度下降优化器
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 训练模型
# num_epochs = 20000  # 训练的轮数

# # 用于存储每200个epoch的损失值和对应的epoch数
# train_losses = []  # 存储训练集损失
# test_losses = []   # 存储测试集损失
# epochs = []

# # ===== 新增早停相关参数 =====
# best_test_loss = float('inf')  # 记录最佳测试集损失
# best_epoch = 0                 # 记录最佳epoch
# patience = 50                # 早停耐心值（连续多少轮测试集损失未改善时停止训练）
# counter = 0                    # 早停计数器
# early_stopped = False          # 是否早停标志
# # ==========================

# start_time = time.time()  # 记录开始时间

# # 创建tqdm进度条
# with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
#     # 训练模型
#     for epoch in range(num_epochs):
#         # 前向传播
#         outputs = model(X_train)  # 隐式调用forward函数
#         train_loss = criterion(outputs, y_train)

#         # 反向传播和优化
#         optimizer.zero_grad()
#         train_loss.backward()
#         optimizer.step()

#         # 记录损失值并更新进度条
#         if (epoch + 1) % 200 == 0:
#             # 计算测试集损失
#             model.eval()
#             with torch.no_grad():
#                 test_outputs = model(X_test)
#                 test_loss = criterion(test_outputs, y_test)
#             model.train()
            
#             train_losses.append(train_loss.item())
#             test_losses.append(test_loss.item())
#             epochs.append(epoch + 1)
            
#             # 更新进度条的描述信息
#             pbar.set_postfix({'Train Loss': f'{train_loss.item():.4f}', 'Test Loss': f'{test_loss.item():.4f}'})
            
#             # ===== 新增早停逻辑 =====
#             if test_loss.item() < best_test_loss: # 如果当前测试集损失小于最佳损失
#                 best_test_loss = test_loss.item() # 更新最佳损失
#                 best_epoch = epoch + 1 # 更新最佳epoch
#                 counter = 0 # 重置计数器
#                 # 保存最佳模型
#                 torch.save(model.state_dict(), 'best_model.pth')
#             else:
#                 counter += 1
#                 if counter >= patience:
#                     print(f"早停触发！在第{epoch+1}轮，测试集损失已有{patience}轮未改善。")
#                     print(f"最佳测试集损失出现在第{best_epoch}轮，损失值为{best_test_loss:.4f}")
#                     early_stopped = True
#                     break  # 终止训练循环
#             # ======================

#         # 每1000个epoch更新一次进度条
#         if (epoch + 1) % 1000 == 0:
#             pbar.update(1000)  # 更新进度条

#     # 确保进度条达到100%
#     if pbar.n < num_epochs:
#         pbar.update(num_epochs - pbar.n)  # 计算剩余的进度并更新

# time_all = time.time() - start_time  # 计算训练时间
# print(f'Training time: {time_all:.2f} seconds')

# # ===== 新增：加载最佳模型用于最终评估 =====
# if early_stopped:
#     print(f"加载第{best_epoch}轮的最佳模型进行最终评估...")
#     model.load_state_dict(torch.load('best_model.pth'))
# # ================================

# # 可视化损失曲线
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_losses, label='Train Loss')
# plt.plot(epochs, test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Test Loss over Epochs')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 在测试集上评估模型
# model.eval()
# with torch.no_grad():
#     outputs = model(X_test)
#     _, predicted = torch.max(outputs, 1)
#     correct = (predicted == y_test).sum().item()
#     accuracy = correct / y_test.size(0)
#     print(f'测试集准确率: {accuracy * 100:.2f}%')


# 加载昨天关于信贷数据的代码模型
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
        self.dropout = nn.Dropout(0.2)  # Dropout层（随机丢弃30%的神经元，防止过拟合）
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
train_losses = []         # 记录每200轮的训练损失
test_losses = []          # 记录每200轮的测试损失
accuracies = []     # 记录每200轮的测试准确率
epochs = []         # 记录对应的轮数
# ==========新增早停相关参数==========
best_test_loss = float('inf')  # 记录最佳测试集损失
best_epoch = 0                 # 记录最佳epoch
patience =  50              # 早停耐心值（连续多少轮测试集损失未改善时停止训练）
counter = 0                    # 早停计数器
early_stopped = False          # 是否早停标志
# =====================================
start_time = time.time()  # 记录训练开始时间

# 创建tqdm进度条（可视化训练进度）
with tqdm(total=num_epochs, desc="训练进度", unit="epoch") as pbar:
    for epoch in range(num_epochs):
        # 前向传播：模型根据输入数据计算预测值
        outputs = model(X_train)  # 模型输出（形状：[训练样本数, 2]，表示每个样本属于2个类别的分数）
        train_loss = criterion(outputs, y_train)  # 计算损失（预测值与真实标签的差异，越小越好）
 
        # 反向传播和参数更新
        optimizer.zero_grad()  # 清空历史梯度（避免梯度累加）
        train_loss.backward()        # 反向传播计算梯度（自动求导）
        optimizer.step()       # 根据梯度更新模型参数（优化器核心操作）
 
        # 每200轮记录一次损失和准确率（避免记录太频繁影响速度）
        if (epoch + 1) % 200 == 0:
            # 在测试集上评估模型（不更新参数，只看效果）
            model.eval()  # 切换到评估模式（关闭Dropout，保证结果稳定）
            with torch.no_grad():  # 禁用梯度计算（节省内存，加速推理）
                test_outputs = model(X_test)  # 测试集预测值
                test_loss = criterion(test_outputs, y_test)  # 计算测试集损失
            model.train()  # 切换回训练模式

            # 记录损失值和准确率
            train_losses.append(train_loss.item())  # 训练集损失
            test_losses.append(test_loss.item())  # 测试集损失
            epochs.append(epoch + 1)  # 记录轮数
            # 更新进度条显示的信息（当前损失和准确率）
            pbar.set_postfix({'Train Loss': f'{train_loss.item():.4f}', 'Test Loss': f'{test_loss.item():.4f}'})
            # ===== 新增早停逻辑 =====
            if test_loss.item() < best_test_loss: # 如果当前测试集损失小于最佳损失
                best_test_loss = test_loss.item() # 更新最佳损失
                best_epoch = epoch + 1 # 更新最佳epoch
                counter = 0 # 重置计数器
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停触发！在第{epoch+1}轮，测试集损失已有{patience}轮未改善。")
                    print(f"最佳测试集损失出现在第{best_epoch}轮，损失值为{best_test_loss:.4f}")
                    early_stopped = True
                    break  # 终止训练循环
            # ======================
        # 每1000轮更新一次进度条（避免进度条刷新太频繁）
        if (epoch + 1) % 1000 == 0:
            pbar.update(1000)  # 进度条前进1000步
 
    # 确保进度条最终显示100%（防止最后一轮未更新）
    if pbar.n < num_epochs:
        pbar.update(num_epochs - pbar.n)
 
# 计算总训练时间并打印
time_all = time.time() - start_time
print(f'Training time: {time_all:.2f} seconds')

# ===== 新增：加载最佳模型用于最终评估 =====
if early_stopped:
    print(f"加载第{best_epoch}轮的最佳模型进行最终评估...")
    model.load_state_dict(torch.load('best_model.pth'))
# ================================
 
# ------------------- 可视化训练结果 -------------------
# 创建双y轴图表（损失和准确率在同一张图显示）
# 可视化损失曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
 
# ------------------- 最终测试集评估 -------------------
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    outputs = model(X_test)  # 测试集预测值
    _, predicted = torch.max(outputs, 1)  # 取预测类别（0或1）
    correct = (predicted == y_test).sum().item()  # 正确预测的样本数
    accuracy = correct / y_test.size(0)  # 计算准确率
    print(f'测试集准确率: {accuracy * 100:.2f}%')  # 打印准确率（百分比形式）





