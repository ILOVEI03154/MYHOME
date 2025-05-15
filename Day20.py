import pandas as pd
import pandas as pd    #用于数据处理和分析，可处理表格数据。
import numpy as np     #用于数值计算，提供了高效的数组操作。
import matplotlib.pyplot as plt    #用于绘制各种类型的图表
import seaborn as sns   #基于matplotlib的高级绘图库，能绘制更美观的统计图形。
import warnings
warnings.filterwarnings("ignore")
 
 # 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
data = pd.read_csv('python60-days-challenge\python-learning-library\heart.csv')    #读取数据
print(data.columns)    #'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
# 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
#提取连续特征值
continuous_features = ['age','trestbps', 'chol', 'thalach','oldpeak']
print(continuous_features)    #打印出连续特征值的列名
#提取离散特征值
discrete_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal','target']
print(discrete_features)    #打印出离散特征值的列名

#使用映射字典进行转换
mapping_dict = {
    'cp':{0:0,1:1,2:2,3:3},
   'restecg':{0:0,1:1,2:2},
   'slope':{0:0,1:1,2:2},
    'thal':{0:0,1:1,2:2,3:3},
    'ca':{0:0,1:1,2:2,3:3,4:4}
}
for feature, mapping_dict in mapping_dict.items():    #遍历映射字典
    data[feature] = data[feature].map(mapping_dict)    #将映射字典中的值替换为原数据中的值

#对离散特征值进行独热编码
data = pd.get_dummies(data, columns=['sex', 'fbs','exang'])    #对离散特征值进行独热编码    
print(data.columns)    #打印出数据的列名
data2 = pd.read_csv("python60-days-challenge\python-learning-library\heart.csv")    #读取数据
list_final = []    #新建一个空列表，用于存放独热编码后新增的特征名
for i in data.columns:    #遍历数据的列名
    if i not in data2.columns:    #如果列名不在原数据的列名中
        list_final.append(i)    #将列名添加到列表中
for i in list_final:    #遍历列表中的列名
    data[i] = data[i].astype(int)    #将列名转换为int类型

#划分训练集和测试集
from sklearn.model_selection import train_test_split    #导入train_test_split函数
X = data.drop(['target'], axis=1)    #特征，axis=1表示按列删除
y = data['target']    #标签
#按照8:2划分训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    #80%训练集，20%测试集
# print(X_train.shape)    #打印出训练集的形状
# print(X_test.shape)    #打印出测试集的形状
from sklearn.ensemble import RandomForestClassifier #随机森林分类器

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # 用于评估分类器性能的指标
from sklearn.metrics import classification_report, confusion_matrix #用于生成分类报告和混淆矩阵
import warnings #用于忽略警告信息
warnings.filterwarnings("ignore") # 忽略所有警告信息

#数据标准化
from sklearn.preprocessing import StandardScaler #用于数据标准化
scaler = StandardScaler() #创建一个StandardScaler对象
X_train = scaler.fit_transform(X_train) #对训练集进行标准化
X_test = scaler.transform(X_test) #对测试集进行标准化


# 如何对训练集进行 SVD 降维，训练模型，并对测试集应用相同的降维变换。
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 对训练集进行 SVD 分解
U_train, sigma_train, Vt_train = np.linalg.svd(X_train, full_matrices=False)
print(f"Vt_train 矩阵形状: {Vt_train.shape}")

# 选择保留的奇异值数量 k
k = 10
Vt_k = Vt_train[:k, :]  # 保留前 k 行，形状为 (k, 50)
print(f"保留 k={k} 后的 Vt_k 矩阵形状: {Vt_k.shape}")

# 降维训练集：X_train_reduced = X_train @ Vt_k.T
X_train_reduced = X_train @ Vt_k.T
print(f"降维后训练集形状: {X_train_reduced.shape}")

# 使用相同的 Vt_k 对测试集进行降维：X_test_reduced = X_test @ Vt_k.T
X_test_reduced = X_test @ Vt_k.T
print(f"降维后测试集形状: {X_test_reduced.shape}")

# 训练模型（以逻辑回归为例）
model = LogisticRegression(random_state=42)
model.fit(X_train_reduced, y_train)

# 预测并评估
y_pred = model.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.3f}")

# 计算训练集的近似误差（可选，仅用于评估降维效果）
X_train_approx = U_train[:, :k] @ np.diag(sigma_train[:k]) @ Vt_k
error = np.linalg.norm(X_train - X_train_approx, 'fro') / np.linalg.norm(X_train, 'fro')
print(f"训练集近似误差 (Frobenius 范数相对误差): {error:.3f}")


