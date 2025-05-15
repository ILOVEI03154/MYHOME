# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns  # 基于matplotlib的高级数据可视化库
import warnings  # 用于处理警告信息
# 忽略所有警告信息，避免影响程序运行时的输出
warnings.filterwarnings("ignore")

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 读取数据
# 从指定路径读取CSV文件，并将数据存储在data变量中
data  = pd.read_csv('python60-days-challenge\python-learning-library\data.csv')

# 2. 筛选字符串变量
# 筛选出数据集中类型为object（通常为字符串）的列，并将列名存储在discrete_features列表中
discrete_features = data.select_dtypes(include=['object']).columns.tolist()
# 打印离散变量名
print(discrete_features)
# 离散变量名：['Home Ownership', 'Years in current job', 'Purpose', 'Term']

# 分别进行 标签编码
# Home Ownership 标签编码
# 定义映射字典，将不同的房屋所有权状态映射为对应的整数
home_ownership_mapping = {
    'Own Home': 1,
    'Rent': 2,
    'Have Mortgage': 3,
    'Home Mortgage': 4
}
# 使用映射字典将Home Ownership列的值进行替换
data['Home Ownership'] = data['Home Ownership'].map(home_ownership_mapping)

# Years in current job 标签编码
# 定义映射字典，将在当前工作的年限映射为对应的整数
years_in_job_mapping = {
    '< 1 year': 1,
    '1 year': 2,
    '2 years': 3,
    '3 years': 4,
    '4 years': 5,
    '5 years': 6,
    '6 years': 7,
    '7 years': 8,
    '8 years': 9,
    '9 years': 10,
    '10+ years': 11
}
# 使用映射字典将Years in current job列的值进行替换
data['Years in current job'] = data['Years in current job'].map(years_in_job_mapping)

# Purpose 独热编码，需记得要将bool类型转换为数值
# 对Purpose列进行独热编码，将其转换为多个二进制列
data = pd.get_dummies(data, columns=['Purpose'])
# 重新读取数据，用来做列名对比
data2 = pd.read_csv("python60-days-challenge\python-learning-library\data.csv") 
# 新建一个空列表，用于存放独热编码后新增的特征名
list_final = [] 
# 遍历data的列名，找出独热编码后新增的列名
for i in data.columns:
    if i not in data2.columns:
       list_final.append(i) # 这里打印出来的就是独热编码后的特征名
# 将独热编码后的列的数据类型转换为整数
for i in list_final:
    data[i] = data[i].astype(int) # 这里的i就是独热编码后的特征名

# Term 0 - 1 映射
# 定义映射字典，将短期和长期贷款期限映射为0和1
term_mapping = {
    'Short Term': 0,
    'Long Term': 1
}
# 使用映射字典将Term列的值进行替换
data['Term'] = data['Term'].map(term_mapping)
# 重命名列名，将Term改为Long Term，inplace=True表示直接在原数据上修改，不返回新数据
data.rename(columns={'Term': 'Long Term'}, inplace=True) 
# 筛选出数据集中类型为float64和int64的列，并将列名存储在continues_features列表中
continues_features = data.select_dtypes(include=['float64','int64']).columns.tolist() 

# 查看数据是否具有缺失值
print(data.isnull().sum()) # 查看数据是否具有缺失值，发现没有缺失值，不需要处理缺失值

# 补全数据
# 连续特征用中位数补全缺失值
# 遍历连续特征列，用每列的中位数填充该列的缺失值
for i in continues_features:
    data[i].fillna(data[i].median(), inplace=True) # 用中位数补全缺失值，inplace=True表示直接在原数据上修改，不返回新数据  

# 3. 查看数据
print(data.isnull().sum()) # 查看数据是否具有缺失值，发现没有缺失值，不需要处理缺失值

# 由于许多调参函数自带交叉验证，甚至必选参数
# 导入train_test_split函数，用于划分训练集和测试集
from sklearn.model_selection import train_test_split 
# 特征按列删除，axis=1表示按列删除，将Credit Default列从数据集中删除，得到特征矩阵X
X = data.drop('Credit Default', axis=1) 
# 标签按列选择，axis=1表示按列选择，选取Credit Default列作为标签向量y
y = data['Credit Default'] 
# 划分训练集和测试集，test_size=0.2表示测试集占20%，random_state=42表示随机种子，保证每次运行结果一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


# 导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier 
# 导入评估分类器性能的指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
# 导入生成分类报告和混淆矩阵的函数
from sklearn.metrics import classification_report, confusion_matrix,make_scorer 
# 导入警告信息
import warnings 
# 忽略所有警告信息
warnings.filterwarnings("ignore") 
# --- 1. 默认参数的随机森林 ---
# 评估基准模型，这里确实不需要验证集
print("--- 1. 默认参数随机森林 (训练集 -> 测试集) ---")
# 这里介绍一个新的库，time库，主要用于时间相关的操作，因为调参需要很长时间，记录下会帮助后人知道大概的时长
import time 
# 记录开始时间
start_time = time.time() 
# 随机森林分类器，random_state=42表示随机种子，保证每次运行结果一致
rf_model = RandomForestClassifier(random_state=42) 
# 在训练集上训练随机森林模型
rf_model.fit(X_train, y_train) 
# 在测试集上进行预测
rf_pred = rf_model.predict(X_test) 
# 记录结束时间
end_time = time.time() 
# 打印训练与预测耗时
print(f"训练与预测耗时: {end_time - start_time:.4f} 秒") 
# 打印默认随机森林在测试集上的分类报告
print("\n默认随机森林 在测试集上的分类报告：") 
print(classification_report(y_test, rf_pred))
# 打印默认随机森林在测试集上的混淆矩阵
print("默认随机森林 在测试集上的混淆矩阵：")
print(confusion_matrix(y_test, rf_pred))


"""
## SHAP 原理简介

**目标：** 理解复杂机器学习模型（尤其是“黑箱”模型，如随机森林、梯度提升树、神经网络等）**为什么**会对**特定输入**做出**特定预测**。 SHAP 提供了一种统一的方法来解释模型的输出。

**核心思想：合作博弈论中的 Shapley 值**

SHAP (SHapley Additive exPlanations) 的核心基于博弈论中的 **Shapley 值**概念。想象一个合作游戏：

1.  **玩家 (Players):** 模型的**特征 (Features)** 就是玩家。
2.  **游戏 (Game):** 目标是预测某个样本的输出值。
3.  **合作 (Coalition):** 不同的特征子集可以“合作”起来进行预测。
4.  **奖励/价值 (Payout/Value):** 某个特征子集进行预测得到的值。
5.  **目标：** 如何**公平地**将最终预测结果（相对于平均预测结果的“收益”）分配给每个参与的特征（玩家）？

**Shapley 值的计算思路（概念上）：**

为了计算一个特定特征（比如“特征 A”）对某个预测的贡献（它的 Shapley 值），SHAP 会考虑：

1.  **所有可能的特征组合（子集/联盟）：** 从没有特征开始，到包含所有特征。
2.  **特征 A 的边际贡献：** 对于**每一个**特征组合，比较“包含特征 A 的组合的预测值”与“不包含特征 A 但包含其他相同特征的组合的预测值”之间的**差异**。这个差异就是特征 A 在这个特定组合下的“边际贡献”。
3.  **加权平均：** Shapley 值是该特征在**所有可能**的特征组合中边际贡献的**加权平均值**。权重确保了分配的公平性。

**SHAP 的关键特性 (加性解释 - Additive Explanations):**

SHAP 的一个重要特性是**加性 (Additive)**。这意味着：

*   **基准值 (Base Value / Expected Value):** 这是模型在整个训练（或背景）数据集上的平均预测输出。可以理解为没有任何特征信息时的“默认”预测。
*   **SHAP 值之和：** 对于**任何一个**样本的预测，**所有特征的 SHAP 值加起来，再加上基准值，就精确地等于该样本的模型预测值**。
    ```
    模型预测值(样本 X) = 基准值 + SHAP值(特征1) + SHAP值(特征2) + ... + SHAP值(特征N)
    ```

**为什么会生成 `shap_values` 数组？**

根据上述原理，SHAP 需要为**每个样本的每个特征**计算一个贡献值（SHAP 值）：

1.  **解释单个预测：** SHAP 的核心是解释**单个**预测结果。
2.  **特征贡献：** 对于这个预测，我们需要知道**每个特征**是把它往“高”推了，还是往“低”推了（相对于基准值），以及推了多少。
3.  **数值化：** 这个“推力”的大小和方向就是该特征对该样本预测的 **SHAP 值**。

因此：

*   **对于回归问题：**
    *   模型只有一个输出。
    *   对 `n_samples` 个样本中的**每一个**，计算 `n_features` 个特征各自的 SHAP 值。
    *   这就自然形成了形状为 `(n_samples, n_features)` 的数组。 `shap_values[i, j]` 代表第 `i` 个样本的第 `j` 个特征对该样本预测值的贡献。

*   **对于分类问题：**
    *   模型通常为**每个类别**输出一个分数或概率。
    *   SHAP 需要解释模型是如何得到**每个类别**的分数的。
    *   因此，对 `n_samples` 个样本中的**每一个**，**分别为每个类别**计算 `n_features` 个特征的 SHAP 值。
    *   最常见的组织方式是返回一个**列表**，列表长度等于类别数。列表的第 `k` 个元素是一个 `(n_samples, n_features)` 的数组，表示所有样本的所有特征对预测**类别 `k`** 的贡献。
    *   `shap_values[k][i, j]` 代表第 `i` 个样本的第 `j` 个特征对该样本预测**类别 `k`** 的贡献。

**总结:**

SHAP 通过计算每个特征对单个预测（相对于平均预测）的边际贡献（Shapley 值），提供了一种将模型预测分解到每个特征上的方法。这种分解对于每个样本和每个特征（以及分类问题中的每个类别）都需要进行，因此生成了我们看到的 `shap_values` 数组结构。
"""

# 导入 shap 库，用于解释模型预测
import shap
import matplotlib.pyplot as plt # 用于数据可视化
# 初始化 SHAP 解释器，使用随机森林模型作为基础
explainer = shap.Explainer(rf_model) # 这里的rf_model是随机森林模型

# 计算 SHAP 值，对测试集进行解释
shap_values = explainer(X_test) # 这里的X_test是测试集

"""
## shap的维度要求

非常多的同学和我反映过代码跑不通，甚至换个电脑就不行了等玄学问题，本质都是由于没有搞清楚shap的维度要求，这里记录下，希望对大家有所帮助。

分类问题和回归问题输出的shap_values的形状不同。

- 分类问题：shap_values.shape =(n_samples, n_features, n_classes)
- 回归问题：shap_values.shape = (n_samples, n_features)

**数据维度的要求将是未来学习神经网络最重要的东西之一。**
"""

print(shap_values)

print(shap_values.shape) # 查看shap_values的形状，这里是(10000, 20)，表示有10000个样本，每个样本有20个特征

print("shap_values shape:", shap_values.shape)
print("shap_values[0] shape:", shap_values[0].shape)
print("shap_values[:, :, 0] shape:", shap_values[:, :, 0].shape)
print("X_test shape:", X_test.shape)

# --- 1. SHAP 特征重要性条形图 (Summary Plot - Bar) ---
print("--- 1. SHAP 特征重要性条形图 ---")
shap.summary_plot(shap_values[:, :, 0], X_test, plot_type="bar",show=False)  #  这里的show=False表示不直接显示图形,这样可以继续用plt来修改元素，不然就直接输出了
plt.title("SHAP Feature Importance (Bar Plot)")
plt.show()

# --- 2. SHAP 特征重要性蜂巢图 (Summary Plot - Violin) ---
print("--- 2. SHAP 特征重要性蜂巢图 ---")
shap.summary_plot(shap_values[:, :, 0], X_test,plot_type="violin",show=False,max_display=10) # 这里的show=False表示不直接显示图形,这样可以继续用plt来修改元素，不然就直接输出了
plt.title("SHAP Feature Importance (Violin Plot)")
plt.show()
