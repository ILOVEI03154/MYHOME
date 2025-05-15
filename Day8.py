import pandas as pd # 导入pandas库，用于数据处理和分析。
data = pd.read_csv('python60-days-challenge\python-learning-library\data.csv') # 读取名为"data.csv"的CSV文件，并将其存储在名为"data"的变量中。

# 定义映射字典
mapping = {
        "Rent": 0,
        "Own Home": 1,
        "Have Mortgage  ": 2,
        "Home Mortgage": 3
    
}
print(data["Home Ownership"].head())

data["Home Ownership"] = data["Home Ownership"].map(mapping)
print(data["Home Ownership"].head())

print(data["Term"].value_counts())

# 定义映射字典
mapping = {
    "Short Term": 1,
    "Long Term": 0
}

# 进行映射
data["Term"] = data["Term"].map(mapping)
print(data["Term"].head())

import pandas as pd

# 重新读取数据
data = pd.read_csv("python60-days-challenge\python-learning-library\data.csv")
# 嵌套映射字典
mapping = {
    "Term": {
        "Short Term": 1,
        "Long Term": 0
    },
    "Home Ownership": {
        "Rent": 0,
        "Own Home": 1,
        "Have Mortgage  ": 2,
        "Home Mortgage": 3
    }
}

print(mapping["Term"])

# 对 Home Ownership 列进行映射
data["Home Ownership"] = data["Home Ownership"].map(mapping["Home Ownership"])

# 对 Term 列进行映射
data["Term"] = data["Term"].map(mapping["Term"])

print(data.head())

# 对Annual Income列做归一化，手动构建函数实现
# 自行学习下如何创建函数，这个很简单很常用
def manual_normalize(data):
    """
    此函数用于对输入的数据进行归一化处理
    :param data: 输入的一维数据（如 Pandas 的 Series）
    :return: 归一化后的数据
    """
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data
data['Annual Income'] = manual_normalize(data['Annual Income'])
print(data['Annual Income'].head())

# 借助sklearn库进行归一化处理

from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = pd.read_csv("python60-days-challenge\python-learning-library\data.csv")# 重新读取数据


# 归一化处理
min_max_scaler = MinMaxScaler() # 实例化 MinMaxScaler类，之前课上也说了如果采取这种导入函数的方式，不需要申明库名
data['Annual Income'] = min_max_scaler.fit_transform(data[['Annual Income']])

print(data['Annual Income'].head())

# 标准化处理
data = pd.read_csv("python60-days-challenge\python-learning-library\data.csv")# 重新读取数据
scaler = StandardScaler() # 实例化 StandardScaler，
data['Annual Income'] = scaler.fit_transform(data[['Annual Income']])
print(data['Annual Income'].head())