# import pandas as pd
# data = pd.read_csv('python60-days-challenge\data.csv')
# print(data.columns)
# print(data.dtypes)
# for discrete_features in data.columns:
#     if data[discrete_features].dtype == 'object':
#         print(discrete_features)
# print(data['Home Ownership'])
# print(data['Home Ownership'].value_counts())      
# data = pd.get_dummies(data,columns=['Home Ownership']) 
# print(data.columns) 
# print(data.head(5))
# data['Home Ownership_Have Mortgage']=data['Home Ownership_Have Mortgage'].astype(int)
# print(data['Home Ownership_Have Mortgage'])

# data2 = pd.read_csv('python60-days-challenge\data.csv')
# discrete_lists = []
# for discrete_features in data2.columns:
#     if data2[discrete_features].dtype == 'object':
#         discrete_lists.append(discrete_features)

# data2 = pd.get_dummies(data2, columns=discrete_lists, drop_first=True)

# print(data2.columns)

# import pandas as pd

# data3 = pd.read_csv("python60-days-challenge\data.csv")

# orriginal_columns = data3.columns

# discrete_lists = []

# for discrete_features in data3.columns:
#     if data3[discrete_features].dtype == 'object':
#         discrete_lists.append(discrete_features)

# data3 = pd.get_dummies(data3, columns=discrete_lists, drop_first=True)

# list_final = []

# # print(data3.columns)
# for i in data3.columns:
#     if i not in orriginal_columns:
#         list_final.append(i)
# print(list_final)

# for i in list_final:
#     data3[i] = data3[i].astype(int)
# print(data3.head())
# print(data3.dtypes)
# print(data3.isnull().sum())

# for i in data3.columns:
#     if data3[i].isnull().sum() > 0:
#         mean_value = data3[i].mean()
#         data3[i].fillna(mean_value, inplace = True)

# print(data3.isnull().sum())

# 导入 pandas 库，这是一个用于数据处理和分析的强大库
import pandas as pd 

# 读取 CSV 文件，使用原始字符串（在字符串前加 r）避免反斜杠转义问题
# 初学者注意：路径中的反斜杠在普通字符串中需要转义（写成两个反斜杠），使用原始字符串可以简化这个过程
data = pd.read_csv(r"C:\Users\I.Love.I\Desktop\Python_code\python60-days-challenge\data.csv") 

# 保存原始数据的列名，后续用于对比独热编码前后的列名
original_features = data.columns 

# 初始化一个空列表，用于存储数据类型为 'object' 的列名
discrete_lists = [] 

# 遍历数据的所有列名
for discrete_features in data.columns: 
    # 检查当前列的数据类型是否为 'object'，通常 'object' 类型表示该列包含字符串或分类数据
    if data[discrete_features].dtype == 'object': 
        # 如果是 'object' 类型，将该列名添加到 discrete_lists 列表中
        discrete_lists.append(discrete_features) 

# 对数据进行独热编码，将 discrete_lists 中的列转换为二进制的列
# drop_first=True 表示删除每个分类变量的第一个类别，以避免多重共线性问题
# 初学者注意：独热编码会增加数据的列数，可能导致数据维度爆炸
data = pd.get_dummies(data,columns=discrete_lists,drop_first=True) 

# 初始化一个空列表，用于存储独热编码后新增的列名
list_final = [] 

# 遍历独热编码后数据的所有列名
for i in data.columns: 
    # 检查当前列名是否不在原始列名列表中
    if i not in original_features: 
        # 如果不在，说明该列是独热编码后新增的，将其添加到 list_final 列表中
        list_final.append(i) 

# 打印独热编码后新增的列名
print(list_final) 

# 遍历独热编码后新增的列名
for i in list_final: 
    # 将新增列的数据类型转换为整数类型
    # 初学者注意：如果列中包含非数值或缺失值，转换可能会出错
    data[i] = data[i].astype(int) 

# 统计数据中每列的缺失值数量并打印
print(data.isnull().sum()) 

# 遍历数据的所有列名
for i in data.columns: 
    # 检查当前列的缺失值数量是否大于 0
    if data[i].isnull().sum() > 0: 
        # 计算当前列的均值
        mean_value = data[i].mean() 
        # 使用均值填充当前列的缺失值
        # inplace=True 表示直接在原数据上进行修改
        # 初学者注意：使用均值填充可能会改变数据的分布，需要谨慎使用
        data[i].fillna(mean_value,inplace=True) 

# 再次统计数据中每列的缺失值数量并打印，验证填充是否成功
print(data.isnull().sum()) 