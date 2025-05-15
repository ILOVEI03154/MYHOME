#复习日
#复现前15日所学内容，尽量并找到与控制工程相关的数据集，进行分析（最终不知道怎么找，目前没找到，就找了了一个非常简单的数据集）
#导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import shap
import warnings  # 用于处理警告信息
# 忽略所有警告信息，避免影响程序运行时的输出
warnings.filterwarnings("ignore")

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

#导入数据
data = pd.read_csv('python60-days-challenge\ssd_failure_tag.csv')
#数据探索
#1.查看数据的基本信息
# print(data.head())
print(data.info())
# #2.查看数据分布
# print(data.describe())
# #3.检查缺失值
# print(data.isnull().sum())
# #4.检查重复值
# print(data.duplicated().sum())#并没有重复值

# #数据清洗
# #1.处理缺失值
# #观察数据的箱线图
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=data)
# plt.title('Boxplot of Data')
# plt.show()#可以看出，数据分布较为集中，没有异常值，因此不需要处理缺失值
# #由箱线图可知说明的问题
# # 异常值存在：图中须之外有多个圆圈，表明数据集中存在异常值。这些异常值可能是由于数据录入错误、测量误差或真实存在的极端情况导致。
# # 数据分布：通过箱体位置和须的长度，可以大致判断数据分布情况。若箱体较短，说明数据在中位数附近较为集中；若箱体较长，数据分布较分散。图中不同变量箱体和须的情况不同，反映各变量数据分布特征有差异。
# # 比较变量：该图展示多个变量的箱线图，可对比不同变量的分布特征、异常值情况等。例如，能看出不同变量异常值数量和分布位置的差异，帮助了解各变量数据的离散程度和集中趋势等特征。
# #由于箱体较短，说明数据在中位数附近较为集中，因此使用中位数对缺失值进行补全
# #使用中位数对缺失值进行补全
# #使用连续变量的中位数对缺失值进行补全
# #首先将连续变量进行筛选
# continuous_columns = data.select_dtypes(include=[np.number]).columns
# # 使用中位数对缺失值进行补全
# for column in continuous_columns:
#     # 修改此处，避免使用 inplace 参数
#     data[column] = data[column].fillna(data[column].median())
# #再次检查缺失值
# print(data.isnull().sum())#缺失值已经补全
# #其次将离散变量进行筛选
# discrete_columns = data.select_dtypes(include=[object]).columns
# #使用众数对缺失值进行补全
# for column in discrete_columns:
#     # 修改此处，避免使用 inplace 参数
#     data[column] = data[column].fillna(data[column].mode()[0])
# #再次检查缺失值
# print(data.isnull().sum())#缺失值已经补全
#2.处理重复值
#由于没有重复值，因此不需要处理重复值
#特征工程
#1.特征选择
# print(data.columns)#查看列名
# """
# ['model', 'failure_time', 'failure', 'app', 'r_5', 'n_5', 'r_183',
# 'n_183', 'r_184', 'n_184', 'r_187', 'n_187', 'r_195', 'n_195', 'r_197',
# 'n_197', 'r_199', 'n_199', 'r_program', 'n_program', 'r_erase',
# 'n_erase', 'n_blocks', 'n_wearout', 'r_241', 'n_241', 'r_242', 'n_242',
# 'r_9', 'n_9', 'r_12', 'n_12', 'r_174', 'n_174', 'n_175', 'disk_id',
# 'node_id', 'rack_id', 'machine_room_id']
# """

#2.特征转换
#3.特征构造
#数据可视化
#1.单特征可视化
#2.特征与标签关系可视化
#3.多特征可视化
#模型构建
#1.选择模型
#2.模型训练
#3.模型评估




