# 导入相关库函数进而对数据进行初步的处理，包括数据读取、数据预处理、数据可视化等。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time # 导入 time 库
import warnings
warnings.filterwarnings("ignore")

# 正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 防止负号显示问题
 
# 导入 Pipeline 和相关预处理工具
from sklearn.pipeline import Pipeline #  用于创建机器学习工作流
from sklearn.compose import ColumnTransformer # 用于将不同的预处理应用于不同的列，之前是对datafame的某一列手动处理，如果在pipeline中直接用standardScaler等函数就会对所有列处理，所以要用到这个工具
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler # 用于数据预处理
from sklearn.impute import SimpleImputer # 用于处理缺失值
 
# 机器学习相关库
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split # 只导入 train_test_split
 
 
# --- 加载原始数据 ---
data = pd.read_csv(r'C:\Users\I.Love.I\Desktop\Python_code\python60-days-challenge\python-learning-library\adult11.csv')
# Pipeline 将直接处理分割后的原始数据 X_train, X_test
# 原手动预处理步骤 (将被Pipeline替代):
# Home Ownership 标签编码
# Years in current job 标签编码
# Purpose 独热编码
# Term 0 - 1 映射并重命名
# 连续特征用众数补全
print(data.columns)
data = data.drop(columns=["occupation","native-country","workclass"])#看具体数据进行选择
# --- 分离特征和标签 (使用原始数据) --
y = data['salary']
X = data.drop(['salary'], axis=1)
 
# --- 划分训练集和测试集 (在任何预处理之前划分) ---
# X_train 和 X_test 现在是原始数据中划分出来的部分，不包含你之前的任何手动预处理结果
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
 
# --- 定义不同列的类型和它们对应的预处理步骤 (这些将被放入 Pipeline 的 ColumnTransformer 中) ---
# 这些定义是基于原始数据 X 的列类型来确定的
 
# 识别原始的 object 列 (对应你原代码中的 discrete_features 在预处理前)
object_cols = X.select_dtypes(include=['object']).columns.tolist()
 
# 有序分类特征 (对应你之前的标签编码)
# 注意：OrdinalEncoder默认编码为0, 1, 2... 对应你之前的1, 2, 3...需要在模型解释时注意
# 这里的类别顺序需要和你之前映射的顺序一致
ordinal_features = ['education', 'marital-status']
# 定义每个有序特征的类别顺序，这个顺序决定了编码后的数值大小
ordinal_categories = [
    ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'], 
    ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent']
]
# 先用众数填充分类特征的缺失值，然后进行有序编码
ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 用众数填充分类特征的缺失值
    ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1))
])
 
# 分类特征 
nominal_features = [
    'relationship',   # 家庭关系
    'race',           # 种族
    'gender',         # 性别
    ] # 使用原始列名
# 先用众数填充分类特征的缺失值，然后进行独热编码
nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 用众数填充分类特征的缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False 使输出为密集数组
])
 
# 连续特征
# 从X的列中排除掉分类特征，得到连续特征列表
continuous_features = X.columns.difference(object_cols).tolist() # 原始X中非object类型的列
 
# 先用众数填充缺失值，然后进行标准化
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # 用众数填充缺失值 (复现你的原始逻辑)
    ('scaler', StandardScaler()) # 标准化，一个好的实践
])
 
# --- 构建 ColumnTransformer ---
# 将不同的预处理应用于不同的列子集，构造一个完备的转化器
preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', ordinal_transformer, ordinal_features),
        ('nominal', nominal_transformer, nominal_features),
        ('continuous', continuous_transformer, continuous_features)
    ],
    remainder='passthrough' # 保留没有在transformers中指定的列（如果存在的话），或者 'drop' 丢弃
)
 
# --- 构建完整的 Pipeline ---
# 将预处理器和模型串联起来
# 使用你原代码中 RandomForestClassifier 的默认参数和 random_state，这里的参数用到了元组这个数据结构
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # 第一步：应用所有的预处理 (ColumnTransformer)
    ('classifier', RandomForestClassifier(random_state=42)) # 第二步：随机森林分类器
])
 
# --- 1. 使用 Pipeline 在划分好的训练集和测试集上评估 ---
 
print("--- 1. 默认参数随机森林 (训练集 -> 测试集) ---") 
start_time = time.time() # 记录开始时间
 
# 在原始的 X_train 上拟合整个Pipeline
# Pipeline会自动按顺序执行preprocessor的fit_transform(X_train)，然后用处理后的数据拟合classifier
pipeline.fit(X_train, y_train)
 
# 在原始的 X_test 上进行预测
# Pipeline会自动按顺序执行preprocessor的transform(X_test)，然后用处理后的数据进行预测
pipeline_pred = pipeline.predict(X_test)
 
end_time = time.time() # 记录结束时间
 
print(f"训练与预测耗时: {end_time - start_time:.4f} 秒") # 使用你原代码的输出格式
 
print("\n默认随机森林 在测试集上的分类报告：") # 使用你原代码的输出文本
print(classification_report(y_test, pipeline_pred))
print("默认随机森林 在测试集上的混淆矩阵：") # 使用你原代码的输出文本
print(confusion_matrix(y_test, pipeline_pred))


