# 导入 numpy 库，用于科学计算
import numpy as np
# 导入 pandas 库，用于数据处理和分析
import pandas as pd
# 导入 calendar 模块，用于处理日期和日历相关操作
import calendar
# 从 datetime 模块中导入 datetime 类，用于处理日期和时间
from datetime import datetime
# 导入 scipy 库的 stats 模块，用于统计分析
from scipy import stats
# 从 scipy.stats 模块中导入 norm 类，用于正态分布相关计算
from scipy.stats import norm

# 导入 matplotlib 的 pyplot 模块，用于数据可视化
import matplotlib.pyplot as plt
# 导入 seaborn 库，用于更美观的统计图表绘制
import seaborn as sns

# 从指定路径读取 CSV 文件，并将数据存储在 data 变量中
data = pd.read_csv("python60-days-challenge\python-learning-library\data.csv")

# 创建嵌套字典 mapping，用于将分类变量映射为数值变量
mapping  = {
    # 对 'Years in current job' 列的映射规则
    "Years in current job": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,       
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0
    },
    # 对 'Home Ownership' 列的映射规则
    "Home Ownership": {
        "Home Mortgage": 0,
        "Rent": 1,
        "Own Home": 2,
        "Have Mortgage": 3
    },
    # 对 'Term' 列的映射规则
    "Term": {
        "Short Term": 1,
        "Long Term": 0
    }
}

# 使用 mapping 字典对 'Years in current job' 列进行映射转换
data['Years in current job'] = data['Years in current job'].map(mapping['Years in current job'])
# 使用 mapping 字典对 'Home Ownership' 列进行映射转换
data['Home Ownership'] = data['Home Ownership'].map(mapping['Home Ownership'])
# 使用 mapping 字典对 'Term' 列进行映射转换
data['Term'] = data['Term'].map(mapping['Term'])

# 打印数据集的前 10 行，用于快速查看数据的结构和内容
print(data.head(10))

# 对 'Purpose' 列进行独热编码，drop_first=True 表示删除第一个类别以避免多重共线性
data = pd.get_dummies(data, columns=['Purpose'], drop_first=True)
# 打印编码后数据集的所有列名
print(data.columns)

# 提取数据集中类型为 float64 和 int64 的列，即连续特征，并存储在列表中
continues_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
# 打印提取的连续特征列表
print(continues_features)

# 遍历连续特征列表，对每个特征的缺失值用众数进行填充
for feature in continues_features:
    # 计算当前特征的众数
    mode_value = data[feature].mode()[0]
    # 用众数填充当前特征的缺失值
    data[feature].fillna(mode_value, inplace=True)

# 从 sklearn.model_selection 模块中导入 train_test_split 函数，用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 从数据集中删除 'Credit Default' 列，将剩余列作为特征矩阵 X
X = data.drop(['Credit Default'], axis=1)
# 将 'Credit Default' 列作为目标向量 y
y = data['Credit Default']
# 划分训练集和测试集，测试集占比 20%，随机种子为 42，保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 重复划分训练集和测试集，这一步可能是多余的，可删除
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印训练集和测试集的形状
print(f"训练集的形状：{X_train.shape},测试集的形状：{X_test.shape}")

# 从 sklearn.svm 模块中导入 SVC 类，即支持向量机分类器
from sklearn.svm import SVC 
# 从 sklearn.neighbors 模块中导入 KNeighborsClassifier 类，即 K 近邻分类器
from sklearn.neighbors import KNeighborsClassifier 
# 从 sklearn.linear_model 模块中导入 LogisticRegression 类，即逻辑回归分类器
from sklearn.linear_model import LogisticRegression 
# 导入 xgboost 库并别名为 xgb，用于使用 XGBoost 分类器
import xgboost as xgb 
# 导入 lightgbm 库并别名为 lgb，用于使用 LightGBM 分类器
import lightgbm as lgb 
# 从 sklearn.ensemble 模块中导入 RandomForestClassifier 类，即随机森林分类器
from sklearn.ensemble import RandomForestClassifier 
# 从 catboost 模块中导入 CatBoostClassifier 类，即 CatBoost 分类器
from catboost import CatBoostClassifier 
# 从 sklearn.tree 模块中导入 DecisionTreeClassifier 类，即决策树分类器
from sklearn.tree import DecisionTreeClassifier 
# 从 sklearn.naive_bayes 模块中导入 GaussianNB 类，即高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB 
# 从 sklearn.metrics 模块中导入评估分类器性能的指标函数
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
# 从 sklearn.metrics 模块中导入生成分类报告和混淆矩阵的函数
from sklearn.metrics import classification_report, confusion_matrix 
# 导入 warnings 模块，用于忽略警告信息
import warnings 
# 忽略所有警告信息
warnings.filterwarnings("ignore") 

# 初始化支持向量机分类器，设置随机种子为 42
svm_model = SVC(random_state=42)
# 使用训练集数据对支持向量机分类器进行训练
svm_model.fit(X_train, y_train)
# 使用训练好的支持向量机分类器对测试集数据进行预测
svm_pred = svm_model.predict(X_test)

# 打印支持向量机分类器的分类报告标题
print("\nSVM 分类报告：")
# 打印支持向量机分类器的分类报告
print(classification_report(y_test, svm_pred))  
# 打印支持向量机分类器的混淆矩阵标题
print("SVM 混淆矩阵：")
# 打印支持向量机分类器的混淆矩阵
print(confusion_matrix(y_test, svm_pred))  

# 计算支持向量机分类器的准确率
svm_accuracy = accuracy_score(y_test, svm_pred)
# 计算支持向量机分类器的精确率
svm_precision = precision_score(y_test, svm_pred)
# 计算支持向量机分类器的召回率
svm_recall = recall_score(y_test, svm_pred)
# 计算支持向量机分类器的 F1 值
svm_f1 = f1_score(y_test, svm_pred)
# 打印支持向量机分类器的评估指标标题
print("SVM 模型评估指标：")
# 打印支持向量机分类器的准确率，保留四位小数
print(f"准确率: {svm_accuracy:.4f}")
# 打印支持向量机分类器的精确率，保留四位小数
print(f"精确率: {svm_precision:.4f}")
# 打印支持向量机分类器的召回率，保留四位小数
print(f"召回率: {svm_recall:.4f}")
# 打印支持向量机分类器的 F1 值，保留四位小数
print(f"F1 值: {svm_f1:.4f}")

# 初始化 K 近邻分类器
knn_model = KNeighborsClassifier()
# 使用训练集数据对 K 近邻分类器进行训练
knn_model.fit(X_train, y_train)
# 使用训练好的 K 近邻分类器对测试集数据进行预测
knn_pred = knn_model.predict(X_test)

# 打印 K 近邻分类器的分类报告标题
print("\nKNN 分类报告：")
# 打印 K 近邻分类器的分类报告
print(classification_report(y_test, knn_pred))
# 打印 K 近邻分类器的混淆矩阵标题
print("KNN 混淆矩阵：")
# 打印 K 近邻分类器的混淆矩阵
print(confusion_matrix(y_test, knn_pred))

# 计算 K 近邻分类器的准确率
knn_accuracy = accuracy_score(y_test, knn_pred)
# 计算 K 近邻分类器的精确率
knn_precision = precision_score(y_test, knn_pred)
# 计算 K 近邻分类器的召回率
knn_recall = recall_score(y_test, knn_pred)
# 计算 K 近邻分类器的 F1 值
knn_f1 = f1_score(y_test, knn_pred)
# 打印 K 近邻分类器的评估指标标题
print("KNN 模型评估指标：")
# 打印 K 近邻分类器的准确率，保留四位小数
print(f"准确率: {knn_accuracy:.4f}")
# 打印 K 近邻分类器的精确率，保留四位小数
print(f"精确率: {knn_precision:.4f}")
# 打印 K 近邻分类器的召回率，保留四位小数
print(f"召回率: {knn_recall:.4f}")
# 打印 K 近邻分类器的 F1 值，保留四位小数
print(f"F1 值: {knn_f1:.4f}")

# 初始化逻辑回归分类器，设置随机种子为 42
logreg_model = LogisticRegression(random_state=42)
# 使用训练集数据对逻辑回归分类器进行训练
logreg_model.fit(X_train, y_train)
# 使用训练好的逻辑回归分类器对测试集数据进行预测
logreg_pred = logreg_model.predict(X_test)

# 打印逻辑回归分类器的分类报告标题
print("\n逻辑回归 分类报告：")
# 打印逻辑回归分类器的分类报告
print(classification_report(y_test, logreg_pred))
# 打印逻辑回归分类器的混淆矩阵标题
print("逻辑回归 混淆矩阵：")
# 打印逻辑回归分类器的混淆矩阵
print(confusion_matrix(y_test, logreg_pred))

# 计算逻辑回归分类器的准确率
logreg_accuracy = accuracy_score(y_test, logreg_pred)
# 计算逻辑回归分类器的精确率
logreg_precision = precision_score(y_test, logreg_pred)
# 计算逻辑回归分类器的召回率
logreg_recall = recall_score(y_test, logreg_pred)
# 计算逻辑回归分类器的 F1 值
logreg_f1 = f1_score(y_test, logreg_pred)
# 打印逻辑回归分类器的评估指标标题
print("逻辑回归 模型评估指标：")
# 打印逻辑回归分类器的准确率，保留四位小数
print(f"准确率: {logreg_accuracy:.4f}")
# 打印逻辑回归分类器的精确率，保留四位小数
print(f"精确率: {logreg_precision:.4f}")
# 打印逻辑回归分类器的召回率，保留四位小数
print(f"召回率: {logreg_recall:.4f}")
# 打印逻辑回归分类器的 F1 值，保留四位小数
print(f"F1 值: {logreg_f1:.4f}")

# 初始化高斯朴素贝叶斯分类器
nb_model = GaussianNB()
# 使用训练集数据对高斯朴素贝叶斯分类器进行训练
nb_model.fit(X_train, y_train)
# 使用训练好的高斯朴素贝叶斯分类器对测试集数据进行预测
nb_pred = nb_model.predict(X_test)

# 打印高斯朴素贝叶斯分类器的分类报告标题
print("\n朴素贝叶斯 分类报告：")
# 打印高斯朴素贝叶斯分类器的分类报告
print(classification_report(y_test, nb_pred))
# 打印高斯朴素贝叶斯分类器的混淆矩阵标题
print("朴素贝叶斯 混淆矩阵：")
# 打印高斯朴素贝叶斯分类器的混淆矩阵
print(confusion_matrix(y_test, nb_pred))

# 计算高斯朴素贝叶斯分类器的准确率
nb_accuracy = accuracy_score(y_test, nb_pred)
# 计算高斯朴素贝叶斯分类器的精确率
nb_precision = precision_score(y_test, nb_pred)
# 计算高斯朴素贝叶斯分类器的召回率
nb_recall = recall_score(y_test, nb_pred)
# 计算高斯朴素贝叶斯分类器的 F1 值
nb_f1 = f1_score(y_test, nb_pred)
# 打印高斯朴素贝叶斯分类器的评估指标标题
print("朴素贝叶斯 模型评估指标：")
# 打印高斯朴素贝叶斯分类器的准确率，保留四位小数
print(f"准确率: {nb_accuracy:.4f}")
# 打印高斯朴素贝叶斯分类器的精确率，保留四位小数
print(f"精确率: {nb_precision:.4f}")
# 打印高斯朴素贝叶斯分类器的召回率，保留四位小数
print(f"召回率: {nb_recall:.4f}")
# 打印高斯朴素贝叶斯分类器的 F1 值，保留四位小数
print(f"F1 值: {nb_f1:.4f}")

# 初始化决策树分类器，设置随机种子为 42
dt_model = DecisionTreeClassifier(random_state=42)
# 使用训练集数据对决策树分类器进行训练
dt_model.fit(X_train, y_train)
# 使用训练好的决策树分类器对测试集数据进行预测
dt_pred = dt_model.predict(X_test)

# 打印决策树分类器的分类报告标题
print("\n决策树 分类报告：")
# 打印决策树分类器的分类报告
print(classification_report(y_test, dt_pred))
# 打印决策树分类器的混淆矩阵标题
print("决策树 混淆矩阵：")
# 打印决策树分类器的混淆矩阵
print(confusion_matrix(y_test, dt_pred))

# 计算决策树分类器的准确率
dt_accuracy = accuracy_score(y_test, dt_pred)
# 计算决策树分类器的精确率
dt_precision = precision_score(y_test, dt_pred)
# 计算决策树分类器的召回率
dt_recall = recall_score(y_test, dt_pred)
# 计算决策树分类器的 F1 值
dt_f1 = f1_score(y_test, dt_pred)
# 打印决策树分类器的评估指标标题
print("决策树 模型评估指标：")
# 打印决策树分类器的准确率，保留四位小数
print(f"准确率: {dt_accuracy:.4f}")
# 打印决策树分类器的精确率，保留四位小数
print(f"精确率: {dt_precision:.4f}")
# 打印决策树分类器的召回率，保留四位小数
print(f"召回率: {dt_recall:.4f}")
# 打印决策树分类器的 F1 值，保留四位小数
print(f"F1 值: {dt_f1:.4f}")

# 初始化随机森林分类器，设置随机种子为 42
rf_model = RandomForestClassifier(random_state=42)
# 使用训练集数据对随机森林分类器进行训练
rf_model.fit(X_train, y_train)
# 使用训练好的随机森林分类器对测试集数据进行预测
rf_pred = rf_model.predict(X_test)

# 打印随机森林分类器的分类报告标题
print("\n随机森林 分类报告：")
# 打印随机森林分类器的分类报告
print(classification_report(y_test, rf_pred))
# 打印随机森林分类器的混淆矩阵标题
print("随机森林 混淆矩阵：")
# 打印随机森林分类器的混淆矩阵
print(confusion_matrix(y_test, rf_pred))

# 计算随机森林分类器的准确率
rf_accuracy = accuracy_score(y_test, rf_pred)
# 计算随机森林分类器的精确率
rf_precision = precision_score(y_test, rf_pred)
# 计算随机森林分类器的召回率
rf_recall = recall_score(y_test, rf_pred)
# 计算随机森林分类器的 F1 值
rf_f1 = f1_score(y_test, rf_pred)
# 打印随机森林分类器的评估指标标题
print("随机森林 模型评估指标：")
# 打印随机森林分类器的准确率，保留四位小数
print(f"准确率: {rf_accuracy:.4f}")
# 打印随机森林分类器的精确率，保留四位小数
print(f"精确率: {rf_precision:.4f}")
# 打印随机森林分类器的召回率，保留四位小数
print(f"召回率: {rf_recall:.4f}")
# 打印随机森林分类器的 F1 值，保留四位小数
print(f"F1 值: {rf_f1:.4f}")

# 初始化 XGBoost 分类器，设置随机种子为 42
xgb_model = xgb.XGBClassifier(random_state=42)
# 使用训练集数据对 XGBoost 分类器进行训练
xgb_model.fit(X_train, y_train)
# 使用训练好的 XGBoost 分类器对测试集数据进行预测
xgb_pred = xgb_model.predict(X_test)

# 打印 XGBoost 分类器的分类报告标题
print("\nXGBoost 分类报告：")
# 打印 XGBoost 分类器的分类报告
print(classification_report(y_test, xgb_pred))
# 打印 XGBoost 分类器的混淆矩阵标题
print("XGBoost 混淆矩阵：")
# 打印 XGBoost 分类器的混淆矩阵
print(confusion_matrix(y_test, xgb_pred))

# 计算 XGBoost 分类器的准确率
xgb_accuracy = accuracy_score(y_test, xgb_pred)
# 计算 XGBoost 分类器的精确率
xgb_precision = precision_score(y_test, xgb_pred)
# 计算 XGBoost 分类器的召回率
xgb_recall = recall_score(y_test, xgb_pred)
# 计算 XGBoost 分类器的 F1 值
xgb_f1 = f1_score(y_test, xgb_pred)
# 打印 XGBoost 分类器的评估指标标题
print("XGBoost 模型评估指标：")
# 打印 XGBoost 分类器的准确率，保留四位小数
print(f"准确率: {xgb_accuracy:.4f}")
# 打印 XGBoost 分类器的精确率，保留四位小数
print(f"精确率: {xgb_precision:.4f}")
# 打印 XGBoost 分类器的召回率，保留四位小数
print(f"召回率: {xgb_recall:.4f}")
# 打印 XGBoost 分类器的 F1 值，保留四位小数
print(f"F1 值: {xgb_f1:.4f}")

# 初始化 LightGBM 分类器，设置随机种子为 42
lgb_model = lgb.LGBMClassifier(random_state=42)
# 使用训练集数据对 LightGBM 分类器进行训练
lgb_model.fit(X_train, y_train)
# 使用训练好的 LightGBM 分类器对测试集数据进行预测
lgb_pred = lgb_model.predict(X_test)

# 打印 LightGBM 分类器的分类报告标题
print("\nLightGBM 分类报告：")
# 打印 LightGBM 分类器的分类报告
print(classification_report(y_test, lgb_pred))
# 打印 LightGBM 分类器的混淆矩阵标题
print("LightGBM 混淆矩阵：")
# 打印 LightGBM 分类器的混淆矩阵
print(confusion_matrix(y_test, lgb_pred))

# 计算 LightGBM 分类器的准确率
lgb_accuracy = accuracy_score(y_test, lgb_pred)
# 计算 LightGBM 分类器的精确率
lgb_precision = precision_score(y_test, lgb_pred)
# 计算 LightGBM 分类器的召回率
lgb_recall = recall_score(y_test, lgb_pred)
# 计算 LightGBM 分类器的 F1 值
lgb_f1 = f1_score(y_test, lgb_pred)
# 打印 LightGBM 分类器的评估指标标题
print("LightGBM 模型评估指标：")
# 打印 LightGBM 分类器的准确率，保留四位小数
print(f"准确率: {lgb_accuracy:.4f}")
# 打印 LightGBM 分类器的精确率，保留四位小数
print(f"精确率: {lgb_precision:.4f}")
# 打印 LightGBM 分类器的召回率，保留四位小数
print(f"召回率: {lgb_recall:.4f}")
print(f"F1 值: {lgb_f1:.4f}")