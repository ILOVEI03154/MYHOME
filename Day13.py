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

# 基准模型
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

# 过采样
# 随机采样和SMOTE随机过采样是从少数类中随机选择样本，并将其复制后添加到训练集。
# 随机过采样的步骤如下：
# 1. 确定少数类的样本数。
# 2. 从少数类中随机选择样本，并将其复制。
# 3. 将复制的样本添加到训练集。
# 随机过采样的优点是，它可以增加少数类的样本数，从而提高模型的泛化能力。
# 随机过采样的缺点是，它可能会增加训练集的大小，从而增加训练时间。此外，它可能会增加噪声，并且可能会增加模型的偏差。

# 随机过采样
# 导入随机过采样函数，imblearn库是专门用于处理不平衡数据集的库，imblearn.over_sampling是专门用于过采样的库，RandomOverSampler是随机过采样函数，random_state=42表示随机种子，保证每次运行结果一致
from imblearn.over_sampling import RandomOverSampler 
# 随机过采样函数，random_state=42表示随机种子，保证每次运行结果一致
ros = RandomOverSampler(random_state=42) 
# 在训练集上进行过采样，fit_resample是过采样函数，返回的是过采样后的训练集和标签集，random_state=42表示随机种子，保证每次运行结果一致
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train) 
# 打印过采样后训练集的形状
print("过采样后训练集的形状：",X_train_ros.shape,y_train_ros.shape) 

# 训练随机森林模型（使用随机过采样后的训练集）
# 随机森林分类器，random_state=42表示随机种子，保证每次运行结果一致
rf_model_ros = RandomForestClassifier(random_state=42) 
# 记录开始时间
start_time = time.time() 
# 在过采样后的训练集上训练随机森林模型
rf_model_ros.fit(X_train_ros, y_train_ros) 
# 记录结束时间
end_time = time.time() 
# 打印训练与预测耗时
print(f"训练与预测耗时: {end_time - start_time:.4f} 秒") 
# 在测试集上进行预测
rf_pred_ros = rf_model_ros.predict(X_test) 
# 打印随机过采样后的随机森林在测试集上的分类报告
print("\n随机过采样后的随机森林 在测试集上的分类报告：") 
print(classification_report(y_test, rf_pred_ros)) 
# 打印随机过采样后随机森林在测试集上的混淆矩阵
print("随机过采样后随机森林 在测试集上的混淆矩阵：")
print(confusion_matrix(y_test, rf_pred_ros)) 

# SMOTE
# SMOTE是一种过采样方法，它通过插值的方式生成新的少数类样本。
# SMOTE的步骤如下：
# 1. 确定少数类的样本数。
# 2. 从少数类中随机选择一个样本。
# 3. 计算该样本与最近邻样本之间的距离。
# 4. 在该样本和最近邻样本之间生成一个新的样本。
# 5. 将新的样本添加到训练集。
# SMOTE的优点是，它可以增加少数类的样本数，从而提高模型的泛化能力。
# SMOTE的缺点是，它可能会增加噪声，并且可能会增加模型的偏差。

# SMOTE
# 导入SMOTE函数，imblearn库是专门用于处理不平衡数据集的库，imblearn.over_sampling是专门用于过采样的库，SMOTE是SMOTE函数，random_state=42表示随机种子，保证每次运行结果一致
from imblearn.over_sampling import SMOTE 
# SMOTE函数，random_state=42表示随机种子，保证每次运行结果一致
smote = SMOTE(random_state=42) 
# 在训练集上进行过采样，fit_resample是过采样函数，返回的是过采样后的训练集和标签集，random_state=42表示随机种子，保证每次运行结果一致
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train) 
# 打印SMOTE过采样后训练集的形状
print("SMOTE过采样后训练集的形状：",X_train_smote.shape,y_train_smote.shape) 

# 训练随机森林模型（使用SMOTE过采样后的训练集）
# 随机森林分类器，random_state=42表示随机种子，保证每次运行结果一致
rf_model_smote = RandomForestClassifier(random_state=42) 
# 记录开始时间
start_time = time.time() 
# 在过采样后的训练集上训练随机森林模型
rf_model_smote.fit(X_train_smote, y_train_smote) 
# 记录结束时间
end_time = time.time() 
# 打印训练与预测耗时
print(f"训练与预测耗时: {end_time - start_time:.4f} 秒") 
# 在测试集上进行预测
rf_pred_smote = rf_model_smote.predict(X_test) 
# 打印SMOTE过采样后的随机森林在测试集上的分类报告
print("\nSMOTE过采样后的随机森林 在测试集上的分类报告：") 
print(classification_report(y_test, rf_pred_smote)) 
# 打印SMOTE过采样后随机森林在测试集上的混淆矩阵
print("SMOTE过采样后随机森林 在测试集上的混淆矩阵：")
print(confusion_matrix(y_test, rf_pred_smote)) 

# --- 1. 默认参数随机森林 (训练集 -> 测试集) ---
print("--- 1. 默认参数随机森林 (训练集 -> 测试集) ---")
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
print("^" * 50)

# --- 2. 带权重的随机森林 + 交叉验证 (在训练集上进行CV) ---
print("--- 2. 带权重随机森林 + 交叉验证 (在训练集上进行) ---")
# 确定少数类标签 (非常重要！)
# 假设是二分类问题，我们需要知道哪个是少数类标签才能正确解读 recall, precision, f1
# 例如，如果标签是 0 和 1，可以这样查看：
# 统计每个标签的数量
counts = np.bincount(y_train) 
# 找到计数最少的类别的标签
minority_label = np.argmin(counts) 
# 找到计数最多的类别的标签
majority_label = np.argmax(counts) 
# 打印训练集中各类别数量
print(f"训练集中各类别数量: {counts}") 
# 打印少数类标签和多数类标签
print(f"少数类标签: {minority_label}, 多数类标签: {majority_label}") 
#!!下面的 scorer 将使用这个 minority_label!!
# 定义带权重的模型
rf_model_weighted = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)
# 新增 StratifiedKFold 导入
from sklearn.model_selection import StratifiedKFold  
# 设置交叉验证策略 (使用 StratifiedKFold 保证每折类别比例相似)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # 5折交叉验证
# 定义用于交叉验证的评估指标
# 特别关注少数类的指标，使用 make_scorer 指定 pos_label
# 注意：如果你的少数类标签不是 1，需要修改 pos_label
scoring = {
    'accuracy': 'accuracy', # 准确率
    'precision_minority': make_scorer(precision_score, pos_label=minority_label, zero_division=0), # 精确度
    'recall_minority': make_scorer(recall_score, pos_label=minority_label), # 召回率
    'f1_minority': make_scorer(f1_score, pos_label=minority_label) # F1分数
}
# 新增 cross_validate 导入
from sklearn.model_selection import cross_validate 
# 打印开始交叉验证
print(f"开始进行 {cv_strategy.get_n_splits()} 折交叉验证...") 
# 记录开始时间
start_time_cv = time.time() 
# 执行交叉验证 (在 X_train, y_train 上进行) 
results = cross_validate(
    rf_model_weighted, # 模型
    X_train, y_train, # 训练集
    cv=cv_strategy, # 交叉验证策略
    scoring=scoring, # 评估指标
    return_train_score=True # 返回训练集上的分数
)
# 记录结束时间
end_time_cv = time.time() 
# 打印交叉验证耗时
print(f"交叉验证耗时: {end_time_cv - start_time_cv:.4f} 秒") 
# 打印交叉验证结果
print("交叉验证结果：") 
# 打印交叉验证结果的平均值
print("\n带权重随机森林 交叉验证平均性能 (基于训练集划分)：")
# 遍历评估指标
for metric_name, scores in results.items(): 
    if metric_name.startswith('test_'): # 我们关心的是在验证折上的表现
         # 提取指标名称（去掉 'test_' 前缀）
        metric_name = metric_name.split('test_')[1] 
        # 打印平均值和标准差
        print(f"{metric_name}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})") 
print("-" * 50) # 打印分隔线

# --- 3. 带权重的随机森林 + 交叉验证 (验证集上进行) ---
print("--- 3. 带权重随机森林 + 交叉验证 (验证集上进行) ---")
# 记录开始时间
start_time_cv = time.time() 
# 定义带权重的随机森林模型
rf_model_weighted = RandomForestClassifier(
    random_state=42,
    class_weight='balanced' # 关键：自动根据类别频率调整权重
    # class_weight={minority_label: 10, majority_label: 1} # 或者可以手动设置权重字典
)

# 在训练集上训练模型
rf_model_weighted.fit(X_train, y_train) 
# 在测试集上进行预测
rf_pred_weighted = rf_model_weighted.predict(X_test) 
# 记录结束时间
end_time_cv = time.time() 
# 打印训练与预测耗时
print(f"训练与预测耗时: {end_time_cv - start_time_cv:.4f} 秒") 
# 打印带权重随机森林在测试集上的分类报告
print("\n带权重随机森林 在测试集上的分类报告：") 
print(classification_report(y_test, rf_pred_weighted)) 
# 打印带权重随机森林在测试集上的混淆矩阵
print("带权重随机森林 在测试集上的混淆矩阵：") 
print(confusion_matrix(y_test, rf_pred_weighted)) 
print("-" * 50) # 打印分隔线

# 对比总结
print("--- 4. 对比总结 ---")
print("性能指标对比(测试集上的少数召回率Recall):")
print(f"默认随机森林: {recall_score(y_test, rf_pred, pos_label=minority_label):.4f}")
print(f"随机过采样后随机森林: {recall_score(y_test, rf_pred_ros, pos_label=minority_label):.4f}")