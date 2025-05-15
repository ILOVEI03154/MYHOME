# 导入 pandas 库，用于数据处理和分析
import pandas as pd
# 读取数据文件，假设数据文件名为 heart.csv，需根据实际情况修改文件路径
# 注意：在 Windows 中，反斜杠 \ 是转义字符，这里使用原始字符串避免转义问题
data = pd.read_csv('python60-days-challenge\heart.csv')

# 默认参数模型(基准模型)

# 从 sklearn 库的 model_selection 模块导入 train_test_split 函数，用于划分训练集和测试集
from sklearn.model_selection import train_test_split
# 从数据集中删除 'target' 列，将剩余的特征数据存储在变量 x 中
x = data.drop(["target"], axis=1)
# 提取 'target' 列作为目标变量，存储在变量 y 中
y = data["target"]
# 使用 train_test_split 函数将数据集划分为训练集和测试集，测试集占比 20%，随机种子设置为 42 以保证结果可复现
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 从 sklearn 库的 ensemble 模块导入 RandomForestClassifier 类，用于创建随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 从 sklearn 库的 metrics 模块导入 classification_report 函数，用于生成分类报告
from sklearn.metrics import classification_report
# 导入 time 模块，用于记录时间
import time
# 记录开始时间
start_time = time.time() 
# 创建一个随机森林分类器实例，设置随机种子为 42 以保证结果可复现
rf_model = RandomForestClassifier(random_state=42)
# 使用训练集数据对随机森林分类器进行训练
rf_model.fit(x_train, y_train) 
# 使用训练好的随机森林分类器对测试集数据进行预测
rf_pred = rf_model.predict(x_test)
# 记录结束时间
end_time = time.time() 
# 打印训练与预测所花费的时间，保留四位小数
print(f"训练与预测耗时: {end_time - start_time:.4f} 秒")
# 打印随机森林分类器在测试集上的分类报告
print(classification_report(y_test, rf_pred))

# 1. 开始网格搜索部分
# 从 sklearn 库的 model_selection 模块导入 GridSearchCV 类，用于进行网格搜索调参
from sklearn.model_selection import GridSearchCV
# 再次导入 RandomForestClassifier 类，虽然之前已导入，但这里为了代码结构清晰再次导入
from sklearn.ensemble import RandomForestClassifier
# 再次导入 classification_report 函数，用于生成分类报告
from sklearn.metrics import classification_report
# 再次导入 time 模块，用于记录时间
import time

# 2. 定义网格搜索的参数空间
param_grid = {
    # 随机森林中树的数量，尝试 50、100 和 200 这三个值
    'n_estimators': [50, 100, 200],
    # 树的最大深度，尝试 None（不限制深度）、10、20 和 30 这几个值
    'max_depth': [None, 10, 20, 30],
    # 拆分内部节点所需的最小样本数，尝试 2、5 和 10 这三个值
    'min_samples_split': [2, 5, 10],
    # 叶子节点所需的最小样本数，尝试 1、2 和 4 这三个值
    'min_samples_leaf': [1, 2, 4]
}

# 3. 创建 GridSearchCV 实例
grid_search = GridSearchCV(
    # 要调参的模型，这里是随机森林分类器，设置随机种子为 42 以保证结果可复现
    estimator=RandomForestClassifier(random_state=42),
    # 定义好的参数空间
    param_grid=param_grid, 
    # 交叉验证的折数，设置为 5 折交叉验证
    cv=5, 
    # 使用所有可用的 CPU 核心进行并行计算
    n_jobs=-1, 
    # 评估指标，这里使用准确率
    scoring='accuracy'
) 

# 4. 开始网格搜索并记录时间
start_time = time.time()
# 使用训练集数据进行网格搜索
grid_search.fit(x_train, y_train) 
end_time = time.time()
# 打印网格搜索所花费的时间，保留四位小数
print(f"网格搜索耗时: {end_time - start_time:.4f} 秒")
# 打印网格搜索得到的最佳参数
print(f"最佳参数: {grid_search.best_params_}")

# 5. 获取网格搜索得到的最佳模型
best_model = grid_search.best_estimator_ 
# 使用最佳模型对测试集数据进行预测
best_pred = best_model.predict(x_test) 

# 6. 打印最佳模型在测试集上的分类报告
print(classification_report(y_test, best_pred))

# 1. 开始贝叶斯优化部分
# 从 skopt 库导入 BayesSearchCV 类，用于进行贝叶斯优化调参
from skopt import BayesSearchCV
# 从 skopt.space 模块导入 Integer 类，用于定义整数参数空间
from skopt.space import Integer
# 再次导入 RandomForestClassifier 类，用于创建随机森林分类器
from sklearn.ensemble import RandomForestClassifier
# 再次导入 classification_report 函数，用于生成分类报告
from sklearn.metrics import classification_report
# 再次导入 time 模块，用于记录时间
import time

# 2. 定义贝叶斯优化的搜索空间
search_space = {
    # 随机森林中树的数量，范围在 50 到 200 之间的整数
    'n_estimators': Integer(50, 200),
    # 树的最大深度，范围在 10 到 30 之间的整数
    'max_depth': Integer(10, 30),
    # 拆分内部节点所需的最小样本数，范围在 2 到 10 之间的整数
    'min_samples_split': Integer(2, 10),
    # 叶子节点所需的最小样本数，范围在 1 到 4 之间的整数
    'min_samples_leaf': Integer(1, 4)
}

# 3. 创建 BayesSearchCV 实例
bayes_search = BayesSearchCV(
    # 要调参的模型，这里是随机森林分类器，设置随机种子为 42 以保证结果可复现
    estimator=RandomForestClassifier(random_state=42),
    # 定义好的搜索空间
    search_spaces=search_space,
    # 贝叶斯优化的迭代次数，设置为 32 次
    n_iter=32, 
    # 交叉验证的折数，设置为 5 折交叉验证
    cv=5, 
    # 使用所有可用的 CPU 核心进行并行计算
    n_jobs=-1,
    # 评估指标，这里使用准确率
    scoring='accuracy'
)
# 记录开始时间
start_time = time.time()
# 4. 使用训练集数据进行贝叶斯优化
bayes_search.fit(x_train, y_train)
# 记录结束时间
end_time = time.time()
# 打印贝叶斯优化所花费的时间，保留四位小数
print(f"贝叶斯优化耗时: {end_time - start_time:.4f} 秒")
# 打印贝叶斯优化得到的最佳参数
print(f"最佳参数:{bayes_search.best_params_}")

# 5. 获取贝叶斯优化得到的最佳模型
best_model = bayes_search.best_estimator_
# 使用最佳模型对测试集数据进行预测
best_pred = best_model.predict(x_test)

# 6. 打印贝叶斯优化后的随机森林在测试集上的分类报告
print("\n贝叶斯优化后的随机森林 在测试集上的分类报告：")
print(classification_report(y_test, best_pred))