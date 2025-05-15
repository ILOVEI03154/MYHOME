# 导入 pandas 库，用于数据处理和分析，可处理表格数据
# 注意：这里重复导入了 pandas，可删除一处
import pandas as pd
import pandas as pd    #用于数据处理和分析，可处理表格数据。
# 导入 numpy 库，用于数值计算，提供高效的数组操作
import numpy as np     
# 导入 matplotlib.pyplot 库，用于绘制各种类型的图表
import matplotlib.pyplot as plt    
# 导入 seaborn 库，基于 matplotlib 的高级绘图库，能绘制更美观的统计图形
import seaborn as sns   

# 设置中文字体（解决中文显示问题）
# Windows 系统常用黑体字体
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False    
# 读取数据文件，需根据实际情况修改文件路径
data = pd.read_csv('python60-days-challenge\python-learning-library\data.csv')    

# 先筛选字符串变量 
# 选择数据中类型为 object 的列，并将列名转换为列表
discrete_features = data.select_dtypes(include=['object']).columns.tolist()

# Home Ownership 标签编码
# 定义 Home Ownership 列的映射关系
home_ownership_mapping = {
    'Own Home': 1,
    'Rent': 2,
    'Have Mortgage': 3,
    'Home Mortgage': 4
}
# 使用映射关系对 Home Ownership 列进行编码
data['Home Ownership'] = data['Home Ownership'].map(home_ownership_mapping)

# Years in current job 标签编码
# 定义 Years in current job 列的映射关系
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
# 使用映射关系对 Years in current job 列进行编码
data['Years in current job'] = data['Years in current job'].map(years_in_job_mapping)

# Purpose 独热编码，记得需要将 bool 类型转换为数值
# 对 Purpose 列进行独热编码
data = pd.get_dummies(data, columns=['Purpose'])
# 重新读取数据，用来做列名对比
data2 = pd.read_csv("python60-days-challenge\python-learning-library\data.csv") 
# 新建一个空列表，用于存放独热编码后新增的特征名
list_final = [] 
# 遍历数据的列名
for i in data.columns:
    if i not in data2.columns:
        # 这里打印出来的就是独热编码后的特征名
        list_final.append(i) 
# 将独热编码后的列的数据类型转换为 int
for i in list_final:
    data[i] = data[i].astype(int) 

# Term 0 - 1 映射
# 定义 Term 列的映射关系
term_mapping = {
    'Short Term': 0,
    'Long Term': 1
}
# 使用映射关系对 Term 列进行编码
data['Term'] = data['Term'].map(term_mapping)
# 重命名列
data.rename(columns={'Term': 'Long Term'}, inplace=True) 
# 把筛选出来的数值类型的列名转换成列表
continuous_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()  

# 连续特征用众数补全
for feature in continuous_features:     
    # 获取该列的众数
    mode_value = data[feature].mode()[0]            
    # 用众数填充该列的缺失值，inplace=True 表示直接在原数据上修改
    data[feature].fillna(mode_value, inplace=True)          

# 最开始也说了 很多调参函数自带交叉验证，甚至是必选的参数，你如果想要不交叉反而实现起来会麻烦很多
# 所以这里我们还是只划分一次数据集
# 从 sklearn 库的 model_selection 模块导入 train_test_split 函数
from sklearn.model_selection import train_test_split
# 特征，axis=1 表示按列删除
X = data.drop(['Credit Default'], axis=1)  
# 标签
y = data['Credit Default'] 
# 按照 8:2 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# 从 sklearn 库的 ensemble 模块导入随机森林分类器
from sklearn.ensemble import RandomForestClassifier 

# 从 sklearn 库的 metrics 模块导入用于评估分类器性能的指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
# 从 sklearn 库的 metrics 模块导入用于生成分类报告和混淆矩阵的函数
from sklearn.metrics import classification_report, confusion_matrix 
# 导入 warnings 库，用于忽略警告信息
import warnings 
# 忽略所有警告信息
warnings.filterwarnings("ignore") 

# --- 1. 默认参数的随机森林 ---
# 评估基准模型，这里确实不需要验证集
print("--- 1. 默认参数随机森林 (训练集 -> 测试集) ---")
# 导入 time 库，主要用于时间相关的操作，因为调参需要很长时间，记录下会帮助后人知道大概的时长
import time 
# 记录开始时间
start_time = time.time() 
# 创建随机森林分类器实例，设置随机种子以保证结果可复现
rf_model = RandomForestClassifier(random_state=42)
# 在训练集上训练模型
rf_model.fit(X_train, y_train) 
# 在测试集上进行预测
rf_pred = rf_model.predict(X_test) 
# 记录结束时间
end_time = time.time() 

# 打印训练与预测耗时，保留四位小数
print(f"训练与预测耗时: {end_time - start_time:.4f} 秒")
print("\n默认随机森林 在测试集上的分类报告：")
# 打印分类报告
print(classification_report(y_test, rf_pred))
print("默认随机森林 在测试集上的混淆矩阵：")
# 打印混淆矩阵
print(confusion_matrix(y_test, rf_pred))

# 再次导入相关库，可优化导入逻辑，避免重复导入
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")
import time
# 导入 DEAP 库，用于遗传算法和进化计算
from deap import base, creator, tools, algorithms 
import random
import numpy as np

# --- 2. 遗传算法优化随机森林 ---
print("\n--- 2. 遗传算法优化随机森林 (训练集 -> 测试集) ---")

# 定义适应度函数和个体类型
# 创建一个最大化适应度的类型
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 创建一个个体类型，继承自列表，包含适应度属性
creator.create("Individual", list, fitness=creator.FitnessMax)

# 定义超参数范围
n_estimators_range = (50, 200)
max_depth_range = (10, 30)
min_samples_split_range = (2, 10)
min_samples_leaf_range = (1, 4)

# 初始化工具盒
toolbox = base.Toolbox()

# 定义基因生成器
# 注册生成 n_estimators 参数的函数
toolbox.register("attr_n_estimators", random.randint, *n_estimators_range)
# 注册生成 max_depth 参数的函数
toolbox.register("attr_max_depth", random.randint, *max_depth_range)
# 注册生成 min_samples_split 参数的函数
toolbox.register("attr_min_samples_split", random.randint, *min_samples_split_range)
# 注册生成 min_samples_leaf 参数的函数
toolbox.register("attr_min_samples_leaf", random.randint, *min_samples_leaf_range)

# 定义个体生成器
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_n_estimators, toolbox.attr_max_depth,
                  toolbox.attr_min_samples_split, toolbox.attr_min_samples_leaf), n=1)

# 定义种群生成器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 定义评估函数
def evaluate(individual):
    # 解包个体中的参数
    n_estimators, max_depth, min_samples_split, min_samples_leaf = individual
    # 创建随机森林分类器实例
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42)
    # 在训练集上训练模型
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,

# 注册评估函数
toolbox.register("evaluate", evaluate)

# 注册遗传操作
# 注册两点交叉操作
toolbox.register("mate", tools.cxTwoPoint)
# 注册均匀整数变异操作
toolbox.register("mutate", tools.mutUniformInt, low=[n_estimators_range[0], max_depth_range[0],
                                                     min_samples_split_range[0], min_samples_leaf_range[0]],
                 up=[n_estimators_range[1], max_depth_range[1],
                     min_samples_split_range[1], min_samples_leaf_range[1]], indpb=0.1)
# 注册锦标赛选择操作
toolbox.register("select", tools.selTournament, tournsize=3)

# 初始化种群
pop = toolbox.population(n=20)

# 遗传算法参数
# 迭代次数
NGEN = 10
# 交叉概率
CXPB = 0.5
# 变异概率
MUTPB = 0.2

# 记录开始时间
start_time = time.time()
# 运行遗传算法
for gen in range(NGEN):
    # 生成后代
    offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
    # 评估后代的适应度
    fits = toolbox.map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    # 选择下一代种群
    pop = toolbox.select(offspring, k=len(pop))

# 记录结束时间
end_time = time.time()

# 找到最优个体
best_ind = tools.selBest(pop, k=1)[0]
best_n_estimators, best_max_depth, best_min_samples_split, best_min_samples_leaf = best_ind

# 打印遗传算法优化耗时，保留四位小数
print(f"遗传算法优化耗时: {end_time - start_time:.4f} 秒")
print("最佳参数: ", {
    'n_estimators': best_n_estimators,
    'max_depth': best_max_depth,
    'min_samples_split': best_min_samples_split,
    'min_samples_leaf': best_min_samples_leaf
})

# 使用最佳参数的模型进行预测
best_model = RandomForestClassifier(n_estimators=best_n_estimators,
                                    max_depth=best_max_depth,
                                    min_samples_split=best_min_samples_split,
                                    min_samples_leaf=best_min_samples_leaf,
                                    random_state=42)
# 在训练集上训练模型
best_model.fit(X_train, y_train)
# 在测试集上进行预测
best_pred = best_model.predict(X_test)

print("\n遗传算法优化后的随机森林 在测试集上的分类报告：")
# 打印分类报告
print(classification_report(y_test, best_pred))
print("遗传算法优化后的随机森林 在测试集上的混淆矩阵：")
# 打印混淆矩阵
print(confusion_matrix(y_test, best_pred))

# --- 2. 粒子群优化算法优化随机森林 ---
print("\n--- 2. 粒子群优化算法优化随机森林 (训练集 -> 测试集) ---")

# 定义适应度函数，本质就是构建了一个函数实现 参数--> 评估指标的映射
def fitness_function(params): 
    # 序列解包，允许你将一个可迭代对象（如列表、元组、字符串等）中的元素依次赋值给多个变量
    n_estimators, max_depth, min_samples_split, min_samples_leaf = params 
    # 创建随机森林分类器实例
    model = RandomForestClassifier(n_estimators=int(n_estimators),
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   random_state=42)
    # 在训练集上训练模型
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 粒子群优化算法实现
def pso(num_particles, num_iterations, c1, c2, w, bounds): 
    # num_particles：粒子的数量，即算法中用于搜索最优解的个体数量。
    # num_iterations：迭代次数，算法运行的最大循环次数。
    # c1：认知学习因子，用于控制粒子向自身历史最佳位置移动的程度。
    # c2：社会学习因子，用于控制粒子向全局最佳位置移动的程度。
    # w：惯性权重，控制粒子的惯性，影响粒子在搜索空间中的移动速度和方向。
    # bounds：超参数的取值范围，是一个包含多个元组的列表，每个元组表示一个超参数的最小值和最大值。

    # 超参数的数量
    num_params = len(bounds) 
    # 初始化粒子位置
    particles = np.array([[random.uniform(bounds[i][0], bounds[i][1]) for i in range(num_params)] for _ in
                          range(num_particles)])
    # 初始化粒子速度
    velocities = np.array([[0] * num_params for _ in range(num_particles)])
    # 初始化个体历史最佳位置
    personal_best = particles.copy()
    # 初始化个体历史最佳适应度
    personal_best_fitness = np.array([fitness_function(p) for p in particles])
    # 找到全局最佳位置的索引
    global_best_index = np.argmax(personal_best_fitness)
    # 初始化全局最佳位置
    global_best = personal_best[global_best_index]
    # 初始化全局最佳适应度
    global_best_fitness = personal_best_fitness[global_best_index]

    for _ in range(num_iterations):
        # 生成随机数 r1
        r1 = np.array([[random.random() for _ in range(num_params)] for _ in range(num_particles)])
        # 生成随机数 r2
        r2 = np.array([[random.random() for _ in range(num_params)] for _ in range(num_particles)])

        # 更新粒子速度
        velocities = w * velocities + c1 * r1 * (personal_best - particles) + c2 * r2 * (
                global_best - particles)
        # 更新粒子位置
        particles = particles + velocities

        for i in range(num_particles):
            for j in range(num_params):
                # 确保粒子位置在边界内
                if particles[i][j] < bounds[j][0]:
                    particles[i][j] = bounds[j][0]
                elif particles[i][j] > bounds[j][1]:
                    particles[i][j] = bounds[j][1]

        # 计算新的适应度值
        fitness_values = np.array([fitness_function(p) for p in particles])
        # 找到适应度提高的粒子索引
        improved_indices = fitness_values > personal_best_fitness
        # 更新个体历史最佳位置
        personal_best[improved_indices] = particles[improved_indices]
        # 更新个体历史最佳适应度
        personal_best_fitness[improved_indices] = fitness_values[improved_indices]

        # 找到当前全局最佳位置的索引
        current_best_index = np.argmax(personal_best_fitness)
        if personal_best_fitness[current_best_index] > global_best_fitness:
            # 更新全局最佳位置
            global_best = personal_best[current_best_index]
            # 更新全局最佳适应度
            global_best_fitness = personal_best_fitness[current_best_index]

    return global_best, global_best_fitness

# 超参数范围
bounds = [(50, 200), (10, 30), (2, 10), (1, 4)]  # n_estimators, max_depth, min_samples_split, min_samples_leaf

# 粒子群优化算法参数
# 粒子数量
num_particles = 20
# 迭代次数
num_iterations = 10
# 认知学习因子
c1 = 1.5
# 社会学习因子
c2 = 1.5
# 惯性权重
w = 0.5

# 记录开始时间
start_time = time.time()
# 运行粒子群优化算法
best_params, best_fitness = pso(num_particles, num_iterations, c1, c2, w, bounds)
# 记录结束时间
end_time = time.time()

# 打印粒子群优化算法优化耗时，保留四位小数
print(f"粒子群优化算法优化耗时: {end_time - start_time:.4f} 秒")
print("最佳参数: ", {
    'n_estimators': int(best_params[0]),
    'max_depth': int(best_params[1]),
    'min_samples_split': int(best_params[2]),
    'min_samples_leaf': int(best_params[3])
})

# 使用最佳参数的模型进行预测
best_model = RandomForestClassifier(n_estimators=int(best_params[0]),
                                    max_depth=int(best_params[1]),
                                    min_samples_split=int(best_params[2]),
                                    min_samples_leaf=int(best_params[3]),
                                    random_state=42)
# 在训练集上训练模型
best_model.fit(X_train, y_train)
# 在测试集上进行预测
best_pred = best_model.predict(X_test)

print("\n粒子群优化算法优化后的随机森林 在测试集上的分类报告：")
# 打印分类报告
print(classification_report(y_test, best_pred))
print("粒子群优化算法优化后的随机森林 在测试集上的混淆矩阵：")
# 打印混淆矩阵
print(confusion_matrix(y_test, best_pred))

# --- 2. 模拟退火算法优化随机森林 ---
print("\n--- 2. 模拟退火算法优化随机森林 (训练集 -> 测试集) ---")

# 定义适应度函数
def fitness_function(params): 
    # 序列解包，将参数依次赋值给变量
    n_estimators, max_depth, min_samples_split, min_samples_leaf = params
    # 创建随机森林分类器实例
    model = RandomForestClassifier(n_estimators=int(n_estimators),
                                   max_depth=int(max_depth),
                                   min_samples_split=int(min_samples_split),
                                   min_samples_leaf=int(min_samples_leaf),
                                   random_state=42)
    # 在训练集上训练模型
    model.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# 模拟退火算法实现
def simulated_annealing(initial_solution, bounds, initial_temp, final_temp, alpha):
    # 当前解
    current_solution = initial_solution
    # 当前解的适应度
    current_fitness = fitness_function(current_solution)
    # 最佳解
    best_solution = current_solution
    # 最佳解的适应度
    best_fitness = current_fitness
    # 当前温度
    temp = initial_temp

    while temp > final_temp:
        # 生成邻域解
        neighbor_solution = []
        for i in range(len(current_solution)):
            # 生成新的值
            new_val = current_solution[i] + random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0]) * 0.1
            # 确保新的值在边界内
            new_val = max(bounds[i][0], min(bounds[i][1], new_val))
            neighbor_solution.append(new_val)

        # 计算邻域解的适应度
        neighbor_fitness = fitness_function(neighbor_solution)
        # 计算适应度差值
        delta_fitness = neighbor_fitness - current_fitness

        if delta_fitness > 0 or random.random() < np.exp(delta_fitness / temp):
            # 更新当前解
            current_solution = neighbor_solution
            # 更新当前解的适应度
            current_fitness = neighbor_fitness

        if current_fitness > best_fitness:
            # 更新最佳解
            best_solution = current_solution
            # 更新最佳解的适应度
            best_fitness = current_fitness

        # 降温
        temp *= alpha

    return best_solution, best_fitness

# 超参数范围
bounds = [(50, 200), (10, 30), (2, 10), (1, 4)]  # n_estimators, max_depth, min_samples_split, min_samples_leaf

# 模拟退火算法参数
# 初始温度
initial_temp = 100 
# 终止温度
final_temp = 0.1 
# 温度衰减系数
alpha = 0.95 

# 初始化初始解
initial_solution = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]

# 记录开始时间
start_time = time.time()
# 运行模拟退火算法
best_params, best_fitness = simulated_annealing(initial_solution, bounds, initial_temp, final_temp, alpha)
# 记录结束时间
end_time = time.time()

# 打印模拟退火算法优化耗时，保留四位小数
print(f"模拟退火算法优化耗时: {end_time - start_time:.4f} 秒")
print("最佳参数: ", {
    'n_estimators': int(best_params[0]),
    'max_depth': int(best_params[1]),
    'min_samples_split': int(best_params[2]),
    'min_samples_leaf': int(best_params[3])
})

# 使用最佳参数的模型进行预测
best_model = RandomForestClassifier(n_estimators=int(best_params[0]),
                                    max_depth=int(best_params[1]),
                                    min_samples_split=int(best_params[2]),
                                    min_samples_leaf=int(best_params[3]),
                                    random_state=42)
# 在训练集上训练模型
best_model.fit(X_train, y_train)
# 在测试集上进行预测
best_pred = best_model.predict(X_test)

print("\n模拟退火算法优化后的随机森林 在测试集上的分类报告：")
# 打印分类报告
print(classification_report(y_test, best_pred))
print("模拟退火算法优化后的随机森林 在测试集上的混淆矩阵：")
# 打印混淆矩阵
print(confusion_matrix(y_test, best_pred))