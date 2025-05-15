import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 设置全局字体为支持中文的字体 (例如 SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('python60-days-challenge\heart.csv')

# print(data.columns)
# print(data.head())
# print(data.info())
# print(data.describe())
# print(data.isnull().sum())

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# 定义离散变量列表
discrete_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

# 定义连续变量列表
continues_features = [col for col in columns if col not in discrete_features]

# print("离散变量:", discrete_features)
# print("连续变量:", continues_features)

# discrete_features = data.select_dtypes(include=['object']).columns.tolist()
# continues_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
#这里经过调试发现源数据是经过独热编码的，所以不能有效进行离散变量与连续变量的分离


for i in discrete_features:
    if data[i].isnull().sum() > 0:
        mode_value = data[i].mode()[0]
        print(f"特征 {i} 的众数为: {mode_value}")
        data[i].fillna(mode_value, inplace=True)

for i in continues_features:
    if data[i].isnull().sum() > 0:
        mean_value = data[i].mean()
        print(f"特征 {i} 的均值为: {mean_value}")
        data[i].fillna(mean_value, inplace=True)

# # 检查离散特征列表是否为空
# if discrete_features:
#     # 对离散特征进行独热编码
#     data = pd.get_dummies(data, columns=discrete_features, drop_first=True, dtype=int)
#进行独热编码之后得到更好的处理效果，便于处理数据
#便于计算机处理：计算机在处理数据时更擅长处理数字类型的数据。离散变量通常是一些分类数据，如颜色（红、绿、蓝）、性别（男、女）等，将其转换为独热编码后，就可以将这些分类信息表示为计算机易于处理的数字向量形式。
#例如，对于颜色变量，若采用独热编码，“红” 可能表示为 [1, 0, 0]，“绿” 表示为 [0, 1, 0]，“蓝” 表示为 [0, 0, 1]，这样计算机可以更方便地进行存储、计算和模型训练。

correlation_matrix = data.corr()
plt.rcParams['figure.dpi'] = 100
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()



# 绘制箱线图
# 遍历每个离散变量
for discrete_col in discrete_features: 
    # 遍历每个连续变量
    for continuous_col in continues_features: 
        plt.figure(figsize=(10, 6))      
        sns.boxplot(x=discrete_col, y=continuous_col, data=data)    
        plt.title(f'Boxplot of {continuous_col} by {discrete_col}')    
        plt.show()
# 该代码会生成所有离散变量与连续变量的组合的箱线图，
# 但由于离散变量的取值较多，可能会导致生成的图数量过多，
# 影响可读性和美观性。
# 为了避免这种情况，可以考虑对离散变量进行分组，
# 或者只选择部分离散变量进行分析。
# 此外，也可以考虑使用其他可视化工具，如平行坐标图或小提琴图，
# 来更清晰地展示离散变量与连续变量之间的关系。  



# 将所有图片显示在一个大图中大图包括所有的子图
# 计算子图的行数和列数
num_discrete = len(discrete_features)
num_continuous = len(continues_features)
num_subplots = num_discrete * num_continuous
num_rows = num_discrete
num_cols = num_continuous

# 创建一个包含所有子图的大图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

# 处理只有一个子图的特殊情况
if num_subplots == 1:
    axes = [[axes]]
elif num_rows == 1:
    axes = [axes]

# 遍历每个离散变量和连续变量
for i, discrete_col in enumerate(discrete_features): 
    for j, continuous_col in enumerate(continues_features): 
        sns.boxplot(x=discrete_col, y=continuous_col, data=data, ax=axes[i][j])    
        axes[i][j].set_title(f'Boxplot of {continuous_col} by {discrete_col}')    

# 调整子图之间的间距
plt.tight_layout()
plt.show()



# 绘制每个离散变量与连续变量的直方图
# 遍历每个离散变量
for discrete_col in discrete_features: 
    # 遍历每个连续变量
    for continuous_col in continues_features: 
        plt.figure(figsize=(10, 6))
        # 按照离散变量分组绘制连续变量的直方图
        for category in data[discrete_col].unique():
            subset = data[data[discrete_col] == category]
            sns.histplot(subset[continuous_col], label=category, kde=True)
        
        plt.title(f'Histogram of {continuous_col} by {discrete_col}')
        plt.xlabel(continuous_col)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

# 计算子图的行数和列数
num_discrete = len(discrete_features)
num_continuous = len(continues_features)
num_subplots = num_discrete * num_continuous
num_rows = num_discrete
num_cols = num_continuous

# 创建一个包含所有子图的大图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

# 处理只有一个子图的特殊情况
if num_subplots == 1:
    axes = [[axes]]
elif num_rows == 1:
    axes = [axes]

# 遍历每个离散变量和连续变量
for i, discrete_col in enumerate(discrete_features): 
    for j, continuous_col in enumerate(continues_features): 
        ax = axes[i][j]
        # 按照离散变量分组绘制连续变量的直方图
        for category in data[discrete_col].unique():
            subset = data[data[discrete_col] == category]
            sns.histplot(subset[continuous_col], label=category, kde=True, ax=ax)
        
        ax.set_title(f'Histogram of {continuous_col} by {discrete_col}')
        ax.set_xlabel(continuous_col)
        ax.set_ylabel('Frequency')
        ax.legend()

# 调整子图之间的间距
plt.tight_layout()
plt.show()


# 选择部分离散变量和连续变量
selected_discrete_features = ['sex', 'cp']  # 可根据需求修改
selected_continuous_features = ['age', 'trestbps']  # 可根据需求修改

# 计算子图的行数和列数
num_discrete = len(selected_discrete_features)
num_continuous = len(selected_continuous_features)
num_subplots = num_discrete * num_continuous
num_rows = num_discrete
num_cols = num_continuous

# 创建一个包含所有子图的大图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

# 处理只有一个子图的特殊情况
if num_subplots == 1:
    axes = [[axes]]
elif num_rows == 1:
    axes = [axes]

# 遍历每个选定的离散变量和连续变量
for i, discrete_col in enumerate(selected_discrete_features): 
    for j, continuous_col in enumerate(selected_continuous_features): 
        ax = axes[i][j]
        sns.violinplot(x=discrete_col, y=continuous_col, data=data, ax=ax)
        ax.set_title(f'Violinplot of {continuous_col} by {discrete_col}')
        ax.set_xlabel(discrete_col)
        ax.set_ylabel(continuous_col)

# 调整子图之间的间距
plt.tight_layout()
plt.show()



# 选择要绘制条形图的变量
selected_features = ['age', 'sex', 'cp']  # 可根据需求修改
num_features = len(selected_features)
num_rows = 1
num_cols = num_features

# 创建子图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))

# 处理只有一个子图的特殊情况
if num_features == 1:
    axes = [axes]

# 遍历每个变量并绘制条形图
for i, feature in enumerate(selected_features):
    if feature in discrete_features:
        # 离散变量条形图
        sns.countplot(x=feature, data=data, ax=axes[i])
    elif feature in continues_features:
        # 连续变量条形图，这里使用直方图近似
        sns.histplot(data[feature], ax=axes[i], kde=False)
    axes[i].set_title(f'Bar Plot of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Count')

# 调整子图之间的间距
plt.tight_layout()
plt.show()

