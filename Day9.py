# 导入 pandas 库，用于数据处理和分析
import pandas as pd 
# 从指定路径读取 CSV 文件，并将数据存储在 data 变量中
data = pd.read_csv("python60-days-challenge\python-learning-library\data.csv") 
# 打印数据的基本信息，包括列名、数据类型、非空值数量等
print(data.info()) 

# 打印数据集的前 5 行，用于快速查看数据的结构和内容
print(data.head(5)) 

# 统计 'Years in current job' 列中每个唯一值的出现次数
print(data["Years in current job"].value_counts()) 

# 统计 'Home Ownership' 列中每个唯一值的出现次数
print(data["Home Ownership"].value_counts()) 

# 创建嵌套字典用于映射，将分类变量转换为数值变量
mappings = { 
    # 对 'Years in current job' 列的映射规则
    "Years in current job": { 
        "10+ years": 10, 
        "2 years": 2, 
        "3 years": 3, 
        "< 1 year": 0, 
        "5 years": 5, 
        "1 year": 1, 
        "4 years": 4, 
        "6 years": 6, 
        "7 years": 7, 
        "8 years": 8, 
        "9 years": 9 
    }, 
    # 对 'Home Ownership' 列的映射规则
    "Home Ownership": { 
        "Home Mortgage": 0, 
        "Rent": 1, 
        "Own Home": 2, 
        "Have Mortgage": 3 
    } 
} 
# 使用 mappings 字典对 'Years in current job' 列进行映射转换
data["Years in current job"] = data["Years in current job"].map(mappings["Years in current job"]) 
# 使用 mappings 字典对 'Home Ownership' 列进行映射转换
data["Home Ownership"] = data["Home Ownership"].map(mappings["Home Ownership"]) 
# 打印映射转换后数据的基本信息
print(data.info()) 

# 打印映射转换后数据集的前几行
print(data.head()) 

# 打印数据集的所有列名
print(data.columns) 

# 导入 matplotlib 的 pyplot 模块，用于数据可视化
import matplotlib.pyplot as plt 
# 导入 seaborn 库，用于更美观的统计图表绘制
import seaborn as sns 

# 提取连续值特征，存储在列表中
continuous_features = [ 
    'Annual Income', 'Years in current job', 'Tax Liens', 
    'Number of Open Accounts', 'Years of Credit History', 
    'Maximum Open Credit', 'Number of Credit Problems', 
    'Months since last delinquent', 'Bankruptcies', 
    'Current Loan Amount', 'Current Credit Balance', 'Monthly Debt', 
    'Credit Score' 
] 

# 计算连续值特征之间的相关系数矩阵
correlation_matrix = data[continuous_features].corr() 

# 设置图片的清晰度为 100 dpi
plt.rcParams['figure.dpi'] = 100 

# 创建一个新的图形，设置图形的大小为 12x10 英寸
plt.figure(figsize=(12, 10)) 
# 绘制相关系数矩阵的热力图，annot=True 表示显示每个格子的相关系数值，cmap='coolwarm' 表示使用冷暖色调，vmin 和 vmax 设定颜色映射的范围
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1) 
# 设置热力图的标题
plt.title('Correlation Heatmap of Continuous Features') 
# 显示绘制好的热力图
plt.show() 

# 再次导入 pandas 库，虽然重复导入不影响运行，但通常建议只导入一次
import pandas as pd 
# 再次导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt 
# 再次导入 seaborn 库
import seaborn as sns 
# 定义要绘制箱线图的特征列表
features = ['Annual Income', 'Years in current job', 'Tax Liens', 'Number of Open Accounts'] 

# 再次设置图片的清晰度为 100 dpi
plt.rcParams['figure.dpi'] = 100 

# 创建一个包含 2 行 2 列的子图布局，图形大小为 12x8 英寸
fig, axes = plt.subplots(2,2,figsize=(12,8)) 

# 绘制第一个子图的箱线图
i = 0 
feature = features[i] 
# 绘制指定特征的箱线图，dropna() 用于去除缺失值
axes[0,0].boxplot(data[feature].dropna()) 
# 设置第一个子图的标题
axes[0,0].set_title(f'Boxplot of {feature}') 
# 设置第一个子图的 y 轴标签
axes[0,0].set_ylabel(feature) 

# 绘制第二个子图的箱线图
i = 1 
feature = features[i] 
axes[0,1].boxplot(data[feature].dropna()) 
axes[0,1].set_title(f'Boxplot of {feature}') 
axes[0,1].set_ylabel(feature) 

# 绘制第三个子图的箱线图
i = 2 
feature = features[i] 
axes[1,0].boxplot(data[feature].dropna()) 
axes[1,0].set_title(f'Boxplot of {feature}') 
axes[1,0].set_ylabel(feature) 

# 绘制第四个子图的箱线图
i = 3 
feature = features[i] 
axes[1,1].boxplot(data[feature].dropna())      
axes[1,1].set_title(f'Boxplot of {feature}') 
axes[1,1].set_ylabel(feature) 

# 自动调整子图之间的间距，避免标签重叠
plt.tight_layout() 
# 显示绘制好的子图
plt.show() 

# 上述坐标有助于新手观察子图的位置，不用自己去计算

# 循环实现箱线图的绘制
# 再次导入 pandas 库，重复导入可优化
import pandas as pd 
# 再次导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt 
# 再次导入 seaborn 库
import seaborn as sns 
# 定义要绘制箱线图的特征列表
features = ['Annual Income', 'Years in current job', 'Tax Liens', 'Number of Open Accounts'] 

# 再次设置图片的清晰度为 100 dpi
plt.rcParams['figure.dpi'] = 100 

# 创建一个包含 2 行 2 列的子图布局，图形大小为 12x8 英寸
fig, axes = plt.subplots(2,2,figsize=(12,8))     

# 使用循环遍历特征列表，绘制箱线图
for i in range(len(features)): 
    # 计算当前特征对应的子图所在的行索引
    row = i // 2 
    # 计算当前特征对应的子图所在的列索引
    col = i % 2 
    feature = features[i] 
    axes[row,col].boxplot(data[feature].dropna()) 
    axes[row,col].set_title(f'Boxplot of {feature}') 
    axes[row,col].set_ylabel(feature) 

# 自动调整子图之间的间距，避免标签重叠
plt.tight_layout() 
# 显示绘制好的子图
plt.show() 

# 定义要绘制箱线图的特征列表
features = ['Annual Income', 'Years in current job', 'Tax Liens', 'Number of Open Accounts'] 

# 遍历特征列表，打印每个特征的索引和名称
for i, feature in enumerate(features): 
    print(f"索引 {i} 对应的特征是: {feature}") 

# 定义要绘制箱线图的特征列表
features = ['Annual Income', 'Years in current job', 'Tax Liens', 'Number of Open Accounts'] 

# 再次设置图片的清晰度为 100 dpi
plt.rcParams['figure.dpi'] = 100 

# 创建一个包含 2 行 2 列的子图布局，图形大小为 12x8 英寸
fig, axes = plt.subplots(2,2,figsize=(12,8)) 

# 使用循环遍历特征列表，绘制箱线图
for i, feature in enumerate(features): 
    # 计算当前特征对应的子图所在的行索引
    row = i // 2 
    # 计算当前特征对应的子图所在的列索引
    col = i % 2 
    axes[row,col].boxplot(data[feature].dropna()) 
    axes[row,col].set_title(f'Boxplot of {feature}') 
    axes[row,col].set_ylabel(feature) 

# 自动调整子图之间的间距，避免标签重叠
plt.tight_layout() 
# 显示绘制好的子图
plt.show() 