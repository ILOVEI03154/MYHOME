# 导入 pandas 库，用于数据处理和分析
import pandas as pd
# 从指定路径读取 CSV 文件，并将数据存储在 data 变量中
data = pd.read_csv('python60-days-challenge\data.csv')
# 打印数据的前几行，方便查看数据的基本结构
print(data.head())

# 初始化一个空列表，用于存储连续特征的列名
continues_features = []
# 遍历数据的所有列
for i in data.columns:
    # 判断列的数据类型是否不是对象类型（通常表示非字符串类型）
    if data[i].dtype != 'object':
        # 如果满足条件，则将该列名添加到 continues_features 列表中
        continues_features.append(i)

# 打印存储连续特征列名的列表
print(continues_features)

# 使用 pandas 的 select_dtypes 方法筛选出数据类型为 float64 和 int64 的列，并将列名转换为列表
continues_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
# 再次打印存储连续特征列名的列表
print(continues_features)

# 导入 seaborn 库，用于数据可视化
import seaborn as sns
# 导入 matplotlib.pyplot 库，用于绘制图表
import matplotlib.pyplot as plt
# 再次导入 pandas 库（此处重复导入可省略）
import pandas as pd
# 绘制 'Annual Income' 列的箱线图
sns.boxplot(x=data['Annual Income'])
# 设置图表的标题
plt.title('Annual Income 的箱线图')
# 设置 x 轴的标签
plt.xlabel('Annual Income')
# 显示绘制好的图表
plt.show()

# 再次导入 pandas 库（此处重复导入可省略）
import pandas as pd
# 再次导入 matplotlib.pyplot 库（此处重复导入可省略）
import matplotlib.pyplot as plt
# 再次导入 seaborn 库（此处重复导入可省略）
import seaborn as sns

# 设置 matplotlib 中文字体为 SimHei，用于正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] 
# 设置 matplotlib 正常显示负号
plt.rcParams['axes.unicode_minus'] = False 

# 绘制 'Annual Income' 列的箱线图
sns.boxplot(x=data['Annual Income'])
# 设置图表的标题
plt.title('年收入 箱线图')
# 设置 x 轴的标签
plt.xlabel('年收入')
# 显示绘制好的图表
plt.show()

print(data.columns)

# 绘制 'Years in current job' 列的直方图
sns.histplot(data['Years in current job'])
# 设置图表的标题
plt.title('在当前工作年限 直方图')
# 设置 x 轴的标签
plt.xlabel('在当前工作年限')
# 设置 y 轴的标签
plt.ylabel('员工数量')
# 旋转 x 轴标签 45 度，并使其右对齐，方便阅读
plt.xticks(rotation=45, ha = 'right') 
# 调整图表布局，避免标签重叠
plt.tight_layout() 
# 显示绘制好的图表
plt.show()

# 设置画布大小为 8x6 英寸
plt.figure(figsize=(8,6)) 
# 绘制 'Credit Default' 列和 'Annual Income' 列的箱线图
sns.boxplot(x = 'Credit Default',y = 'Annual Income',data = data) 
# 设置图表的标题
plt.title('Annual Income vs. Credit Default')  
# 设置 x 轴的标签
plt.xlabel('Credit Default')  
# 设置 y 轴的标签
plt.ylabel('Annual Income')  
# 显示绘制好的图表
plt.show()  

# 设置画布大小为 8x6 英寸
plt.figure(figsize=(8,6)) 
# 绘制 'Credit Default' 列和 'Annual Income' 列的小提琴图
sns.violinplot(x='Credit Default',y='Annual Income',data=data) 
# 设置图表的标题
plt.title('Annual Income vs. Credit Default')  
# 设置 x 轴的标签
plt.xlabel('Credit Default')  
# 设置 y 轴的标签
plt.ylabel('Annual Income')  
# 显示绘制好的图表
plt.show()

# 设置画布大小为 8x6 英寸
plt.figure(figsize=(8,6)) 
# 绘制 'Annual Income' 列的直方图，并根据 'Credit Default' 列进行分组，同时显示核密度估计曲线
sns.histplot(x = 'Annual Income',hue = 'Credit Default',data = data, kde=True, element='step') 
# 设置图表的标题
plt.title('Annual Income vs. Credit Default')  
# 设置 x 轴的标签
plt.xlabel('Annual Income')  
# 设置 y 轴的标签
plt.ylabel('Count')  
# 显示绘制好的图表
plt.show()  

# 设置画布大小为 8x6 英寸
plt.figure(figsize=(8,6)) 
# 绘制 'Number of Open Accounts' 列的条形图，并根据 'Credit Default' 列进行分组
sns.countplot(x='Number of Open Accounts', hue='Credit Default', data=data) 
# 设置图表的标题
plt.title('Number of Open Accounts vs. Credit Default')  
# 设置 x 轴的标签
plt.xlabel('Number of Open Accounts')  
# 设置 y 轴的标签
plt.ylabel('Count')  
# 显示绘制好的图表
plt.show()  

# 对 'Number of Open Accounts' 列进行分组，并将分组结果存储在新的 'Open Accounts Group' 列中
data['Open Accounts Group'] = pd.cut(data['Number of Open Accounts'],bins=[0,5,10,15,20,float('inf')],labels=['0-5','5-10','10-15','15-20','20+']) 
# 设置画布大小为 12x8 英寸
plt.figure(figsize=(12,8)) 
# 绘制 'Open Accounts Group' 列的条形图，并根据 'Credit Default' 列进行分组
sns.countplot(x='Open Accounts Group', hue='Credit Default', data=data) 
# 设置图表的标题
plt.title('Number of Open Accounts vs. Credit Default')  
# 设置 x 轴的标签
plt.xlabel('Number of Open Accounts')  
# 设置 y 轴的标签
plt.ylabel('Count')  
# 显示绘制好的图表
plt.show()  