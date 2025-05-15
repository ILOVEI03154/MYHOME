# 定义一个包含整数的元组
my_tuple1 = (1, 2, 3)
# 定义一个包含字符串的元组
my_tuple2 = ('a', 'b', 'c')
# 定义一个包含不同类型元素的元组，包括整数、字符串、浮点数和列表
my_tuple3 = (1, 'hello', 3.14, [4, 5]) 
# 打印元组 1
print(my_tuple1)
# 打印元组 2
print(my_tuple2)
# 打印元组 3
print(my_tuple3)

# 可以省略括号来定义元组，逗号是关键
my_tuple4 = 10, 20, 'thirty' 
# 打印元组 4
print(my_tuple4)
# 查看元组 4 的数据类型
print(type(my_tuple4)) 

# 创建空元组的第一种方式
empty_tuple = ()
# 创建空元组的第二种方式，使用 tuple() 函数
empty_tuple2 = tuple()
# 打印空元组 1
print(empty_tuple)
# 打印空元组 2
print(empty_tuple2)

# 定义一个包含字符串的元组
my_tuple = ('P', 'y', 't', 'h', 'o', 'n')
# 打印元组的第一个元素，索引从 0 开始
print(my_tuple[0])  
# 打印元组的第三个元素
print(my_tuple[2])  
# 打印元组的最后一个元素，负数索引表示从后往前数
print(my_tuple[-1]) 

# 定义一个包含整数的元组
my_tuple = (0, 1, 2, 3, 4, 5)
# 切片操作，获取索引从 1 到 3 的元素（不包括索引 4 的元素）
print(my_tuple[1:4])  
# 切片操作，获取从开头到索引 2 的元素
print(my_tuple[:3])   
# 切片操作，获取从索引 3 到结尾的元素
print(my_tuple[3:])   
# 切片操作，每隔一个元素取一个
print(my_tuple[::2])  

# 定义一个包含整数的元组
my_tuple = (1, 2, 3)
# 获取元组的长度
print(len(my_tuple))

# 从 sklearn 库中导入加载鸢尾花数据集的函数
from sklearn.datasets import load_iris
# 从 sklearn 库中导入划分训练集和测试集的函数
from sklearn.model_selection import train_test_split
# 从 sklearn 库中导入数据标准化的类
from sklearn.preprocessing import StandardScaler
# 从 sklearn 库中导入逻辑回归分类器的类
from sklearn.linear_model import LogisticRegression
# 从 sklearn 库中导入管道类，用于构建机器学习工作流
from sklearn.pipeline import Pipeline
# 从 sklearn 库中导入计算准确率的函数
from sklearn.metrics import accuracy_score

# 1. 加载鸢尾花数据集
iris = load_iris()
# 获取数据集的特征数据
X = iris.data
# 获取数据集的标签数据
y = iris.target

# 2. 划分训练集和测试集，测试集占比 30%，随机种子为 42，保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 构建管道
# 管道按顺序执行以下步骤：
#    - StandardScaler(): 标准化数据（移除均值并缩放到单位方差）
#    - LogisticRegression(): 逻辑回归分类器
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression())
])

# 4. 使用训练集数据训练模型
pipeline.fit(X_train, y_train)

# 5. 使用训练好的模型对测试集数据进行预测
y_pred = pipeline.predict(X_test)

# 6. 评估模型，计算模型在测试集上的准确率
accuracy = accuracy_score(y_test, y_pred)
# 打印模型在测试集上的准确率，保留两位小数
print(f"模型在测试集上的准确率: {accuracy:.2f}")

# 定义一个列表
my_list = [1, 2, 3, 4, 5]
print("迭代列表:")
# 遍历列表中的每个元素并打印
for item in my_list:
    print(item)

# 定义一个元组
my_tuple = ('a', 'b', 'c')
print("迭代元组:")
# 遍历元组中的每个元素并打印
for item in my_tuple:
    print(item)

# 定义一个字符串
my_string = "hello"
print("迭代字符串:")
# 遍历字符串中的每个字符并打印
for char in my_string:
    print(char)

print("迭代 range:")
# 遍历 range 生成的 0 到 4 的整数并打印
for number in range(5):  
    print(number)

# 定义一个集合，集合中的元素是唯一的，无序的
my_set = {3, 1, 4, 1, 5, 9}
print("迭代集合:")
# 遍历集合中的每个元素并打印
for item in my_set:
    print(item)

# 定义一个字典
my_dict = {'name': 'Alice', 'age': 30, 'city': 'Singapore'}
print("迭代字典 (默认迭代键):")
# 遍历字典的键并打印
for key in my_dict:
    print(key)

print("迭代字典的值:")
# 遍历字典的值并打印
for value in my_dict.values():
    print(value)

print("迭代字典的键值对:")
# 遍历字典的键值对并打印
for key, value in my_dict.items(): 
    print(f"Key: {key}, Value: {value}")

# 导入 os 模块，用于与操作系统进行交互
import os
# 获取当前工作目录的绝对路径并打印
print(os.getcwd()) 
# 获取当前工作目录下的文件列表并打印
print(os.listdir()) 

# 定义一个路径，使用原始字符串避免转义问题
path_a = r'C:\Users\YourUsername\Documents' 
# 定义一个子目录名
path_b = 'MyProjectData'
# 定义一个文件名
file = 'results.csv'

# 使用 os.path.join 将路径、子目录名和文件名安全地拼接起来
file_path = os.path.join(path_a , path_b, file)
# 打印拼接后的文件路径
print(file_path)

# 打印当前环境变量的字典
print(os.environ)

# 遍历环境变量的键值对并打印
for variable_name, value in os.environ.items():
  # 直接打印出变量名和对应的值
  print(f"{variable_name}={value}")

# 打印总共检测到的环境变量数量
print(f"\n--- 总共检测到 {len(os.environ)} 个环境变量 ---")

# 获取当前工作目录
start_directory = os.getcwd() 
print(f"--- 开始遍历目录: {start_directory} ---")

# 遍历指定目录及其子目录
for dirpath, dirnames, filenames in os.walk(start_directory):
    # 打印当前访问的目录路径
    print(f"  当前访问目录 (dirpath): {dirpath}")
    # 打印当前目录下的子目录列表
    print(f"  子目录列表 (dirnames): {dirnames}")
    # 打印当前目录下的文件列表
    print(f"  文件列表 (filenames): {filenames}")

    # # 你可以在这里对文件进行操作，比如打印完整路径
    # print("    文件完整路径:")
    # for filename in filenames:
    #     full_path = os.path.join(dirpath, filename)
    #     print(f"      - {full_path}")
