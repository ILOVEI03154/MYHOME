# import pandas as pd
# import numpy as np
# # data = pd.read_csv('python60-days-challenge\data.csv')#两点表示上一级目录，读取data.csv文件
# #abs_path = r'C:\Users\I.Love.I\Desktop\Python_code\python60-days-challenge\data.csv'#绝对路径读取数据
# #data = pd.read_csv('abs_path')#绝对路径读取数据
# # null_mask = data.isnull()#获取布尔值掩码
# # null_counts = null_mask.sum()#计算每列的缺失值数量
# #print("各列空值数量统计:")
# #print(null_counts)

# #print("\n空值位置布尔矩阵:")
# #print(null_mask)
# #print("\n前15行数据预览：")
# #print(data.head(15))#默认5行结果，如需更多则可加入参数，如data.head(15)

# #
# #print(data2.head(10))
# #print(data2.info())
# #print(data2.shape)
# #print(data2.columns)
# # print(data2.describe)
# # print(data2.dtypes)
# # print(data2.isnull())
# # print(type(data2.isnull()))
# # print(data2.isnull())
# # print(data2.isnull().sum())


# # 1. 读取数据（使用原始字符串避免转义问题）
# data = pd.read_excel(r'python60-days-challenge\data.xlsx')

# # 2. 查看基本信息
# print("=== 数据基本信息 ===")
# print(f"数据尺寸：{data.shape}")
# print(f"\n列名列表:\n{data.columns.tolist()}")
# print(f"\n数据类型:\n{data.dtypes}")

# # 3. 查看空值分布
# print("\n=== 初始空值统计 ===")
# print(data.isnull().sum())

# print(data['Annual Income'])
# print(type(data['Annual Income']))
# median_income = data['Annual Income'].median()
# print(f"中位数: {median_income}")
# data['Annual Income'].fillna(median_income, inplace=True)
# #print(data['Annual Income'].to_string())  # 显示所有行（适合小数据集）
# print(data['Annual Income'])
# print(data['Annual Income'].isnull().sum())

# #使用众数填充缺失值
# mode = data['Annual Income'].mode()
# print(mode)
# #mode = mode[0]
# data['Annual Income'].fillna(mode, inplace=True)
# print(data['Annual Income'].notnull().sum())#统计非空值个数

# print(data.columns)
# print(type(data.columns))
# a = np.array([1,2,3])
# print(a.tolist())
# c = data.columns.tolist()

# print(type(c))

# for i in c:
#     if data[i].dtype !='object':#判断数据类型是否为object
#        if data[i].isnull().sum()>0:
#         mean_value = data[i].mean()
#         data[i].fillna(mean_value, inplace=True)

# print(data.isnull().sum())  

# import pandas as pd
# import numpy as np
# data = pd.read_csv('python60-days-challenge\data.csv')#需要重新读取一遍数据
# print(data.isnull().sum())
# # print(type(data.columns))

# #数据补全方案——中位数补全
# # for col in data.columns:
# #     if data[col].dtype in ['int64', 'float64']:
# #         median_value = data[col].median()
# #         data[col].fillna(median_value, inplace=True)
# #     else:
# #         print(f"列 {col} 没有缺失值。")
# #数据补全方案——众数补全
# for col in data.columns:
#     if data[col].dtype in ['int64', 'float64']:
#         mode_value = data[col].mode()
#         data[col].fillna(mode_value, inplace=True)
#     else:
#         print(f"列 {col} 没有缺失值。")

# print(data.isnull().sum())

# # print(data.notnull().sum())#统计非空值个数
# # print(data.isnull().sum())#统计空值个数
# # # print(data.to_string())
# # # 方式1：设置全局显示行数（推荐）
# # pd.set_option('display.max_rows', 50)  # 设置显示前50行
# # print(data)
import pandas as pd
import numpy as np # 引入 numpy 处理 NaN

data = pd.read_csv('python60-days-challenge\data.csv')#需要重新读取一遍数据

print("处理前缺失值统计:\n", data.isnull().sum())
print("-" * 30)

for col in data.columns:
    # 检查该列是否存在缺失值
    if data[col].isnull().any():
        # 如果是数值类型
        if data[col].dtype in ['int64', 'float64']:
            # 计算众数，并取第一个
            mode_value = data[col].mode()[0]
            data[col].fillna(mode_value, inplace=True)
            print(f"数值列 '{col}' 的缺失值已用众数 ({mode_value}) 填充。")
        # 如果是非数值类型 (可以选择也用众数填充，或其他策略)
        elif data[col].dtype == 'object' or pd.api.types.is_categorical_dtype(data[col]):
            # 同样计算众数并取第一个
            # 注意：如果列中全是 NaN，mode() 可能返回空 Series，需要处理这种情况
            if not data[col].mode().empty:
                mode_value = data[col].mode()[0]
                data[col].fillna(mode_value, inplace=True)
                print(f"非数值列 '{col}' 的缺失值已用众数 ('{mode_value}') 填充。")
            else:
                # 如果全是 NaN，无法计算众数，可以选择填充特定值或跳过
                placeholder = 'Unknown' # 或者 None，或者保持 NaN
                # data[col].fillna(placeholder, inplace=True)
                print(f"非数值列 '{col}' 几乎全是缺失值，无法计算有效众数，未填充或使用占位符。")
        else:
            # 其他类型（如日期时间等）可能需要不同的处理策略
            print(f"列 '{col}' 是 {data[col].dtype} 类型，包含缺失值，但未定义填充策略。")
    else:
        # 只有在确实没有缺失值时才打印这条信息
        print(f"列 '{col}' 没有缺失值。")

print("-" * 30)
print("处理后缺失值统计:\n", data.isnull().sum())
# print("\n处理后的数据:\n", data)
