# 定义原始字符串
str1 = 'Hello'  # 第一个英文字符串
str2 = 'Python'  # 第二个英文字符串

# 字符串拼接的两种方式
greeting = f'{str1} {str2}'  # 方式一：使用f-string格式化（会被下一行覆盖）
greeting = str1 + ' ' + str2  # 方式二：使用加号拼接（最终生效的方式）

# 字符串属性操作
length = len(greeting)         # 获取字符串总长度
first_char = greeting[0]       # 获取第一个字符（索引0）
second_char = greeting[1]      # 获取第二个字符（索引1）
last_char = greeting[-1]       # 获取最后一个字符（负索引）

# 结果输出
print(f"拼接结果:{greeting}")    # 显示完整拼接结果
print(f"字符串长度:{length}")     # 输出字符串总长度
print(f"第一个字符:{first_char}") # 输出首字符
print(f"第二个字符:{second_char}") # 输出第二个字符
print(f"最后一个字符是:{last_char}") # 输出末尾字符