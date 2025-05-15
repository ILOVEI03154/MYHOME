# 函数定义与调用


# 题目1：计算圆的面积
# 任务：编写一个名为 calculate_circle_area 的函数，该函数接收圆的半径 radius 作为参数，并返回圆的面积。圆的面积 = π * radius² (可以使用 math.pi 作为 π 的值)
# ●要求：函数接收一个位置参数 radius。计算半径为5、0、-1时候的面积
# ●注意点：可以采取try-except 使函数变得更加稳健，如果传入的半径为负数，函数应该返回 0 (或者可以考虑引发一个ValueError，但为了简单起见，先返回0)。
import math

def calculate_circle_area(radius):
    try:
        if radius < 0:
            raise ValueError("半径不能为负数")
        return math.pi * radius * radius
    except ValueError as e:
        print(f"错误：{e}")
        return 0

print(calculate_circle_area(5))  # 输出: 78.53981633974483
print(calculate_circle_area(0))  # 输出: 0
print(calculate_circle_area(-1)) # 输出: 0

# 题目2：计算矩形的面积
# ●任务： 编写一个名为 calculate_rectangle_area 的函数，该函数接收矩形的长度 length 和宽度 width 作为参数，并返回矩形的面积。
# ●公式： 矩形面积 = length * width
# ●要求：函数接收两个位置参数 length 和 width。
# ○函数返回计算得到的面积。
# ○如果长度或宽度为负数，函数应该返回 0。

def calculate_rectangle_area(length, width):
    try:
        if length < 0 or width < 0:
            raise ValueError("长度或宽度不能为负数")
        return length * width
    except ValueError as e:
        print(f"错误：{e}")
        return 0

print(calculate_rectangle_area(5, 10))  # 输出: 50
print(calculate_rectangle_area(0, 10))  # 输出: 0
print(calculate_rectangle_area(5, 0))   # 输出: 0
print(calculate_rectangle_area(-1, 10)) # 输出: 0
print(calculate_rectangle_area(5, -1))  # 输出: 0

# 题目3：计算任意数量数字的平均值
# ●任务： 编写一个名为 calculate_average 的函数，该函数可以接收任意数量的数字作为参数（引入可变位置参数 (*args)），并返回它们的平均值。
# ●要求：使用 *args 来接收所有传入的数字。
# ○如果没有任何数字传入，函数应该返回 0。
# ○函数返回计算得到的平均值。

def calculate_average(*args):
    if not args:
        return 0
    return sum(args) / len(args)

print(calculate_average(1, 2, 3, 4, 5))  # 输出: 3.0
print(calculate_average(10, 20, 30))     # 输出: 20.0
print(calculate_average())               # 输出: 0.0
print(calculate_average(-1, -2, -3))     # 输出: -2.0
print(calculate_average(0))              # 输出: 0.0
print(calculate_average(1.5, 2.5, 3.5))  # 输出: 2.5

# 题目4：打印用户信息
# ●任务： 编写一个名为 print_user_info 的函数，该函数接收一个必需的参数 user_id，以及任意数量的额外用户信息（作为关键字参数）。
# ●要求：
# ○user_id 是一个必需的位置参数。
# ○使用 **kwargs 来接收额外的用户信息。
# ○函数打印出用户ID，然后逐行打印所有提供的额外信息（键和值）。
# ○函数不需要返回值

def print_user_info(user_id, **kwargs):
    print(f"用户ID: {user_id}")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_user_info(123, 姓名="张三", 年龄=25, 城市="北京")

# 题目5：格式化几何图形描述
# ●任务： 编写一个名为 describe_shape 的函数，该函数接收图形的名称 shape_name (必需)，一个可选的 color (默认 “black”)，以及任意数量的描述该图形尺寸的关键字参数 (例如 radius=5 对于圆，length=10, width=4 对于矩形)。
# ●要求：shape_name 是必需的位置参数。
# ○color 是一个可选参数，默认值为 “black”。
# ○使用 **kwargs 收集描述尺寸的参数。
# ○函数返回一个描述字符串，格式如下：
# ○“A [color] [shape_name] with dimensions: [dim1_name]=[dim1_value], [dim2_name]=[dim2_value], …”如果 **kwargs 为空，则尺寸部分为 “with no specific dimensions.”

def describe_shape(shape_name, color="black", **kwargs):
    dimensions = ", ".join([f"{key}={value}" for key, value in kwargs.items()])
    if dimensions:
        dimensions = f"with dimensions: {dimensions}"
    else:
        dimensions = "with no specific dimensions."
    return f"A {color} {shape_name} {dimensions}"

print(describe_shape("circle", color="blue", radius=5))  # 输出: A blue circle with dimensions: radius=5
print(describe_shape("rectangle", length=10, width=4))  # 输出: A black rectangle with dimensions: length=10, width=4
print(describe_shape("triangle"))  # 输出: A black triangle with no specific dimensions.
print(describe_shape("square", color="red", side_length=6))  # 输出: A red square with dimensions: side_length=6