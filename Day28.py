# 作业
# 题目1：定义圆（Circle）类
# 要求：
# 1.包含属性：半径 radius。
# 2.包含方法：
# ●calculate_area()：计算圆的面积（公式：πr²）。
# ●calculate_circumference()：计算圆的周长（公式：2πr）。
# 3.初始化时需传入半径，默认值为 1。

class Circle:
    def __init__(self, radius=1):
        self.radius = radius

    def calculate_area(self):
        return 3.14 * self.radius ** 2

    def calculate_circumference(self):
        return 2 * 3.14 * self.radius

# 创建一个圆对象
circle = Circle(5)  # 半径为 5 的圆
print(f"半径为 {circle.radius} 的圆的面积为 {circle.calculate_area()}")
print(f"半径为 {circle.radius} 的圆的周长为 {circle.calculate_circumference()}")

# 题目2：定义长方形（Rectangle）类
# 1.包含属性：长 length、宽 width。
# 2.包含方法：
# ●calculate_area()：计算面积（公式：长×宽）。
# ●calculate_perimeter()：计算周长（公式：2×(长+宽)）。 is_square() 方法，判断是否为正方形（长 == 宽）。
# 3.初始化时需传入长和宽，默认值均为 1。
class Rectangle:
    def __init__(self, length=1, width=1):
        self.length = length
        self.width = width

    def calculate_area(self):
        return self.length * self.width

    def calculate_perimeter(self):
        return 2 * (self.length + self.width)

    def is_square(self):
        return self.length == self.width

# 创建一个长方形对象
rectangle = Rectangle(4, 6)  # 长为 4，宽为 6 的长方形
print(f"长为 {rectangle.length}，宽为 {rectangle.width} 的长方形的面积为 {rectangle.calculate_area()}")
print(f"长为 {rectangle.length}，宽为 {rectangle.width} 的长方形的周长为 {rectangle.calculate_perimeter()}")
print(f"长为 {rectangle.length}，宽为 {rectangle.width} 的长方形是否为正方形：{rectangle.is_square()}")
# 创建一个正方形对象
square = Rectangle(5, 5)  # 长和宽均为 5 的正方形
print(f"长为 {square.length}，宽为 {square.width} 的正方形的面积为 {square.calculate_area()}")
print(f"长为 {square.length}，宽为 {square.width} 的正方形的周长为 {square.calculate_perimeter()}")
print(f"长为 {square.length}，宽为 {square.width} 的正方形是否为正方形：{square.is_square()}")

# 题目3：图形工厂
# 创建一个工厂函数 create_shape(shape_type, *args)，根据类型创建不同图形对象：图形工厂（函数或类）
# shape_type="circle"：创建圆（参数：半径）。
# shape_type="rectangle"：创建长方形（参数：长、宽）。

def create_shape(shape_type, *args):
    if shape_type == "circle":
        return Circle(*args)  # 使用圆类的构造函数创建圆对象
    elif shape_type == "rectangle":
        return Rectangle(*args)  # 使用长方形类的构造函数创建长方形对象
    else:
        raise ValueError("无效的图形类型")

# 创建一个圆对象    
circle = create_shape("circle", 5)  # 半径为 5 的圆
print(f"半径为 {circle.radius} 的圆的面积为 {circle.calculate_area()}")
print(f"半径为 {circle.radius} 的圆的周长为 {circle.calculate_circumference()}")

# 创建一个长方形对象
rectangle = create_shape("rectangle", 4, 6)  # 长为 4，宽为 6 的长方形
print(f"长为 {rectangle.length}，宽为 {rectangle.width} 的长方形的面积为 {rectangle.calculate_area()}") 
print(f"长为 {rectangle.length}，宽为 {rectangle.width} 的长方形的周长为 {rectangle.calculate_perimeter()}")
print(f"长为 {rectangle.length}，宽为 {rectangle.width} 的长方形是否为正方形：{rectangle.is_square()}")
# 创建一个正方形对象
square = create_shape("rectangle", 5, 5)  # 长和宽均为 5 的正方形
print(f"长为 {square.length}，宽为 {square.width} 的正方形的面积为 {square.calculate_area()}")
print(f"长为 {square.length}，宽为 {square.width} 的正方形的周长为 {square.calculate_perimeter()}")
print(f"长为 {square.length}，宽为 {square.width} 的正方形是否为正方形：{square.is_square()}")


