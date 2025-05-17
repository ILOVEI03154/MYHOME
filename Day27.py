# 作业：
# 编写一个装饰器 logger，在函数执行前后打印日志信息（如函数名、参数、返回值）

def logger(func):
    def wrapper(*args, **kwargs):
        print(f"函数名：{func.__name__}")
        print(f"参数：{args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"返回值：{result}")
        return result
    return wrapper

@logger
def add(a, b):
    return a + b

add(2, 3)

add(a = 2, b = 3)

add(2, b = 3)
