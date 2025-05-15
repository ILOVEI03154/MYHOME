# 初始化技术列表，包含三个元素
tech_list = ["Python", "Java", "Go"]

# 获取列表的第一个元素
first_tech = tech_list[0]

# 在列表末尾添加新元素"JavaScript"
tech_list.append("JavaScript")

# 修改列表的第二个元素为"Ruby"
tech_list[1] = "Ruby"

# 从列表中删除元素"Go"
tech_list.remove("Go")

# 获取当前列表的长度
current_length = len(tech_list)

# 打印第一个技术名称
print(f"第一个技术是: {first_tech}")
# 打印当前列表长度 
print(f"当前列表长度: {current_length}")
# 打印最终列表内容
print(f"最终列表内容: {tech_list}")