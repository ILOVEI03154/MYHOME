# 设置温度值
temperature = 38

# 高温红色预警判断（>35℃）
if temperature > 35:
    print("红色预警:高温天气！")
# 炎热黄色预警判断（28℃-35℃）
elif temperature >=28:
    print("黄色预警：天气炎热")
# 适宜温度绿色提示（20℃-28℃）
elif temperature >=20:
    print("绿色提示：适宜温度")
# 低温蓝色预警处理（<20℃）
else:
    print("蓝色预警：注意保暖")