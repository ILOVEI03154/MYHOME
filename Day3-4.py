# 初始化分数列表
scores = [85, 92, 78 , 65, 95, 88]

# 初始化计数器（优秀分数个数）和累加器（总分）
excellent_count = 0
total_score = 0

# 遍历所有分数
for score in scores:
    total_score += score  # 累加总分
    if score >= 90:      # 统计优秀分数（90分及以上）
        excellent_count += 1

# 计算平均分（保留3位小数）
average_score = total_score / len(scores)

# 输出统计结果
print(f"优秀分数个数: {excellent_count}")  # 优秀成绩数量
print(f"分数总和: {total_score}")         # 总分展示
print(f"平均分数: {average_score:.3f}")   # 格式化平均分
