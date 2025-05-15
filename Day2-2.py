# 定义两个分数变量
score_a = 75  # 学生A的分数
score_b = 90  # 学生B的分数

# 比较运算结果存储
is_a_higher = score_a > score_b            # 判断A是否高于B
is_a_lower_or_equal = score_a <= score_b   # 判断A是否小于等于B
is_different = score_a != score_b         # 判断两分数是否不同

# 输出比较结果
print(f"{score_a}是否于 {score_b}:{is_a_higher}")          # 注意此处"于"应为"大于"，需校对文本
print(f"{score_a}是否小于等于 {score_b}:{is_a_lower_or_equal}")  # 输出小于等于比较结果
print(f"{score_a}是否不等于 {score_b}:{is_different}")      # 输出不等于比较结果