# 定义价格和折扣率
price = 19.9      # 商品原价
discount = 0.8    # 折扣率（8折）

# 计算价格相关数值
final_price = price * discount    # 折后价 = 原价 × 折扣率
saved_mony = price - final_price  # 节省金额 = 原价 - 折后价

# 输出未格式化的金额
print(f"最终价格是：{final_price}\n节省金额是:{saved_mony}")      # 直接输出浮点数值

# 输出格式化的金额（保留两位小数）
print(f"最终价格是：{final_price:.2f}\n节省金额是:{saved_mony:.2f}")  # 规范金额显示格式



