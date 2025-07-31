from datetime import datetime

# 获取当前时间
now = datetime.now()

# 格式化时间字符串
time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

print(time_string)
