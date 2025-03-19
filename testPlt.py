import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 生成数据点
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制图形
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')

# 添加标题和标签
plt.title('正弦函数图像', fontsize=14, pad=15)
plt.xlabel('x', fontsize=12)
plt.ylabel('sin(x)', fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=10)

# 优化坐标轴
plt.axis([0, 6, -1.2, 1.2])

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()


