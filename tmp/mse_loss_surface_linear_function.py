import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# -------------------------- 1. 解决中文显示问题 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows：黑体；macOS替换为'PingFang SC'
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# -------------------------- 2. 生成数据 --------------------------
# 生成x数据
x = np.linspace(-5, 5, 100)
# 真实函数 y=2x+3，加入少量噪声模拟实际场景
true_k, true_b = 2, 3
y_true = true_k * x + true_b
y_noise = y_true + np.random.normal(0, 0.5, size=x.shape)  # 加入高斯噪声

# -------------------------- 3. 定义损失函数 --------------------------
def mse_loss(k, b, x, y):
    """计算均方误差损失 L = (1/n) * Σ(ŷ_i - y_i)²"""
    y_hat = k * x + b  # 预测值
    loss = np.mean((y_hat - y) ** 2)  # 均方误差
    return loss

# -------------------------- 4. 生成损失曲面数据 --------------------------
# 生成k和b的网格（围绕真实值2,3展开）
k_range = np.linspace(0, 4, 100)   # 斜率k的范围：0到4
b_range = np.linspace(1, 5, 100)   # 截距b的范围：1到5
K, B = np.meshgrid(k_range, b_range)

# 计算每个(k,b)组合对应的损失值
L = np.zeros_like(K)
for i in range(len(k_range)):
    for j in range(len(b_range)):
        L[j, i] = mse_loss(K[j, i], B[j, i], x, y_noise)

# -------------------------- 5. 绘制损失曲面并制作GIF --------------------------
# 创建画布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D损失曲面
surf = ax.plot_surface(K, B, L, cmap='viridis', alpha=0.8, edgecolor='none')

# 标记真实参数对应的损失点（k=2, b=3）
true_loss = mse_loss(true_k, true_b, x, y_noise)
ax.scatter(true_k, true_b, true_loss, color='red', s=100, 
           label=f'真实参数 (k={true_k}, b={true_b})', zorder=5)

# 设置坐标轴标签和标题
ax.set_xlabel('斜率 k', fontsize=12)
ax.set_ylabel('截距 b', fontsize=12)
ax.set_zlabel('损失 L', fontsize=12)
ax.set_title('y=2x+3 的均方误差损失曲面', fontsize=14, pad=20)

# 添加颜色条（表示损失值大小）
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label='损失值 L')

# 显示图例
ax.legend(loc='upper right')

# 定义动画更新函数（让视角旋转）
total_frames = 180  # 总帧数（旋转一周约180帧）
def update(frame):
    """更新每一帧的视角"""
    ax.view_init(elev=20, azim=45 + frame * 2)  # azim角度随帧递增，实现旋转
    return surf,

# 创建动画（使用你指定的参数）
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=total_frames, 
    interval=100,  # 每帧间隔100ms（比原20ms慢5倍）
    blit=False,    # 关闭blit提高兼容性
    repeat=False,
    cache_frame_data=False  # 禁用帧缓存
)

# 保存为GIF（使用你指定的参数）
gif_filename = "mse_loss_surface_linear_function.gif"
writer = animation.PillowWriter(
    fps=50,  # 帧率，1000/interval=10（注意：1000/100=10，这里修正为10更匹配）
    metadata=dict(artist='Me'),  # 可选：添加元数据
    bitrate=1800  # 可选：比特率，数值越高画质越好
)
ani.save(gif_filename, writer=writer)
print(f"GIF已保存为：{gif_filename}")

# 优化布局并显示图像（可选）
plt.tight_layout()
plt.show()