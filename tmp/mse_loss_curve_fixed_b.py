import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------- 1. 解决中文显示问题 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 2. 生成数据 --------------------------
# 缩小x范围，让梯度值更可控，便于观察围绕最小值的震荡
x = np.linspace(-5, 5, 200)  #
true_w, true_b = 2, 3
y_true = true_w * x + true_b
y_noise = y_true + np.random.normal(0, 0.1, size=x.shape)  # 噪声
n = len(x)

# -------------------------- 3. 核心函数 --------------------------
def mse_loss(w, b, x, y):
    """计算均方误差损失 L = (1/n) * Σ(ŷ_i - y_i)²"""
    y_hat = w * x + b
    loss = np.mean((y_hat - y) ** 2)
    return loss

def calculate_gradient(w, b, x, y):
    """计算梯度：∂L/∂w 和 ∂L/∂b"""
    y_hat = w * x + b
    grad_w = (2/n) * np.sum((y_hat - y) * x)  # 损失对w的梯度（上升方向）
    grad_b = (2/n) * np.sum(y_hat - y)        # 损失对b的梯度（上升方向）
    return grad_w, grad_b

# -------------------------- 4. 生成固定损失曲线 --------------------------
w_range = np.linspace(0, 4, 300)
loss_values = [mse_loss(w, true_b, x, y_noise) for w in w_range]

# -------------------------- 5. 无学习率梯度下降 --------------------------
def gradient_descent_no_lr(initial_w=20, initial_b=2, epochs=30):
    """无学习率更新：w = w - grad_w"""
    w, b = initial_w, initial_b
    history = []
    
    for _ in range(epochs):
        loss = mse_loss(w, b, x, y_noise)
        history.append((w, loss))
        
        grad_w, grad_b = calculate_gradient(w, b, x, y_noise)
        # 计算步长
        a = 0.1
        w = w - a * grad_w  
        b = b - a * grad_b
    
    return np.array(history)

# 执行梯度下降（初始值靠近最优值，便于观察震荡）
history = gradient_descent_no_lr(initial_w=3.7, initial_b=2.5, epochs=30)

# -------------------------- 6. 可视化（固定曲线 + 震荡轨迹） --------------------------
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制固定损失曲线
ax.plot(w_range, loss_values, 'b-', lw=2, label='损失曲线 (固定b=3)')
# 标记最优w值（损失最小值点）
min_loss_idx = np.argmin(loss_values)
optimal_w = w_range[min_loss_idx]
ax.axvline(x=optimal_w, color='green', linestyle='--', linewidth=2, label=f'最优w值: {optimal_w:.2f}')

# 动态元素
oscillation_point, = ax.plot([], [], 'ro', markersize=8, label='当前w值（震荡）')
oscillation_trace, = ax.plot([], [], 'r--', alpha=0.5, label='w迭代轨迹')

# 图表设置
ax.set_xlabel('权重 w (斜率)', fontsize=12)
ax.set_ylabel('损失 L', fontsize=12)
ax.set_title('加入学习率梯度下降的w震荡过程', fontsize=14, pad=20)
ax.set_xlim(0, 4)
ax.set_ylim(0, max(loss_values) * 1.2)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 初始化
def init():
    oscillation_point.set_data([], [])
    oscillation_trace.set_data([], [])
    return oscillation_point, oscillation_trace

# 更新函数
def update(frame):
    current_w = history[frame, 0]
    current_loss = history[frame, 1]
    oscillation_point.set_data([current_w], [current_loss])
    oscillation_trace.set_data(history[:frame+1, 0], history[:frame+1, 1])
    return oscillation_point, oscillation_trace

# 动画设置
total_frames = len(history)
ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=total_frames, 
    interval=500,  # 放慢速度，便于观察
    blit=False,    
    repeat=False,
    cache_frame_data=False
)

# 保存GIF
gif_filename = "mse_loss_curve_fixed_b.gif"
writer = animation.PillowWriter(
    fps=2,  # 匹配500ms间隔
    metadata=dict(artist='Me'),
    bitrate=1800
)
ani.save(gif_filename, writer=writer)

plt.tight_layout()
plt.show()