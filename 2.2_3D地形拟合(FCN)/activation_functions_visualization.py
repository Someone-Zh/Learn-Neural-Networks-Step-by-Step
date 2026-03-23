"""
神经网络激活函数可视化动画
修复了颜色参数错误和索引越界问题
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec

# -------------------------- 1. 解决中文显示问题 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 2. 定义所有激活函数 --------------------------
def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh激活函数"""
    return np.tanh(x)

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU激活函数"""
    return np.where(x > 0, x, alpha * x)

def prelu(x, alpha=0.25):
    """Parametric ReLU激活函数"""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """ELU激活函数"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x, lambda_=1.0507, alpha=1.67326):
    """SELU激活函数"""
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softplus(x):
    """Softplus激活函数"""
    return np.log(1 + np.exp(x))

def swish(x, beta=1.0):
    """Swish激活函数"""
    return x * sigmoid(beta * x)

def mish(x):
    """Mish激活函数"""
    return x * np.tanh(np.log(1 + np.exp(x)))

def gelu(x):
    """GELU激活函数"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# -------------------------- 3. 激活函数配置 --------------------------
activation_functions = [
    {
        'name': 'Sigmoid',
        'func': sigmoid,
        'color': '#FF6B6B',  # 红色
        'description': '输出范围(0,1)，用于二分类输出层'
    },
    {
        'name': 'Tanh',
        'func': tanh,
        'color': '#4ECDC4',  # 青色
        'description': '输出范围(-1,1)，零中心化，用于隐藏层'
    },
    {
        'name': 'ReLU',
        'func': relu,
        'color': '#45B7D1',  # 蓝色
        'description': 'f(x)=max(0,x)，计算简单，缓解梯度消失'
    },
    {
        'name': 'Leaky ReLU',
        'func': lambda x: leaky_relu(x, 0.01),
        'color': '#96CEB4',  # 绿色
        'description': 'f(x)=max(αx,x)，解决神经元死亡问题'
    },
    {
        'name': 'ELU',
        'func': lambda x: elu(x, 1.0),
        'color': '#FFEAA7',  # 黄色
        'description': '平滑负区域，均值接近零'
    },
    {
        'name': 'Swish',
        'func': lambda x: swish(x, 1.0),
        'color': '#DDA0DD',  # 紫色
        'description': 'x·σ(βx)，Google提出的自门控激活函数'
    },
    {
        'name': 'Mish',
        'func': mish,
        'color': '#FFA07A',  # 橙色
        'description': 'x·tanh(ln(1+e^x))，平滑且无上界'
    },
    {
        'name': 'GELU',
        'func': gelu,
        'color': '#98D8C8',  # 浅绿色
        'description': '0.5x·(1+tanh(√(2/π)(x+0.044715x³)))，用于Transformer'
    }
]

# -------------------------- 4. 创建图形和坐标轴 --------------------------
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# 主图：激活函数曲线
ax_main = fig.add_subplot(gs[0:2, 0:2])
ax_main.set_xlim(-5, 5)
ax_main.set_ylim(-2, 2)
ax_main.set_xlabel('输入 x', fontsize=12)
ax_main.set_ylabel('输出 f(x)', fontsize=12)
ax_main.set_title('神经网络激活函数可视化', fontsize=16, fontweight='bold')
ax_main.grid(True, alpha=0.3)
ax_main.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
ax_main.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)

# 导数图
ax_derivative = fig.add_subplot(gs[0:2, 2])
ax_derivative.set_xlim(-5, 5)
ax_derivative.set_ylim(-0.5, 1.5)
ax_derivative.set_xlabel('输入 x', fontsize=12)
ax_derivative.set_ylabel("导数 f'(x)", fontsize=12)
ax_derivative.set_title('激活函数导数', fontsize=14)
ax_derivative.grid(True, alpha=0.3)

# 信息面板
ax_info = fig.add_subplot(gs[2, :])
ax_info.axis('off')
info_text = ax_info.text(0.02, 0.5, '', fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# -------------------------- 5. 初始化数据 --------------------------
x = np.linspace(-5, 5, 1000)
current_func_idx = 0
history = []  # 存储历史函数数据

# 计算所有函数的y值
for func_info in activation_functions:
    y = func_info['func'](x)
    history.append({
        'x': x.copy(),
        'y': y.copy(),
        'func_info': func_info
    })

# -------------------------- 6. 动画更新函数 --------------------------
def update(frame):
    """更新动画帧"""
    global current_func_idx
    
    # 清除之前的图形
    ax_main.clear()
    ax_derivative.clear()
    
    # 设置主图
    ax_main.set_xlim(-5, 5)
    ax_main.set_ylim(-2, 2)
    ax_main.set_xlabel('输入 x', fontsize=12)
    ax_main.set_ylabel('输出 f(x)', fontsize=12)
    ax_main.set_title(f'激活函数: {activation_functions[frame]["name"]}', 
                     fontsize=16, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    ax_main.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
    
    # 绘制当前激活函数
    func_info = activation_functions[frame]
    y = func_info['func'](x)
    
    # 使用正确的颜色参数（英文颜色名或十六进制）
    ax_main.plot(x, y, color=func_info['color'], linewidth=3, 
                label=func_info['name'])
    ax_main.legend(loc='upper left')
    
    # 绘制导数
    dy = np.gradient(y, x)
    ax_derivative.plot(x, dy, color=func_info['color'], linewidth=2, 
                      linestyle='--', label="导数")
    ax_derivative.set_xlim(-5, 5)
    ax_derivative.set_ylim(-0.5, 1.5)
    ax_derivative.set_xlabel('输入 x', fontsize=12)
    ax_derivative.set_ylabel("导数 f'(x)", fontsize=12)
    ax_derivative.set_title('函数导数', fontsize=14)
    ax_derivative.grid(True, alpha=0.3)
    ax_derivative.legend()
    
    # 更新信息面板
    info_text.set_text(
        f"函数名称: {func_info['name']}\n\n"
        f"函数描述: {func_info['description']}\n\n"
        f"关键特性:\n"
        f"• 输出范围: [{y.min():.2f}, {y.max():.2f}]\n"
        f"• 在x=0处的值: {func_info['func'](0):.3f}\n"
        f"• 在x=0处的导数: {np.gradient(y, x)[500]:.3f}\n\n"
        f"应用场景:\n"
        f"{get_application_scene(func_info['name'])}"
    )
    
    current_func_idx = frame
    return [ax_main, ax_derivative, info_text]

def get_application_scene(func_name):
    """获取应用场景描述"""
    scenes = {
        'Sigmoid': '二分类输出层、概率输出、门控机制',
        'Tanh': 'RNN/LSTM隐藏层、需要零中心化输出的场景',
        'ReLU': 'CNN隐藏层、全连接网络、大多数现代神经网络',
        'Leaky ReLU': '解决ReLU死亡神经元问题、GANs',
        'ELU': '需要平滑负区域的网络、自编码器',
        'Swish': 'Google提出的替代ReLU、MobileNetV3',
        'Mish': '目标检测(YOLOv4)、需要平滑激活的场景',
        'GELU': 'Transformer(BERT/GPT)、预训练模型'
    }
    return scenes.get(func_name, '广泛用于深度学习模型')

# -------------------------- 7. 创建并保存动画 --------------------------
try:
    # 动画设置
    total_frames = len(activation_functions)
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=total_frames, 
        interval=1500,  # 每1.5秒切换一个函数
        blit=False,    
        repeat=True,
        cache_frame_data=False
    )
    
    # 保存GIF
    gif_filename = "activation_functions_visualization.gif"
    print(f"正在生成动画，共{total_frames}个激活函数...")
    
    # 使用Pillow写入器保存GIF
    writer = animation.PillowWriter(fps=1, bitrate=1800)
    ani.save(gif_filename, writer=writer)
    
    print(f"✓ 动画已保存为: {gif_filename}")
    print(f"✓ 文件大小: {os.path.getsize(gif_filename) / 1024:.1f} KB")
    
    # 显示动画
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"错误: {e}")
    
# -------------------------- 8. 可选：生成静态对比图 --------------------------
def create_static_comparison():
    """创建静态对比图"""
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, func_info in enumerate(activation_functions[:8]):
        ax = axes[idx]
        y = func_info['func'](x)
        
        ax.plot(x, y, color=func_info['color'], linewidth=2)
        ax.set_title(func_info['name'], fontsize=12, fontweight='bold')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        
        # 添加关键点标注
        ax.text(0.05, 0.95, f'f(0)={func_info["func"](0):.2f}', 
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle('神经网络激活函数对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('activation_functions_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ 静态对比图已保存为: activation_functions_comparison.png")
    plt.show()

# 运行静态对比图生成
create_static_comparison()

print("\n" + "="*50)
print("激活函数可视化完成！")
print("="*50)
