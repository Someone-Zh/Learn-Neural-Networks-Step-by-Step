import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 设置中文字体（解决中文显示乱码和上标问题）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']  # 增加备选字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Microsoft YaHei'
plt.rcParams['mathtext.it'] = 'Microsoft YaHei:italic'

# ===================== 1. 初始化画布与参数 =====================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
axes = [ax1, ax2, ax3]
fig.suptitle('常见函数导数推理动画', fontsize=18, y=0.98)

# 定义3个典型函数及其导数（使用LaTeX语法显示上标，避免字体问题）
functions = [
    {
        'name': '一次函数 $y=2x+3$',
        'der_name': '一次函数导数 $y=2$',
        'original': lambda x: 2*x + 3,    # 原函数
        'derivative': lambda x: np.full_like(x, 2),  
        'x_range': np.linspace(-5, 5, 100),
        'color': 'blue',
        'ax': ax1
    },
    {
        'name': '二次函数 $y=x^2$',
        'der_name': '二次函数导数 $y=2x$',
        'original': lambda x: x**2,       # 原函数
        'derivative': lambda x: 2*x,      # 导数 y’=2x
        'x_range': np.linspace(-5, 5, 100),
        'color': 'red',
        'ax': ax2
    },
    {
        'name': '正弦函数 $y=\\sin(x)$',
        'der_name': '正弦函数 $y=\\cos(x)$',
        'original': lambda x: np.sin(x),  # 原函数
        'derivative': lambda x: np.cos(x),# 导数 y’=cos(x)
        'x_range': np.linspace(-2*np.pi, 2*np.pi, 200),
        'color': 'green',
        'ax': ax3
    }
]

# 初始化绘图元素
plot_elements = []
for func in functions:
    ax = func['ax']
    # 设置坐标轴
    ax.set_xlim(func['x_range'][0], func['x_range'][-1])
    if '一次函数' in func['name']:
        ax.set_ylim(-8, 15)
    elif '二次函数' in func['name']:
        ax.set_ylim(-5, 25)
    else:
        ax.set_ylim(-2, 2)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    
    # 原函数曲线、导数曲线、切线、标注
    orig_line, = ax.plot([], [], color=func['color'], linewidth=2, label=f'原函数 {func["name"]}')
    deriv_line, = ax.plot([], [], color=func['color'], linestyle='--', linewidth=2, label=f'导数 {func["der_name"]}')
    tangent_line, = ax.plot([], [], color='orange', linewidth=1.5, alpha=0)
    annotation = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=9,
                         bbox=dict(boxstyle='round', fc='yellow', alpha=0.7), va='top')
    plot_elements.append({
        'orig_line': orig_line,
        'deriv_line': deriv_line,
        'tangent_line': tangent_line,
        'annotation': annotation,
        'func': func
    })

# 全局图例
handles, labels = [], []
for elem in plot_elements:
    handles.extend([elem['orig_line'], elem['deriv_line']])
    labels.extend([elem['orig_line'].get_label(), elem['deriv_line'].get_label()])
fig.legend(handles[:4], labels[:4], loc='upper right', bbox_to_anchor=(0.98, 0.95))

# ===================== 2. 动画核心更新函数 =====================
def update(frame):
    for idx, elem in enumerate(plot_elements):
        func = elem['func']
        x = func['x_range']
        y_orig = func['original'](x)
        y_deriv = func['derivative'](x)  # 现在返回的是数组
        
        # 分阶段绘制：先画原函数，再画导数，最后标注切线
        total_frames = 300
        phase = frame // (total_frames // 3)
        
        # 阶段1：绘制原函数（0-99帧）
        if phase == 0:
            draw_idx = int(len(x) * frame / (total_frames // 3))
            elem['orig_line'].set_data(x[:draw_idx], y_orig[:draw_idx])
            elem['deriv_line'].set_data([], [])
            elem['tangent_line'].set_alpha(0)
            elem['annotation'].set_text(f'绘制原函数：{func["name"]}')
        
        # 阶段2：绘制导数（100-199帧）
        elif phase == 1:
            elem['orig_line'].set_data(x, y_orig)
            draw_idx = int(len(x) * (frame - total_frames//3) / (total_frames // 3))
            # 安全检查：确保draw_idx在有效范围内
            draw_idx = np.clip(draw_idx, 0, len(x))
            elem['deriv_line'].set_data(x[:draw_idx], y_deriv[:draw_idx])
            elem['tangent_line'].set_alpha(0)
            elem['annotation'].set_text(f'绘制导数：{func["name"].split(" ")[1]}’')
        
        # 阶段3：标注切线（200-299帧）
        else:
            elem['orig_line'].set_data(x, y_orig)
            elem['deriv_line'].set_data(x, y_deriv)
            
            # 选择当前帧对应的x点，绘制切线
            tangent_idx = frame - 2*(total_frames//3)
            tangent_idx = np.clip(tangent_idx, 0, len(x)-1)  # 安全检查
            x0 = x[tangent_idx]
            y0 = y_orig[tangent_idx]
            k = y_deriv[tangent_idx]  # 导数=切线斜率
            
            # 切线方程：y - y0 = k(x - x0)
            x_tangent = np.array([x0-2, x0+2])
            y_tangent = y0 + k*(x_tangent - x0)
            elem['tangent_line'].set_data(x_tangent, y_tangent)
            elem['tangent_line'].set_alpha(1)
            
            # 标注导数的几何意义
            elem['annotation'].set_text(
                f'x={x0:.2f}时\n原函数值：{y0:.2f}\n导数值（切线斜率）：{k:.2f}'
            )
    
    # 收集所有需要更新的元素
    update_elements = []
    for elem in plot_elements:
        update_elements.extend([
            elem['orig_line'], elem['deriv_line'], 
            elem['tangent_line'], elem['annotation']
        ])
    return update_elements

# ===================== 3. 启动动画 =====================
# 创建动画（interval=20表示每20ms更新一帧，共300帧）
ani = animation.FuncAnimation(
    fig, update, frames=300, interval=20, blit=True, repeat=True
)

gif_filename = "derivative.gif"
writer = animation.PillowWriter(
    fps=50,  # 帧率，1000/interval=50（和FuncAnimation的interval对应）
    metadata=dict(artist='Me'),  # 可选：添加元数据
    bitrate=1800  # 可选：比特率，数值越高画质越好
)
ani.save(gif_filename, writer=writer)

# 调整布局并显示
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # 避免标题被遮挡
plt.show()