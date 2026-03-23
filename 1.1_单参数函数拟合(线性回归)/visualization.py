"""
线性回归可视化模块
依赖linear_regression_core模块，专注于结果的可视化展示
"""
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import numpy as np
import sys
import os

# 将当前目录加入sys.path，确保能导入同级的core模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from linear_regression_core import generate_data, linear_regression_train

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_training(train_result):
    """
    可视化训练过程：数据分布 + 拟合曲线变化 + 损失下降
    :param train_result: linear_regression_train返回的训练结果字典
    """
    # 提取训练结果并统一维度
    x = train_result["x"]
    y = train_result["y"]
    w_list = train_result["w_list"]
    b_list = train_result["b_list"]
    loss_list = train_result["loss_list"]
    final_w = train_result["final_w"]
    final_b = train_result["final_b"]
    
    # 关键修复：统一帧维度（去掉最后一个参数值，使w_list/b_list和loss_list长度一致）
    w_list = w_list[:-1]
    b_list = b_list[:-1]
    total_frames = len(loss_list)
    
    # 创建2行1列的子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('线性回归拟合 y=2x+3 (手动梯度下降)', fontsize=16)
    
    # -------------------- 子图1：数据分布与拟合曲线动画 --------------------
    # 绘制原始数据点
    ax1.scatter(x, y, color='blue', alpha=0.6, label='带噪声数据点')
    # 绘制真实曲线
    x_true = np.linspace(10, 100, 100)
    y_true = 2 * x_true + 3
    ax1.plot(x_true, y_true, color='red', linestyle='--', linewidth=2, label='真实曲线 y=2x+3')
    
    # 初始化拟合曲线（先绘制空数据）
    fit_line, = ax1.plot([], [], color='green', linewidth=2, label='拟合曲线')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(10, 100)
    ax1.set_ylim(20, 210)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title('数据分布与拟合曲线变化 (Epoch 0)')
    
    # -------------------- 子图2：损失函数下降曲线 --------------------
    # 初始化损失曲线（设置固定x轴范围）
    loss_line, = ax2.plot([], [], color='orange', linewidth=2)
    ax2.set_xlabel('训练轮数 (Epoch)')
    ax2.set_ylabel('MSE损失')
    ax2.set_xlim(0, total_frames)
    ax2.set_ylim(0, max(loss_list) * 1.1)  # 留一点余量
    ax2.grid(True, alpha=0.3)
    ax2.set_title('损失函数下降过程')
    
    # -------------------- 动画更新函数（修复核心逻辑） --------------------
    def update(frame):
        # 安全检查：防止帧索引越界
        if frame >= total_frames:
            frame = total_frames - 1
        
        # 获取当前帧的参数
        current_w = w_list[frame]
        current_b = b_list[frame]
        
        # 更新拟合曲线（确保x和y维度匹配）
        y_fit = current_w * x_true + current_b
        fit_line.set_data(x_true, y_fit)
        
        # 更新损失曲线（逐帧绘制）
        loss_x = np.arange(frame + 1)
        loss_y = loss_list[:frame + 1]
        loss_line.set_data(loss_x, loss_y)
        
        # 更新标题显示当前参数
        ax1.set_title(f'拟合曲线变化 (Epoch {frame}) | w={current_w:.4f} | b={current_b:.4f}')
        
        # 返回需要更新的元素（确保返回的是可迭代对象）
        return [fit_line, loss_line]
    
    # -------------------- 创建动画（优化参数） --------------------
    # 关键优化：
    # 1. interval=100 降低速度（100ms/帧）
    # 2. blit=False 提高兼容性
    # 3. 设置repeat=False 避免循环
    ani = FuncAnimation(
        fig, 
        update, 
        frames=total_frames, 
        interval=100,  # 从20ms改为100ms，慢5倍
        blit=False,    # 关闭blit提高兼容性
        repeat=False,
        cache_frame_data=False  # 禁用帧缓存
    )
    gif_filename = "instance.gif"
    writer = animation.PillowWriter(
        fps=50,  # 帧率，1000/interval=50（和FuncAnimation的interval对应）
        metadata=dict(artist='Me'),  # 可选：添加元数据
        bitrate=1800  # 可选：比特率，数值越高画质越好
    )
    ani.save(gif_filename, writer=writer)

    # 调整布局防止重叠
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 显示动画
    plt.show()
    
    return final_w, final_b

def main():
    """主函数：调用核心模块生成数据、训练模型，然后可视化"""
    # 1. 生成数据
    x, y = generate_data(n_samples=100)
    
    # 2. 训练模型（减少epochs方便观察，也可以用500）
    train_result = linear_regression_train(x, y, lr=0.000005, epochs=200)
    
    # 3. 可视化训练过程
    final_w, final_b = visualize_training(train_result)
    
    # 输出最终拟合公式
    print(f"\n拟合得到的公式：y = {final_w:.4f}x + {final_b:.4f}")

if __name__ == "__main__":
    main()