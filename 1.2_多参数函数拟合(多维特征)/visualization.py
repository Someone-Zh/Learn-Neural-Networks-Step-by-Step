"""
房价非线性拟合可视化（实时动画版）
核心：实时展示面积→房价倒U型拟合动画 + 损失下降动画
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import os

# 配置中文字体（解决中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 将当前目录加入sys.path，确保能导入同级的core模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from multi_param_regression_core import generate_multi_param_data, multi_param_gradient_descent

def visualize_house_price_animation(train_result):
    """
    实时动画展示训练过程：
    - 上半部分：面积→房价拟合曲线变化
    - 下半部分：损失函数下降过程
    """
    # 提取核心数据
    X = train_result["X"]
    y = train_result["y"]
    w1_list = train_result["w1_list"]
    w2_list = train_result["w2_list"]
    b_list = train_result["b_list"]
    loss_list = train_result["loss_list"]
    
    # 原始特征
    x1 = X[:, 0]  # 面积
    x2 = X[:, 1]  # 楼层
    
    # 过滤NaN损失值并统一参数列表长度
    loss_clean = np.array([l for l in loss_list if not np.isnan(l)])
    total_frames = len(loss_clean)
    
    # 截断参数列表以匹配损失列表长度（防止索引越界）
    w1_list = w1_list[:total_frames]
    w2_list = w2_list[:total_frames]
    b_list = b_list[:total_frames]
    
    # ====================== 创建双图布局 ======================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('房价非线性拟合训练过程（倒U型：面积→房价）', fontsize=16)
    
    # -------------------- 子图1：面积→房价拟合动画 --------------------
    # 绘制静态元素（真实房价散点）
    scatter = ax1.scatter(x1, y, c=x2, cmap='cool', alpha=0.7, s=50,
                          label='真实房价（颜色=楼层，数值越小越贵）')
    # 最优面积线
    ax1.axvline(x=120, color='green', linestyle='--', linewidth=3, label='最优面积(120㎡)')
    # 初始化拟合曲线
    x1_sort = np.linspace(x1.min()-5, x1.max()+5, 200)
    fit_line, = ax1.plot(x1_sort, np.zeros_like(x1_sort), 
                         color='red', linewidth=4, label='拟合曲线（楼层均值）')
    # 轴配置
    ax1.set_xlabel('面积 (㎡)', fontsize=12)
    ax1.set_ylabel('房价 (万元)', fontsize=12)
    ax1.set_xlim(x1.min()-5, x1.max()+5)
    ax1.set_ylim(y.min()-10, y.max()+10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax1, label='楼层')
    # 初始化标题
    ax1.set_title('面积→房价拟合过程（轮数：0）', fontsize=14)
    
    # -------------------- 子图2：损失下降动画 --------------------
    # 初始化损失曲线
    loss_line, = ax2.plot([], [], color='orange', linewidth=3)
    ax2.set_xlabel('训练轮数', fontsize=12)
    ax2.set_ylabel('MSE损失', fontsize=12)
    ax2.set_xlim(0, total_frames)
    ax2.set_ylim(0, np.max(loss_clean) * 1.1)  # 留10%余量
    ax2.grid(True, alpha=0.3)
    ax2.set_title('损失函数下降过程', fontsize=14)
    
    # ====================== 动画更新函数 ======================
    def update(frame):
        # 安全检查：防止帧索引越界
        if frame >= total_frames:
            frame = total_frames - 1
        
        # 获取当前帧的参数
        current_w1 = w1_list[frame]
        current_w2 = w2_list[frame]
        current_b = b_list[frame]
        current_loss = loss_clean[frame]
        
        # 计算拟合曲线（倒U型：面积二次项）
        x1_quad_sort = (x1_sort - 120) ** 2
        y_pred_sort = current_w1 * x1_quad_sort + current_w2 * np.mean(x2) + current_b
        
        # 更新拟合曲线和标题
        fit_line.set_data(x1_sort, y_pred_sort)
        ax1.set_title(
            f'面积→房价拟合过程（轮数：{frame} | w1={current_w1:.4f}, w2={current_w2:.4f}, b={current_b:.2f} | 损失={current_loss:.2f}）'
        )
        
        # 更新损失曲线（逐帧绘制）
        loss_x = np.arange(frame + 1)
        loss_y = loss_clean[:frame + 1]
        loss_line.set_data(loss_x, loss_y)
        
        # 返回需要更新的元素
        return fit_line, loss_line
    
    # ====================== 创建并显示动画 ======================
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=50,  # 50ms/帧（比GIF更快，可根据需要调整）
        repeat=False,
        blit=False,   # 关闭blit提高兼容性
        cache_frame_data=False
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
    
    # 返回最终参数
    final_w1 = w1_list[-1]
    final_w2 = w2_list[-1]
    final_b = b_list[-1]
    return final_w1, final_w2, final_b

def main():
    # 1. 生成非线性房价数据
    print("生成房价数据...")
    X, y = generate_multi_param_data(n_samples=300, noise_std=3.0)
    
    # 2. 训练模型（可调整epochs控制动画时长）
    print("开始拟合...")
    train_result = multi_param_gradient_descent(X, y, lr=0.0000005, epochs=200)
    
    # 3. 实时动画展示训练过程
    print("开始展示动画...")
    final_w1, final_w2, final_b = visualize_house_price_animation(train_result)
    
    # 输出最终拟合公式
    print(f"\n拟合得到的公式：房价 = {final_w1:.6f}*(面积-120)² + {final_w2:.4f}*楼层 + {final_b:.2f}")

if __name__ == "__main__":
    main()