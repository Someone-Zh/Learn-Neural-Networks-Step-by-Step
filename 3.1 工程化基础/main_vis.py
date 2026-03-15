# 神经网络训练可视化主程序
# 包含地形生成、网络训练、可视化等功能
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import random
import time
import sys

from tensor import Tensor
from model import SimpleFCN
from optim import Adam, SGD

# 网络训练配置参数
CONFIG = {
    # 地形类型：双峰地形
    "terrain_type": "double_peak",
    # 隐藏层大小
    "hidden_size": 32,
    # 训练点数量
    "train_points": 150,
    # 可视化网格大小
    "vis_grid_size": 25,
    # 总训练轮数
    "epochs_total": 400,
    # 学习率
    "lr": 0.05,
    # 优化器类型
    "optimizer": "sgd",
    # 快照间隔（每隔多少轮保存一次）
    "snapshot_interval": 10,
    # 缓存文件名
    "cache_file": "training_cache.pkl",
    # 是否使用缓存
    "use_cache": False
}

# 数据生成器函数
# 根据指定类型生成地形数据
# 参数：type_ - 地形类型，n_points - 点的数量，is_grid - 是否为网格形式
# 返回：X_list - 输入坐标列表，z_list - 目标值列表，以及网格坐标（如果需要）
def generate_terrain(type_, n_points, is_grid=False):
    # 如果是网格形式，则生成规则的二维网格点
    if is_grid:
        # 计算网格边长
        side = int(math.sqrt(n_points))
        # 在-2.2到2.2范围内生成等间距的x坐标
        x = np.linspace(-2.2, 2.2, side)
        # 在-2.2到2.2范围内生成等间距的y坐标
        y = np.linspace(-2.2, 2.2, side)
        # 创建网格坐标矩阵
        Xm, Ym = np.meshgrid(x, y)
        # 将网格坐标转换为点列表
        X_list = np.column_stack([Xm.ravel(), Ym.ravel()]).tolist()
    else:
        # 如果不是网格形式，则随机生成点
        X_list = [[random.uniform(-2.2, 2.2), random.uniform(-2.2, 2.2)] for _ in range(n_points)]
        # 非网格情况下不返回网格坐标
        Xm, Ym = None, None

    # 初始化目标值列表
    z_list = []
    # 遍历每个输入点并计算对应的z值
    for pt in X_list:
        x, y = pt[0], pt[1]
        # 双峰地形：两个高斯函数的叠加
        if type_ == "double_peak":
            z = math.exp(-(x**2 + y**2)) + 0.6 * math.exp(-((x-1)**2 + (y-1)**2))
        # 鞍形地形：x²-y²的形式
        elif type_ == "saddle":
            z = (x**2 - y**2) / 4.0 + 0.5
        # 波浪地形：sin(x)*cos(y)的形式
        elif type_ == "wave":
            z = math.sin(x) * math.cos(y) + 0.5
        else:
            # 默认情况
            z = 0.0
        # 添加计算出的z值到列表
        z_list.append(z)
    
    # 如果是网格形式，返回网格坐标
    if is_grid:
        return X_list, z_list, Xm, Ym
    else:
        return X_list, z_list

# 执行神经网络训练模拟
# 负责生成训练数据、归一化处理、训练网络并保存历史记录
# 如果启用了缓存且缓存文件存在，则直接加载缓存数据
def run_training_simulation():
    # 检查是否启用缓存且缓存文件存在
    if CONFIG.get("use_cache", False) and os.path.exists(CONFIG.get("cache_file", "training_cache.pkl")):
        print("="*40)
        print("加载缓存的训练数据...")
        print("="*40)
        # 从缓存文件中加载数据
        with open(CONFIG["cache_file"], "rb") as f:
            cache_data = pickle.load(f)
        # 获取网格坐标
        Xm = cache_data["Xm"]
        Ym = cache_data["Ym"]
        # 返回历史数据和网格坐标
        return cache_data["history_z"], cache_data["history_loss"], cache_data["history_epochs"], Xm, Ym
    
    print("="*40)
    print("阶段一：正在高速训练神经网络 (纯计算)...")
    print("="*40)
    
    # 生成训练数据（非网格形式）
    X_train, z_train = generate_terrain(CONFIG["terrain_type"], CONFIG["train_points"], is_grid=False)
    # 生成可视化数据（网格形式）
    X_vis, z_vis_flat, Xm, Ym = generate_terrain(CONFIG["terrain_type"], CONFIG["vis_grid_size"]**2, is_grid=True)
    
    # 将训练数据转换为numpy数组
    X_train_arr = np.array(X_train)
    z_train_arr = np.array(z_train)
    
    # 计算训练数据的均值和标准差用于归一化
    X_mean = X_train_arr.mean(axis=0)
    X_std = X_train_arr.std(axis=0) + 1e-8  # 添加小常数防止除零错误
    z_mean = z_train_arr.mean()
    z_std = z_train_arr.std() + 1e-8
    
    # 对训练数据进行归一化
    X_train_norm = (X_train_arr - X_mean) / X_std
    z_train_norm = (z_train_arr - z_mean) / z_std
    
    # 转换回列表格式
    X_train = X_train_norm.tolist()
    z_train = z_train_norm.tolist()
    
    # 对可视化数据也进行相同的归一化处理
    X_vis_arr = np.array(X_vis)
    X_vis_norm = (X_vis_arr - X_mean) / X_std
    X_vis = X_vis_norm.tolist()
    
    # 输出归一化统计信息
    print(f"数据归一化: X_mean={X_mean}, X_std={X_std}")
    print(f"数据归一化: z_mean={z_mean:.4f}, z_std={z_std:.4f}")
    
    # 创建神经网络实例
    net = SimpleFCN(hidden_size=CONFIG["hidden_size"])
    # 根据配置选择优化器
    OptimClass = Adam if CONFIG["optimizer"] == "adam" else SGD
    
    # 初始化历史记录列表
    history_z = []
    history_loss = []
    history_epochs = []
    
    # 记录开始时间用于ETA计算
    start_time = time.time()
    
    # 开始训练循环
    for epoch in range(1, CONFIG["epochs_total"] + 1):
        # 执行一步训练
        loss_val = net.train_step(X_train, z_train, CONFIG["lr"], OptimClass)
        
        # 按快照间隔保存当前状态
        if epoch % CONFIG["snapshot_interval"] == 0 or epoch == CONFIG["epochs_total"]:
            # 使用训练好的网络预测可视化数据
            X_vis_tensor = Tensor(X_vis)
            z_pred_tensor = net.forward(X_vis_tensor)
            # 提取预测结果
            z_pred_data = [row[0] for row in z_pred_tensor.data]
            # 反归一化预测结果
            Z_pred = np.array(z_pred_data).reshape(Xm.shape) * z_std + z_mean
            
            # 保存当前预测结果、损失和轮次
            history_z.append(Z_pred)
            history_loss.append(loss_val)
            history_epochs.append(epoch)
            
            # 计算并显示进度和预计剩余时间
            elapsed = time.time() - start_time
            progress = epoch / CONFIG["epochs_total"]
            eta = elapsed * (1/progress - 1) if progress > 0 else 0
            sys.stdout.write(f"\rEpoch: {epoch}/{CONFIG['epochs_total']} | Loss: {loss_val:.6f} | ETA: {eta:.1f}s")
            sys.stdout.flush()
            
    print("\n计算完成！正在保存缓存...")
    # 如果启用缓存，保存数据到文件
    if CONFIG.get("use_cache", False):
        cache_data = {
            "history_z": history_z,
            "history_loss": history_loss,
            "history_epochs": history_epochs,
            "Xm": Xm,
            "Ym": Ym,
            "config": CONFIG
        }
        with open(CONFIG.get("cache_file", "training_cache.pkl"), "wb") as f:
            pickle.dump(cache_data, f)
        print(f"缓存已保存到: {CONFIG.get('cache_file', 'training_cache.pkl')}")
    
    print("计算完成！正在启动可视化窗口...")
    return history_z, history_loss, history_epochs, Xm, Ym

# 启动3D可视化动画
# 显示神经网络学习过程的动态变化
# 参数：history_z - 预测值历史记录，history_loss - 损失历史记录，history_epochs - 轮次历史记录，Xm/Ym - 网格坐标
def start_visualization(history_z, history_loss, history_epochs, Xm, Ym):
    # 生成真实地形数据用于对比
    _, z_gt_flat, _, _ = generate_terrain(CONFIG["terrain_type"], CONFIG["vis_grid_size"]**2, is_grid=True)
    # 将真实地形数据重塑为网格形状
    Z_gt = np.array(z_gt_flat).reshape(Xm.shape)

    # 设置深色背景主题
    plt.style.use('dark_background')
    # 创建图形窗口
    fig = plt.figure(figsize=(10, 8))
    # 创建3D子图
    ax3d = fig.add_subplot(111, projection='3d')

    # 绘制真实地形表面（半透明）
    ax3d.plot_surface(Xm, Ym, Z_gt, color='gray', alpha=0.15, linewidth=0, shade=False)
    # 绘制真实地形线框
    ax3d.plot_wireframe(Xm, Ym, Z_gt, color='#ff4444', alpha=0.6, linewidth=1.2)

    # 初始化当前预测表面
    current_surf = [ax3d.plot_surface(Xm, Ym, history_z[0], cmap='viridis', alpha=0.95, edgecolor='none', vmin=-0.5, vmax=1.2)]

    # 设置Z轴范围
    ax3d.set_zlim(-0.5, 1.5)
    # 设置坐标轴标签
    ax3d.set_xlabel('X', labelpad=10, fontsize=12)
    ax3d.set_ylabel('Y', labelpad=10, fontsize=12)
    ax3d.set_zlabel('Z', labelpad=10, fontsize=12)
    # 设置标题
    ax3d.set_title(f"NNF: {CONFIG['terrain_type']}", pad=20, fontsize=14)
    # 设置视角
    ax3d.view_init(elev=25, azim=-60)
    
    # 创建自定义图例
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#ff4444', lw=2),
                    Line2D([0], [0], color='gray', lw=4, alpha=0.5)]
    ax3d.legend(custom_lines, ['Target Surface', 'Predicted Surface'], loc='upper right', frameon=False, fontsize=10)

    # 创建文本信息显示区域
    text_info = ax3d.text2D(0.02, 0.98, "", transform=ax3d.transAxes, fontsize=13, 
                            verticalalignment='top',
                            bbox=dict(facecolor='black', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.4'))

    # 计算总帧数
    total_frames = len(history_z)

    # 定义更新函数
    def update(frame):
        # 获取当前帧的数据
        Z_curr = history_z[frame]
        loss_curr = history_loss[frame]
        epoch_curr = history_epochs[frame]
        
        # 更新显示信息文本
        info_text = f"Epoch: {epoch_curr}\nLoss: {loss_curr:.6f}"

        # 移除当前表面
        current_surf[0].remove()
        # 绘制新的预测表面
        current_surf[0] = ax3d.plot_surface(Xm, Ym, Z_curr, cmap='viridis', alpha=0.95, edgecolor='none', vmin=-0.5, vmax=1.2)
        
        # 更新信息文本
        text_info.set_text(info_text)

        return [current_surf[0], text_info]

    print("启动可视化...")
    # 创建动画对象
    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=200,  # 每帧间隔200毫秒
        blit=False,    # 不使用blitting优化
        cache_frame_data=False  # 不缓存帧数据
    )
    
    # 调整布局并显示
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    h_z, h_loss, h_epochs, xm, ym = run_training_simulation()
    start_visualization(h_z, h_loss, h_epochs, xm, ym)