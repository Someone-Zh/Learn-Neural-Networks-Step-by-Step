import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from fcn_3d_terrain_core import generate_terrain, SimpleFCN

def main():
    # ========== 可配置参数（改这里切换地形/调整训练） ==========
    terrain_type = "saddle"  # 可选：double_peak/mexican_hat/saddle
    hidden_size = 64
    train_points = 2000  # 训练用随机点数量
    vis_grid_size = 40   # 可视化用网格单边点数（40x40）
    epochs = 3000
    update_interval = 50
    lr = 0.01

    # ========== 生成训练数据（参数化） ==========
    X_train, z_train = generate_terrain(
        terrain_type=terrain_type,
        n_points=train_points,
        is_grid=False
    )
    net = SimpleFCN(hidden_size)

    # ========== 生成可视化用数据（参数化，和训练同源） ==========
    X_vis, z_gt_flat, Xm, Ym = generate_terrain(
        terrain_type=terrain_type,
        n_points=vis_grid_size,
        is_grid=True
    )
    Zgt = z_gt_flat.reshape(Xm.shape)  # 真实地形（参数化生成）

    # ========== 绘图初始化 ==========
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制真实地形（参数化生成，不再硬编码）
    ax.plot_surface(Xm, Ym, Zgt, color='gray', alpha=0.3, linewidth=0)
    # 初始化预测地形
    surf = ax.plot_surface(Xm, Ym, np.zeros_like(Zgt), cmap='viridis', alpha=0.8)

    ax.set_zlim(0, 1)
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (height)')
    ax.view_init(25, 60)

    # ========== 动画更新 ==========
    def update(frame):
        nonlocal surf
        # 训练update_interval轮
        for _ in range(update_interval):
            net.train_step(X_train, z_train, lr)

        # 预测并更新曲面
        Zpred = net.forward(X_vis).reshape(Xm.shape)
        # 裁剪预测值到[0,1]，避免超出显示范围
        Zpred = np.clip(Zpred, 0, 1)
        surf.remove()
        surf = ax.plot_surface(Xm, Ym, Zpred, cmap='viridis', alpha=0.8)
        # 更新标题（显示当前Loss和地形类型）
        loss = net.loss_history[-1]
        ax.set_title(f"Terrain: {terrain_type} | Loss: {loss:.4f}")
        return [surf]

    # 创建动画
    ani = animation.FuncAnimation(
        fig, update,
        frames=epochs//update_interval,
        interval=30,  # 30ms/帧，流畅不卡
        blit=False
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()