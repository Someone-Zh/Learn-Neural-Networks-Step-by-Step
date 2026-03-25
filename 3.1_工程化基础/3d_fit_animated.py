import math
import random
import time
import pickle
import os
from tensor import Tensor
from model import Layer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# -------------------------- 1. 解决中文显示问题 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def mexican_hat(x, y):
    """
    墨西哥帽函数 (Ricker Wavelet 2D)
    公式: z = (1 - r^2) * exp(-r^2 / 2)
    特征: 中间一个高峰，周围一圈凹陷，远处趋于0
    """
    r2 = x*x + y*y
    return (1.0 - r2) * math.exp(-r2 / 2.0)

def double_peak(x, y):
    """
    双峰函数
    公式: 两个高斯分布的叠加
    特征: 两个明显的波峰
    """
    # 峰1: 中心 (0.5, 0.5), 较窄
    g1 = math.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.1)
    # 峰2: 中心 (-0.5, -0.5), 稍宽且矮
    g2 = 0.8 * math.exp(-((x+0.5)**2 + (y+0.5)**2) / 0.15)
    return g1 + g2

def generate_data(target_func, n_samples=400, range_val=3.0, seed=42):
    """
    生成用于回归训练的随机采样数据
    
    参数:
        target_func: 目标数学函数 (如 mexican_hat)
        n_samples: 采样点数量
        range_val: x, y 的取值范围 [-range_val, range_val]
        seed: 随机种子，保证结果可复现
        
    返回:
        X_list: 列表，元素为 [x, y]
        Y_list: 列表，元素为 [z]
    """
    random.seed(seed)
    X_list = []
    Y_list = []
    
    for _ in range(n_samples):
        # 在指定范围内随机生成 x, y
        x = random.uniform(-range_val, range_val)
        y = random.uniform(-range_val, range_val)
        # 计算真实值 z
        z = target_func(x, y)
        
        X_list.append([x, y])
        Y_list.append([z])
        
    return X_list, Y_list

class RegressionNet:
    """
    一个简单的全连接回归网络
    结构: 输入层 -> 隐藏层 (ReLU) -> ... -> 输出层 (线性)
    """
    def __init__(self, input_dim=2, hidden_dims=[40, 40], output_dim=1):
        self.layers = []
        # 构建网络层维度列表，例如 [2, 40, 40, 1]
        dims = [input_dim] + hidden_dims + [output_dim]
        
        # 依次创建全连接层
        for i in range(len(dims) - 1):
            nin, nout = dims[i], dims[i+1]
            self.layers.append(Layer(nin, nout))
            
        # 收集所有参数 (权重 W 和 偏置 b) 以便优化器更新
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def forward(self, x_tensor):
        """
        前向传播：计算预测值
        """
        out = x_tensor
        for i, layer in enumerate(self.layers):
            out = layer(out)
            # 除了最后一层（输出层），其他层都加 ReLU 激活函数
            # 输出层不加激活，因为我们要拟合任意范围的连续值
            if i < len(self.layers) - 1:
                out = out.relu()
        return out

    def train_step(self, x_data, y_true_val, lr):
        """
        单个样本的训练步骤：前向 -> 计算损失 -> 反向传播 -> 参数更新
        
        参数:
            x_data: 输入列表 [x, y]
            y_true_val: 真实值列表 [z]
            lr: 学习率
            
        返回:
            loss_value: 当前样本的损失值 (标量)
        """
        # 1. 将原始数据转换为 Tensor 对象
        # 形状: (1, 2) 表示 1 个样本，2 个特征
        x = Tensor([x_data])       
        # 形状: (1, 1) 表示 1 个样本，1 个目标值
        y_true = Tensor([y_true_val]) 
        
        # 2. 前向传播
        y_pred = self.forward(x)
        
        # 3. 计算损失 (均方误差 MSE)
        # Loss = (pred - true)^2
        diff = y_pred - y_true
        loss = (diff * diff).sum() # 对批次内所有元素求和
        
        # 4. 清零梯度
        # 在反向传播前，必须将所有参数的梯度归零，否则梯度会累加
        for p in self.params:
            p.zero_grad()
            
        # 5. 反向传播
        # 自动计算损失对所有参数的梯度
        loss.backward()
        
        # 6. 参数更新 (随机梯度下降 SGD)
        # 公式: w = w - lr * dw
        for p in self.params:
            # 处理权重矩阵 (2D 列表)
            if isinstance(p.data, list) and len(p.data) > 0 and isinstance(p.data[0], list):
                for r in range(len(p.data)):
                    for c in range(len(p.data[0])):
                        p.data[r][c] -= lr * p.grad[r][c]
            # 处理偏置向量 (1D 列表)
            elif isinstance(p.data, list):
                for i in range(len(p.data)):
                    p.data[i] -= lr * p.grad[i]
            else:
                # 处理标量情况
                p.data -= lr * p.grad
                
        return loss.data

def train_model(target_func_name, save_history_path='training_history.pkl'):
    """
    执行完整的训练过程，记录每一步的损失和网络状态，并保存到文件。
    此函数不包含任何绘图代码，专注于计算。
    """
    print(f"\n=== 开始训练: 拟合 {target_func_name} ===")
    
    # 选择目标函数
    if target_func_name == 'double_peak':
        target_func = double_peak
    else:
        target_func = mexican_hat
        
    # 1. 生成数据集
    # 样本数适中，既能验证泛化性，又不至于让纯 Python 太慢
    X_train, Y_train = generate_data(target_func, n_samples=300, range_val=3.0)
    print(f"数据生成完毕：{len(X_train)} 个样本")
    
    # 2. 初始化网络
    # 输入2维 (x,y), 两个隐藏层各40个神经元, 输出1维 (z)
    model = RegressionNet(input_dim=2, hidden_dims=[40, 40], output_dim=1)
    
    # 超参数设置
    lr = 0.02       # 初始学习率
    epochs = 600    # 训练轮数
    # 注意：纯 Python 下 epoch 太多会很慢，800 轮通常足以收敛
    
    print("网络结构: 2 -> 40 (ReLU) -> 40 (ReLU) -> 1 (Linear)")
    print(f"超参数: Epochs={epochs}, LR={lr}")
    
    # 用于记录训练历史的数据结构
    history = {
        'loss_curve': [],          # 每个 epoch 的平均损失
        'snapshots': [],           # 关键帧的网络状态 (用于动画)
        'epochs': []               # 对应的 epoch 数
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        
        # 打乱数据顺序 (Shuffle)，提高训练稳定性
        indices = list(range(len(X_train)))
        random.shuffle(indices)
        
        # 逐个样本进行训练 (SGD)
        for i in indices:
            loss_val = model.train_step(X_train[i], Y_train[i], lr)
            total_loss += loss_val
        
        avg_loss = total_loss / len(X_train)
        history['loss_curve'].append(avg_loss)
        
        # 简单的学习率衰减策略：后半程减小学习率，使收敛更平稳
        if epoch == int(epochs * 0.6):
            lr *= 0.5
            # print(f"-> Epoch {epoch}: 学习率衰减为 {lr}")
            
        # 记录快照用于动画
        # 为了节省内存和时间，不是每轮都存，而是按频率存
        # 前100轮每10轮存一次，后面每20轮存一次
        save_interval = 10 if epoch <= 100 else 20
        
        if epoch % save_interval == 0 or epoch == 1 or epoch == epochs:
            # 深拷贝当前的网络参数状态 (简化版：只存损失和 epoch，实际动画需要重跑预测或存参数)
            # 为了动画流畅，我们这里只存损失和 epoch，预测曲面在可视化阶段通过“重放”或“插值”来做？
            # 不，最准确的方法是保存当时的参数。但纯 Python 深拷贝大列表很慢。
            # 优化策略：只在可视化阶段，根据保存的关键点参数重新构建网络进行预测。
            # 这里我们保存一份参数的深拷贝 (使用 pickle 序列化模拟深拷贝，或者手动复制)
            # 鉴于性能，我们只保存关键帧的参数快照
            snapshot_params = []
            for p in model.params:
                # 手动深拷贝列表
                if isinstance(p.data, list):
                    copied_data = [row[:] if isinstance(row, list) else row for row in p.data]
                    snapshot_params.append(copied_data)
                else:
                    snapshot_params.append(p.data)
            
            history['snapshots'].append(snapshot_params)
            history['epochs'].append(epoch)
            
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f} | LR: {lr:.4f}")

    end_time = time.time()
    print(f"\n训练完成！总耗时: {end_time - start_time:.2f} 秒")
    
    # 保存训练历史到文件，供可视化脚本使用
    with open(save_history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"训练历史已保存至: {save_history_path}")
    
    return history, model, X_train, Y_train, target_func

# ==========================================
# 第四部分：可视化与动画生成
# ==========================================

def visualize_training(history_path='training_history.pkl', func_name='mexican_hat', output_gif='fitting_process.gif'):
    """
    读取训练历史文件，生成 3D 曲面变化的动画和损失曲线动画。
    此函数不包含训练逻辑。
    """
    print("\n=== 开始生成可视化动画 ===")
    
    # 1. 加载训练历史
    if not os.path.exists(history_path):
        print(f"错误: 找不到历史文件 {history_path}。请先运行训练部分。")
        return
    
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    snapshots = history['snapshots']
    recorded_epochs = history['epochs']
    loss_curve = history['loss_curve']
    
    # 确定目标函数用于绘图
    if func_name == 'double_peak':
        target_func = double_peak
        title_name = "Double Peak (双峰)"
    else:
        target_func = mexican_hat
        title_name = "Mexican Hat (墨西哥帽)"
        
    # 2. 准备网格数据用于绘制曲面
    grid_size = 25  # 网格分辨率，动画中不宜过高以免卡顿
    range_val = 3.0
    xs = np.linspace(-range_val, range_val, grid_size)
    ys = np.linspace(-range_val, range_val, grid_size)
    X_grid, Y_grid = np.meshgrid(xs, ys)
    
    # 计算真实曲面 (固定不变)
    Z_true = np.zeros_like(X_grid)
    for i in range(grid_size):
        for j in range(grid_size):
            Z_true[i, j] = target_func(X_grid[i, j], Y_grid[i, j])
            
    # 3. 定义辅助函数：根据保存的参数快照重建网络并预测
    def predict_from_snapshot(snapshot_params, x_val, y_val):
        """
        利用保存的参数列表，临时构建网络并计算预测值
        为了避免重复定义类，这里简单复刻前向传播逻辑
        """
        # 临时重建网络结构 (硬编码结构以匹配训练时: 2->40->40->1)
        # 注意：这里需要访问 Layer 类来构建张量运算，或者直接复用 Tensor 逻辑
        # 为了简单，我们实例化一个空模型，然后强行注入参数
        
        model = RegressionNet(input_dim=2, hidden_dims=[40, 40], output_dim=1)
        
        # 注入参数
        param_idx = 0
        for layer in model.layers:
            # 恢复权重 W
            w_data = snapshot_params[param_idx]
            layer.W.data = w_data
            param_idx += 1
            
            # 恢复偏置 b
            b_data = snapshot_params[param_idx]
            layer.b.data = b_data
            param_idx += 1
            
        # 执行前向传播
        inp = Tensor([[x_val, y_val]])
        out = model.forward(inp)
        return out.data[0][0]

    # 预计算所有快照的预测曲面 (为了动画流畅，预先计算好所有帧的数据)
    print("正在预计算所有帧的预测曲面 (这可能需要一点时间)...")
    Z_pred_frames = []
    
    total_frames = len(snapshots)
    for idx, params in enumerate(snapshots):
        Z_pred = np.zeros_like(X_grid)
        # 进度条
        if idx % 5 == 0:
            print(f"处理帧 {idx}/{total_frames} (Epoch {recorded_epochs[idx]})")
            
        for i in range(grid_size):
            for j in range(grid_size):
                val = predict_from_snapshot(params, X_grid[i, j], Y_grid[i, j])
                Z_pred[i, j] = val
        Z_pred_frames.append(Z_pred)
        
    print("预计算完成，开始构建动画...")

    # 4. 创建画布
    fig = plt.figure(figsize=(14, 7))
    
    # --- 左侧：3D 曲面演化 ---
    ax3d = fig.add_subplot(121, projection='3d')
    # 绘制真实曲面作为背景 (半透明)
    ax3d.plot_surface(X_grid, Y_grid, Z_true, cmap='Greys', alpha=0.3, label='Ground Truth')
    
    # 初始化预测曲面 (第一帧)
    surf = ax3d.plot_surface(X_grid, Y_grid, Z_pred_frames[0], cmap='viridis', alpha=0.9, label='Prediction')
    
    # 绘制训练样本点 (散点)
    # 这里我们需要重新生成一下数据或者传入数据，为了简单，我们重新生成一份相同种子的数据
    # 或者在训练时把数据也存进 history。这里选择重新生成。
    seed = 42
    random.seed(seed)
    sample_x = []
    sample_y = []
    sample_z = []
    func = double_peak if func_name == 'double_peak' else mexican_hat
    for _ in range(300):
        x = random.uniform(-range_val, range_val)
        y = random.uniform(-range_val, range_val)
        sample_x.append(x)
        sample_y.append(y)
        sample_z.append(func(x, y))
        
    ax3d.scatter(sample_x, sample_y, sample_z, c='red', s=15, edgecolors='k', linewidth=0.5, label='Training Data')
    
    ax3d.set_title(f'拟合过程: {title_name}', fontsize=14)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_zlim(np.min(Z_true)*1.2, np.max(Z_true)*1.2)
    ax3d.legend(loc='upper left', fontsize=8)

    # --- 右侧：损失曲线演化 ---
    ax2d = fig.add_subplot(122)
    
    # 绘制完整的损失曲线 (灰色背景线)
    # 注意：loss_curve 是每个 epoch 的，而快照是稀疏的。
    # 我们需要映射快照的索引到 loss_curve 的索引
    # recorded_epochs 存储的是 [1, 10, 20, ...]
    # loss_curve 索引是 0 对应 epoch 1
    
    ax2d.plot(range(1, len(loss_curve)+1), loss_curve, 'gray', lw=1, alpha=0.5, label='Full Loss History')
    
    # 动态元素：当前点 和 已走过的轨迹
    current_point, = ax2d.plot([], [], 'ro', markersize=8, label='Current State')
    trace_line, = ax2d.plot([], [], 'b-', lw=2, label='Optimization Path')
    
    # 文本标注
    info_text = ax2d.text(0.05, 0.95, '', transform=ax2d.transAxes, 
                          verticalalignment='top', fontsize=10, 
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2d.set_xlabel('Epoch', fontsize=12)
    ax2d.set_ylabel('Loss (MSE)', fontsize=12)
    ax2d.set_title('训练损失下降曲线', fontsize=14)
    ax2d.set_xlim(1, len(loss_curve))
    ax2d.set_ylim(0, max(loss_curve)*1.1)
    ax2d.legend(loc='upper right')
    ax2d.grid(True, alpha=0.3)

    # 定义动画更新函数
    def update(frame_idx):
        # frame_idx 对应 snapshots 的索引
        
        # --- 更新 3D 曲面 ---
        # 方法：移除旧的 surface，绘制新的
        # 注意：ax3d.collections 包含了所有的 3D 集合对象
        # 我们保留第一个 (真实的灰色曲面) 和 散点图 (可能也被算作集合)
        # 为了安全，我们只删除最后一个添加的 surface (即预测曲面)
        # 但散点图也是集合。更好的方法是标记我们创建的 surf 对象
        
        # 由于 matplotlib 3D surface 更新困难，这里采用暴力重绘预测曲面
        # 先清除预测曲面 (假设它是最后加入的 collection，除了 scatter)
        # 实际上，我们可以直接修改 surf._faces 和 _facecolors，但这太底层了。
        # 这里为了稳定性，我们每次重新 plot_surface，并删除上一个
        if len(ax3d.collections) > 2: # 1 (true) + 1 (scatter) + 1 (pred)
             # 删除最后一个集合 (旧的预测曲面)
             ax3d.collections[-1].remove()
             
        # 绘制新的预测曲面
        new_surf = ax3d.plot_surface(X_grid, Y_grid, Z_pred_frames[frame_idx], 
                                     cmap='viridis', alpha=0.9, shade=True)
        
        # 更新标题显示当前 Epoch
        current_epoch = recorded_epochs[frame_idx]
        current_loss = loss_curve[current_epoch-1] # loss_curve 索引从 0 开始 (epoch 1)
        ax3d.set_title(f'拟合过程: {title_name} | Epoch: {current_epoch}', fontsize=14)

        # --- 更新 2D 损失曲线 ---
        # 获取当前帧对应的 epoch 索引
        epoch_num = recorded_epochs[frame_idx]
        
        # 更新当前红点
        current_point.set_data([epoch_num], [current_loss])
        
        # 更新蓝色轨迹线 (从第1轮到当前轮)
        # 注意：我们要画的是连续的损失曲线片段，而不是只连快照点
        # 所以 x 是 1 到 epoch_num, y 是 loss_curve[0 : epoch_num]
        trace_x = list(range(1, epoch_num + 1))
        trace_y = loss_curve[:epoch_num]
        trace_line.set_data(trace_x, trace_y)
        
        # 更新文本信息
        info_text.set_text(f'Epoch: {epoch_num}\nLoss: {current_loss:.5f}')
        
        return current_point, trace_line, info_text

    # 7. 创建动画对象
    # frames 设置为快照的数量
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(snapshots), 
        interval=400,  # 每帧间隔 400ms
        blit=False,    # 3D 动画建议关闭 blit，避免渲染错误
        repeat=True,
        repeat_delay=2000
    )

    # 8. 保存为 GIF
    print(f"正在保存 GIF 动画: {output_gif} ... (这可能需要几十秒)")
    try:
        writer = animation.PillowWriter(fps=5, bitrate=1800)
        ani.save(output_gif, writer=writer)
        print(f"动画保存成功: {output_gif}")
    except Exception as e:
        print(f"保存 GIF 失败: {e}")

    # 显示最终静态图 (如果不保存或保存后查看)
    plt.tight_layout()
    plt.show()

# ==========================================
# 第五部分：主程序入口
# ==========================================

if __name__ == '__main__':
    # 配置项
    TARGET_FUNCTION = 'mexican_hat'  # 可选: 'mexican_hat' 或 'double_peak'
    HISTORY_FILE = 'training_history.pkl'
    OUTPUT_GIF = 'fitting_process.gif'
    
    print("="*50)
    
    # 步骤 1: 训练模型 (纯计算，无绘图)
    # 如果已经训练过且想跳过，可以注释掉下面这行
    # history_data, final_model, data_x, data_y, func_ref = train_model(
    #     target_func_name=TARGET_FUNCTION, 
    #     save_history_path=HISTORY_FILE
    # )
    
    # 步骤 2: 生成可视化动画 (读取历史文件)
    visualize_training(
        history_path=HISTORY_FILE, 
        func_name=TARGET_FUNCTION, 
        output_gif=OUTPUT_GIF
    )
    
    print("\n全部流程结束！请查看生成的 GIF 文件。")