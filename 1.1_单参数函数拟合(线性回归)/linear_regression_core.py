"""
线性回归计算核心模块
可作为独立库被其他文件调用，包含数据生成、模型训练等纯计算逻辑
"""
import numpy as np

def box_muller_transform(n_samples):
    """
    手写Box-Muller变换生成标准高斯噪声 N(0,1)
    :param n_samples: 生成的样本数量
    :return: 标准正态分布的噪声数组
    """
    # 生成均匀分布的随机数
    u1 = np.random.uniform(0, 1, n_samples)
    u2 = np.random.uniform(0, 1, n_samples)
    
    # Box-Muller公式
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z0

def generate_data(n_samples=100):
    """
    生成带高斯噪声的线性数据 y = 2x + 3 + N(0,1)
    :param n_samples: 样本数量
    :return: x, y (numpy数组)
    """
    # 生成x ∈ [10, 100]
    x = np.linspace(10, 100, n_samples)
    
    # 生成高斯噪声（Box-Muller变换）
    noise = box_muller_transform(n_samples)
    
    # 构造y = 2x + 3 + 噪声
    y = 2 * x + 3 + noise
    
    return x, y

def linear_regression_train(x, y, lr=0.0001, epochs=500, verbose=True):
    """
    手动实现线性回归（y = wx + b）的梯度下降训练
    :param x: 输入特征
    :param y: 真实标签
    :param lr: 学习率
    :param epochs: 训练轮数
    :param verbose: 是否打印训练日志
    :return: 训练过程记录（w_list, b_list, loss_list）
    """
    # 初始化参数
    np.random.seed(42)  # 设置随机种子保证结果可复现
    w = np.random.randn()  # 随机初始化w
    b = np.random.randn()  # 随机初始化b
    n = len(x)  # 样本数量
    
    # 记录训练过程
    w_list = [w]
    b_list = [b]
    loss_list = []
    
    for epoch in range(epochs):
        # 前向计算：y_pred = wx + b
        y_pred = w * x + b
        
        # 计算损失：MSE = (1/N) * Σ(y_pred - y_true)²
        loss = np.sum((y_pred - y) ** 2) / n
        loss_list.append(loss)
        
        # 手动计算梯度
        # ∂Loss/∂w = (2/N) * Σ((y_pred - y_true) * x)
        dw = (2 / n) * np.sum((y_pred - y) * x)
        # ∂Loss/∂b = (2/N) * Σ(y_pred - y_true)
        db = (2 / n) * np.sum(y_pred - y)
        
        # 参数更新
        w -= lr * dw
        b -= lr * db
        
        # 记录参数
        w_list.append(w)
        b_list.append(b)
        
        # 打印训练进度
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")
    
    if verbose:
        print(f"\n最终结果 | w: {w:.4f} (目标: 2) | b: {b:.4f} (目标: 3) | 最终Loss: {loss:.4f}")
    
    return {
        "w_list": w_list,
        "b_list": b_list,
        "loss_list": loss_list,
        "final_w": w,
        "final_b": b,
        "x": x,
        "y": y
    }

# 测试代码（仅在直接运行该文件时执行）
if __name__ == "__main__":
    # 测试数据生成
    x, y = generate_data(100)
    print("数据生成测试：")
    print(f"x.shape: {x.shape}, y.shape: {y.shape}")
    print(f"x范围: [{x.min():.1f}, {x.max():.1f}]")
    
    # 测试模型训练
    print("\n模型训练测试：")
    train_result = linear_regression_train(x, y, epochs=100)
    print(f"最终w: {train_result['final_w']:.4f}, 最终b: {train_result['final_b']:.4f}")