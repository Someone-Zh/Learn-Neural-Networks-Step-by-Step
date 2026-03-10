"""
多参数非线性回归计算核心模块（房价预测）
核心：保留原始二次项特征，仅对梯度计算做数值稳定处理
"""
import numpy as np

def generate_multi_param_data(n_samples=300, noise_std=3.0):
    """生成非线性房价数据：面积120㎡最优、楼层越低越贵"""
    np.random.seed(42)
    
    # 特征范围：面积80~160㎡，楼层1~30层
    x1 = np.random.uniform(80, 160, n_samples)   # 面积（㎡）
    x2 = np.random.randint(1, 31, n_samples)     # 楼层（层）
    X = np.column_stack((x1, x2))
    
    # 非线性房价公式（核心：保留原始二次项）
    area_term = -0.05 * (x1 - 120) ** 2  # 120㎡最优（二次项）
    floor_term = -2 * x2                 # 楼层越低越贵（负系数）
    noise = np.random.normal(0, noise_std, n_samples)
    y = area_term + floor_term + 200 + noise
    y = np.clip(y, a_min=50, a_max=None)
    
    return X, y

def multi_param_gradient_descent(X, y, lr=0.000001, epochs=8000, verbose=True):
    """
    非线性梯度下降（数值稳定版）
    关键：直接使用原始二次项，仅做梯度裁剪+小学习率，不做特征归一化（避免线性化）
    """
    np.random.seed(42)
    # 初始化参数（接近目标值，减少震荡）
    w1 = -0.02  # 面积二次项系数（目标-0.05）
    w2 = -1.0   # 楼层系数（目标-2）
    b = 190     # 基础价（目标200）
    n = len(X)
    
    # 提取原始特征（保留二次项）
    x1 = X[:, 0]
    x2 = X[:, 1]
    x1_quad = (x1 - 120) ** 2  # 原始二次项，不做归一化
    
    # 记录训练过程
    w1_list = [w1]
    w2_list = [w2]
    b_list = [b]
    loss_list = []
    
    for epoch in range(epochs):
        # 前向计算（原始二次项，保证非线性）
        y_pred = w1 * x1_quad + w2 * x2 + b
        
        # 数值稳定的损失计算
        loss = np.mean(np.clip((y_pred - y) ** 2, 0, 1e5))
        loss_list.append(loss)
        
        # 梯度计算 + 裁剪（防止爆炸）
        grad_w1 = 2 * np.mean(np.clip((y_pred - y) * x1_quad, -1e3, 1e3))
        grad_w2 = 2 * np.mean(np.clip((y_pred - y) * x2, -1e3, 1e3))
        grad_b = 2 * np.mean(np.clip(y_pred - y, -1e3, 1e3))
        
        # 参数更新（小步长+约束）
        w1 = np.clip(w1 - lr * grad_w1, -0.1, 0)    # 约束在合理范围
        w2 = np.clip(w2 - lr * grad_w2, -3, 0)      # 约束在合理范围
        b = np.clip(b - lr * grad_b, 180, 220)      # 约束在合理范围
        
        # 记录参数
        w1_list.append(w1)
        w2_list.append(w2)
        b_list.append(b)
        
        # 打印日志
        if verbose and epoch % 1000 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | 面积二次项w1: {w1:.4f} | 楼层w2: {w2:.4f} | 基础价b: {b:.2f}")
    
    if verbose:
        print("\n" + "="*70)
        print("训练完成 | 目标值：w1=-0.05, w2=-2, b=200")
        print(f"最终值：w1={w1:.4f}, w2={w2:.4f}, b={b:.2f} | 最终Loss: {loss:.4f}")
        print(f"拟合公式：房价 = {w1:.4f}×(面积-120)² + {w2:.4f}×楼层 + {b:.2f}")
        print("="*70)
    
    return {
        "w1_list": w1_list,
        "w2_list": w2_list,
        "b_list": b_list,
        "loss_list": loss_list,
        "final_w1": w1,
        "final_w2": w2,
        "final_b": b,
        "X": X,
        "y": y
    }

# 测试代码
if __name__ == "__main__":
    X, y = generate_multi_param_data()
    print(f"数据生成 | 样本数：{len(X)} | 房价范围：[{y.min():.1f}, {y.max():.1f}]万元")
    train_result = multi_param_gradient_descent(X, y, epochs=8000)