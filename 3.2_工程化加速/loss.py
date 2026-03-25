# loss.py - NumPy加速版本
from tensor import Tensor
import numpy as np

# 均方误差损失函数
def manual_mse_loss(y_pred, y_true):
    # 确保 y_true 是 Tensor 类型
    if not isinstance(y_true, Tensor):
        y_true = Tensor(y_true)
    
    # 计算预测值与真实值的差值
    diff = y_pred - y_true
    # 计算差值的平方
    sq = diff ** 2
    # 计算平方和
    sum_sq = sq.sum()
    
    # 计算元素数量
    n = y_pred.data.size
    
    # 返回均方误差
    return sum_sq / n

# 交叉熵损失函数
def manual_cross_entropy_loss(logits, targets):
    # logits: Tensor (批量大小, 类别数)
    # targets: 整数列表 (类别索引)
    
    # 1. 计算 LogSoftmax 以提高数值稳定性
    data = logits.data
    batch_size = data.shape[0]
    num_classes = data.shape[1]
    
    # 计算每一行的最大值
    max_vals = np.max(data, axis=1, keepdims=True)
    
    # 移除最大值，避免指数爆炸
    shifted_data = data - max_vals
    shifted = Tensor(shifted_data)
    
    # 计算指数值
    exp_shifted = shifted.exp()
    
    # 计算每行的指数和
    sum_exp_data = np.sum(exp_shifted.data, axis=1)
    # 计算对数和
    log_sum_exp_data = np.log(sum_exp_data)
    
    # 构造对数和矩阵用于广播减法
    log_sum_exp_matrix_data = np.tile(log_sum_exp_data[:, np.newaxis], (1, num_classes))
    log_sum_exp_matrix = Tensor(log_sum_exp_matrix_data)
    
    # 计算对数概率
    log_probs = shifted - log_sum_exp_matrix
    
    # 计算负对数似然损失（使用掩码，便于自动求导）
    # 创建掩码：目标位置为1，其他位置为0
    mask_data = np.zeros((batch_size, num_classes), dtype=np.float64)
    for i, target in enumerate(targets):
        mask_data[i, int(target)] = 1.0
    mask = Tensor(mask_data)
    
    # 计算损失：-sum(log_probs * mask) / N
    product = log_probs * mask
    total = product.sum()
    final_loss = (-total) / batch_size
    
    return final_loss