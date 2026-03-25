# loss.py
from tensor import Tensor
import math

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
    if isinstance(sq.data, list):
        n = sum(len(row) for row in sq.data)
    else:
        n = 1
    
    # 返回均方误差
    return sum_sq / n

# 交叉熵损失函数
def manual_cross_entropy_loss(logits, targets):
    # logits: Tensor (批量大小, 类别数)
    # targets: 整数列表 (类别索引)
    
    # 1. 计算 LogSoftmax 以提高数值稳定性
    data = logits.data
    batch_size = len(data)
    num_classes = len(data[0])
    
    # 计算每一行的最大值
    max_vals = [max(row) for row in data]
    
    # 移除以最大值，避免指数爆炸
    shifted_data = [[data[i][j] - max_vals[i] for j in range(num_classes)] for i in range(batch_size)]
    shifted = Tensor(shifted_data)
    
    # 计算指数值
    exp_shifted = shifted.exp()
    
    # 计算每行的指数和
    sum_exp_data = [sum(row) for row in exp_shifted.data]
    # 计算对数和
    log_sum_exp_data = [math.log(s) for s in sum_exp_data]
    
    # 构造对数和矩阵用于广播减法
    log_sum_exp_matrix_data = [[log_sum_exp_data[i] for _ in range(num_classes)] for i in range(batch_size)]
    log_sum_exp_matrix = Tensor(log_sum_exp_matrix_data)
    
    # 计算对数概率
    log_probs = shifted - log_sum_exp_matrix
    
    # 计算负对数似然损失（使用掩码，便于自动求导）
    # 创建掩码：目标位置为1，其他位置为0
    mask_data = [[1.0 if j == targets[i] else 0.0 for j in range(num_classes)] for i in range(batch_size)]
    mask = Tensor(mask_data)
    
    # 计算损失：-sum(log_probs * mask) / N
    product = log_probs * mask
    total = product.sum()
    final_loss = (-total) / batch_size
    
    return final_loss