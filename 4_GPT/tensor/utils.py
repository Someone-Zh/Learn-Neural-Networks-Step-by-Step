import numpy as np

def _unbroadcast(grad, target_shape):
    """将广播后的梯度还原为原始形状
    
    参数:
        grad: 广播后的梯度
        target_shape: 目标形状
        
    返回:
        还原后的梯度
    """
    # 如果目标是标量
    if target_shape == ():
        return np.sum(grad)
    
    # 如果形状已经匹配
    if grad.shape == target_shape:
        return grad
    
    # 计算需要求和的轴
    # 处理维度数不同的情况
    ndim_diff = len(grad.shape) - len(target_shape)
    
    # 在前面添加1以便对齐维度
    padded_target = (1,) * ndim_diff + target_shape
    
    # 找出哪些轴需要求和
    sum_axes = []
    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, padded_target)):
        if t_dim == 1 and g_dim > 1:
            sum_axes.append(i)
    
    # 执行求和
    result = np.sum(grad, axis=tuple(sum_axes) if sum_axes else None)
    
    # 如果结果形状还不匹配，尝试reshape
    if result.shape != target_shape:
        result = result.reshape(target_shape)
    
    return result