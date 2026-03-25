# model.py - NumPy加速版本
# 导入必要的库和模块
from tensor import Tensor  # 导入自定义的Tensor类，用于自动微分
import numpy as np  # 使用NumPy进行数值计算


def randn(shape):
    """Xavier 初始化函数
    
    Xavier初始化是一种常用的神经网络权重初始化方法，
    可以帮助网络在训练过程中更快地收敛。
    
    参数:
        shape: 权重或偏置的形状，支持二维(权重)或一维(偏置)
        
    返回:
        初始化后的权重或偏置值（NumPy数组）
    """
    if len(shape) == 2:
        rows, cols = shape
        # 计算Xavier初始化的范围
        limit = np.sqrt(6.0 / (rows + cols))
        # 生成指定范围内的随机数矩阵
        return np.random.uniform(-limit, limit, size=shape).astype(np.float64)
    elif len(shape) == 1:
        cols = shape[0]
        # 计算Xavier初始化的范围（假设输入维度为1）
        limit = np.sqrt(6.0 / (1 + cols))
        # 生成指定范围内的随机数列表
        return np.random.uniform(-limit, limit, size=shape).astype(np.float64)
    # 其他情况返回0.0
    return np.array(0.0, dtype=np.float64)


class Layer:
    """神经网络的基本层类
    
    实现了一个全连接层，包含权重和偏置参数，
    并定义了前向传播的计算逻辑。
    """
    def __init__(self, nin, nout):
        """初始化层
        
        参数:
            nin: 输入特征的维度
            nout: 输出特征的维度
        """
        # 初始化权重矩阵 (nin, nout) 和偏置向量 (nout,)
        self.W = Tensor(randn([nin, nout]), label='W')  # 权重矩阵，使用Xavier初始化
        self.b = Tensor(randn([nout]), label='b')  # 偏置向量，使用Xavier初始化
        # 收集所有可学习参数
        self.params = [self.W, self.b]

    def __call__(self, x):
        """前向传播计算
        
        参数:
            x: 输入张量，形状为 (Batch, nin)
            
        返回:
            输出张量，形状为 (Batch, nout)
        """
        # 矩阵乘法: x @ W
        # x的形状是 (B, nin)，W的形状是 (nin, nout)，结果形状是 (B, nout)
        out = x.matmul(self.W)
        
        # 广播偏置: 将形状为 (nout,) 的偏置扩展为 (B, nout)
        B = out.data.shape[0]  # 获取批量大小
        
        # 构造与输出形状相同的偏置张量
        # 使用NumPy广播机制
        b_tensor_data = np.tile(self.b.data, (B, 1))
        
        # 创建偏置张量
        b_tensor = Tensor(b_tensor_data)
        # 返回线性变换结果: x @ W + b
        return out + b_tensor
