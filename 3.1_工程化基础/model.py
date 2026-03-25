# model.py
# 导入必要的库和模块
from tensor import Tensor  # 导入自定义的Tensor类，用于自动微分
import random  # 用于生成随机数
import math  # 用于数学计算


def randn(shape):
    """Xavier 初始化函数
    
    Xavier初始化是一种常用的神经网络权重初始化方法，
    可以帮助网络在训练过程中更快地收敛。
    
    参数:
        shape: 权重或偏置的形状，支持二维列表(权重)或一维列表(偏置)
        
    返回:
        初始化后的权重或偏置值
    """
    # 处理二维形状的情况（权重矩阵）
    if isinstance(shape[0], int) and len(shape) == 2:
        rows, cols = shape
        # 计算Xavier初始化的范围
        limit = math.sqrt(6.0 / (rows + cols))
        # 生成指定范围内的随机数矩阵
        return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]
    # 处理一维形状的情况（偏置向量）
    elif len(shape) == 1:
        cols = shape[0]
        # 计算Xavier初始化的范围（假设输入维度为1）
        limit = math.sqrt(6.0 / (1 + cols))
        # 生成指定范围内的随机数列表
        return [random.uniform(-limit, limit) for _ in range(cols)]
    # 其他情况返回0.0
    return 0.0


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
        B = len(out.data)  # 获取批量大小
        K = len(self.b.data)  # 获取偏置的维度
        
        # 构造与输出形状相同的偏置张量
        b_tensor_data = []
        for i in range(B):
            b_tensor_data.append(self.b.data)  # 为每个样本复制偏置数据
        
        # 创建偏置张量
        b_tensor = Tensor(b_tensor_data)
        # 返回线性变换结果: x @ W + b
        return out + b_tensor


class SimpleFCN:
    """简单的全连接神经网络
    
    实现了一个三层的全连接神经网络，包含两个隐藏层和一个输出层，
    使用ReLU作为激活函数，MSE作为损失函数。
    """
    def __init__(self, hidden_size=16):
        """初始化神经网络
        
        参数:
            hidden_size: 隐藏层的神经元数量，默认为16
        """
        # 定义网络结构
        # 输入层: 2个特征 (x, y)
        # 隐藏层1: hidden_size个神经元
        # 隐藏层2: hidden_size个神经元
        # 输出层: 1个特征 (z)
        self.layer1 = Layer(2, hidden_size)  # 输入层到隐藏层1
        self.layer2 = Layer(hidden_size, hidden_size)  # 隐藏层1到隐藏层2
        self.layer3 = Layer(hidden_size, 1)  # 隐藏层2到输出层
        
        # 收集所有可学习参数
        self.params = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            self.params.extend(layer.params)
            
        # 用于记录训练过程中的损失值
        self.loss_history = []

    def forward(self, x_tensor):
        """前向传播计算
        
        参数:
            x_tensor: 输入张量，形状为 (Batch, 2)
            
        返回:
            输出张量，形状为 (Batch, 1)
        """
        # 第一层前向传播
        h1 = self.layer1(x_tensor)
        
        # ReLU激活函数: max(0, x)
        def relu(t):
            """ReLU激活函数实现
            
            参数:
                t: 输入张量
                
            返回:
                应用ReLU后的张量
            """
            # 处理不同形状的输入数据
            if isinstance(t.data, list):
                # 处理二维数据（批量数据）
                new_data = [[max(0, val) for val in row] for row in t.data]
            else:
                # 处理标量数据
                new_data = max(0, t.data)
            
            # 创建新的张量，记录计算图
            out = Tensor(new_data, (t,), 'relu')
            
            # 定义反向传播函数
            def _backward():
                """ReLU的反向传播
                
                ReLU的导数是：如果输入大于0则为1，否则为0
                """
                g = out.grad  # 获取上游梯度
                # 处理不同形状的梯度
                if isinstance(t.grad, list):
                    if isinstance(t.data[0], list):
                        # 处理二维梯度（批量数据）
                        mask = [[1 if t.data[i][j] > 0 else 0 for j in range(len(t.data[0]))] for i in range(len(t.data))]
                        t.grad = [[t.grad[i][j] + mask[i][j] * g[i][j] for j in range(len(t.grad[0]))] for i in range(len(t.grad))]
                    else:
                        # 处理一维梯度
                        mask = [1 if t.data[i] > 0 else 0 for i in range(len(t.data))]
                        t.grad = [t.grad[i] + mask[i] * g[i] for i in range(len(t.grad))]
                else:
                    # 处理标量梯度
                    t.grad += (1 if t.data > 0 else 0) * g
            
            # 绑定反向传播函数
            out._backward = _backward
            return out
        
        # 应用ReLU激活函数
        h1 = relu(h1)
        
        # 第二层前向传播
        h2 = self.layer2(h1)
        # 应用ReLU激活函数
        h2 = relu(h2)
        
        # 输出层前向传播
        out = self.layer3(h2)
        return out

    def train_step(self, X_data, z_data, lr, optim_class):
        """单步训练
        
        参数:
            X_data: 输入数据，形状为 (Batch, 2)
            z_data: 标签数据，形状为 (Batch,)
            lr: 学习率
            optim_class: 优化器类
            
        返回:
            当前步的损失值
        """
        # 将数据转换为Tensor
        X = Tensor(X_data)  # 输入数据
        z_true = Tensor([[val] for val in z_data])  # 标签数据，形状为 (Batch, 1)
        
        # 前向传播
        z_pred = self.forward(X)
        
        # 计算损失（均方误差MSE）
        diff = z_pred - z_true  # 预测值与真实值的差
        sq = diff ** 2  # 平方
        sum_sq = sq.sum()  # 求和
        n = len(z_data)  # 样本数量
        loss = sum_sq / n  # 平均损失
        
        # 清零梯度
        for p in self.params:
            if isinstance(p.grad, list):
                if isinstance(p.grad[0], list):
                    # 清零二维梯度
                    p.grad = [[0.0 for _ in row] for row in p.grad]
                else:
                    # 清零一维梯度
                    p.grad = [0.0 for _ in p.grad]
            else:
                # 清零标量梯度
                p.grad = 0.0
                
        # 反向传播
        loss.backward()
        
        # 优化器更新参数
        # 注意：这里为了演示方便，每次训练步都创建一个新的优化器实例
        # 在实际代码中，应该保持优化器实例
        opt = optim_class(self.params, lr=lr)
        opt.step()
        
        # 记录损失值
        self.loss_history.append(loss.data)
        return loss.data