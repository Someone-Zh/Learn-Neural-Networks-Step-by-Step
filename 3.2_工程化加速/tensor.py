# 张量类实现自动微分功能 - NumPy加速版本
import numpy as np

class Tensor:
    def __init__(self, data, _prev=(), _op='', label=''):
        # 存储张量的实际数据，转换为NumPy数组
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        elif isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            self.data = np.array(data, dtype=np.float64)
            
        # 初始化梯度：与数据同形状的零值数组
        self.grad = np.zeros_like(self.data, dtype=np.float64)
            
        # 反向传播函数，默认为空操作
        self._backward = lambda: None
        # 前驱节点集合，用于构建计算图
        self._prev = set(_prev)
        # 操作符名称，用于调试和显示
        self._op = _op
        # 张量标签，便于识别
        self.label = label
        # 是否保留计算图的标记
        self._retain_graph = True
    
    def _validate_grad_shape(self):
        """验证梯度形状与数据形状是否匹配，避免广播错误"""
        if self.data.shape != self.grad.shape:
            raise ValueError(
                f"梯度形状不匹配: 数据形状 {self.data.shape}, "
                f"梯度形状 {self.grad.shape}"
            )
    
    def zero_grad(self, release_graph=True):
        """清零梯度，可选择是否释放计算图
        
        参数:
            release_graph: 是否释放计算图节点，默认为 True
        """
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        
        # 释放计算图
        if release_graph:
            self._prev = set()
            self._backward = lambda: None
            self._retain_graph = False

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"

    # 确保输入参数为Tensor类型，如果不是则进行转换
    def _ensure_tensor(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        # 使用NumPy广播机制进行加法
        out_data = self.data + other.data
            
        # 创建新的Tensor对象
        out = Tensor(out_data, (self, other), '+')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 处理梯度更新，考虑广播
            self.grad = self.grad + _unbroadcast(g, self.data.shape)
            other.grad = other.grad + _unbroadcast(g, other.data.shape)
        # 设置反向传播函数
        out._backward = _backward
        return out

    def __mul__(self, other):
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        # 使用NumPy进行逐元素乘法
        out_data = self.data * other.data

        # 创建新的Tensor对象
        out = Tensor(out_data, (self, other), '*')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 处理梯度更新，考虑广播
            self.grad = self.grad + _unbroadcast(g * other.data, self.data.shape)
            other.grad = other.grad + _unbroadcast(g * self.data, other.data.shape)
        # 设置反向传播函数
        out._backward = _backward
        return out

    def __pow__(self, power):
        # 验证幂次必须是数字
        assert isinstance(power, (int, float))
        # 使用NumPy计算幂运算结果
        out_data = self.data ** power
            
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), f'**{power}')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 计算局部导数并更新梯度
            local_grad = power * (self.data ** (power - 1))
            self.grad = self.grad + local_grad * g
        # 设置反向传播函数
        out._backward = _backward
        return out

    # 负号运算：返回自身的相反数
    def __neg__(self): return self * -1
    # 减法运算：通过加法和负号实现
    def __sub__(self, other):
        other = self._ensure_tensor(other)
        return self + (-other)
    
    # 右操作方法：支持 other op self 形式的运算
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rtruediv__(self, other):
        return self._ensure_tensor(other) * (self ** -1)
    # 除法运算：通过乘法和负指数实现
    def __truediv__(self, other): 
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        # 检查除数是否为零
        if np.any(other.data == 0):
            raise ValueError("除数不能为零")
        return self * (other ** -1)

    def matmul(self, other):
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        # 获取两个张量的数据
        A, B = self.data, other.data
        
        # 验证输入类型
        if A.ndim < 1 or B.ndim < 1:
            raise ValueError("矩阵乘法要求两个操作数都是数组")
        
        # 处理一维数组转二维
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        
        # 验证矩阵维度是否匹配
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"矩阵维度不匹配: 第一个矩阵的列数({A.shape[1]}) != 第二个矩阵的行数({B.shape[0]})")

        # 使用NumPy执行矩阵乘法运算
        out_data = np.matmul(A, B)
        # 创建新的Tensor对象
        out = Tensor(out_data, (self, other), '@')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 获取原始形状
            orig_A_shape = A.shape
            orig_B_shape = B.shape
            
            # 计算A的梯度：dA = g @ B.T
            dA = np.matmul(g, B.T)
            # 计算B的梯度：dB = A.T @ g
            dB = np.matmul(A.T, g)

            # 恢复原始形状的梯度
            if self.data.ndim == 1:
                dA = dA.flatten()
            if other.data.ndim == 1:
                dB = dB.flatten()
            
            # 更新梯度
            self.grad = self.grad + dA
            other.grad = other.grad + dB
        # 设置反向传播函数
        out._backward = _backward
        return out

    def sum(self):
        # 使用NumPy计算总和
        total = np.sum(self.data)
        # 创建表示总和的新张量
        out = Tensor(total, (self,), 'sum')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 将梯度分配给原张量的所有元素
            self.grad = self.grad + np.ones_like(self.data) * g
        # 设置反向传播函数
        out._backward = _backward
        return out

    def exp(self):
        # 使用NumPy计算自然指数函数e^x
        out_data = np.exp(self.data)
        # 创建表示指数函数结果的新张量
        out = Tensor(out_data, (self,), 'exp')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 指数函数的导数是其自身，应用链式法则
            self.grad = self.grad + out.data * g
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    def relu(self):
        """ReLU激活函数"""
        out_data = np.maximum(0, self.data)
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), 'relu')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # ReLU的导数是：x>0时为1，否则为0
            local = (self.data > 0).astype(np.float64)
            self.grad = self.grad + local * g
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    def tanh(self):
        """tanh激活函数"""
        out_data = np.tanh(self.data)
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), 'tanh')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # tanh的导数是：1 - tanh^2(x)
            local = 1.0 - out.data ** 2
            self.grad = self.grad + local * g
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    def sigmoid(self):
        """sigmoid激活函数"""
        out_data = 1.0 / (1.0 + np.exp(-self.data))
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), 'sigmoid')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # sigmoid的导数是：sigmoid(x) * (1 - sigmoid(x))
            local = out.data * (1.0 - out.data)
            self.grad = self.grad + local * g
        # 设置反向传播函数
        out._backward = _backward
        return out

    def log(self):
        # 使用NumPy计算自然对数函数ln(x)
        # 检查是否有非正数
        if np.any(self.data <= 0):
            raise ValueError("对数函数要求输入值大于0")
        out_data = np.log(self.data)
        # 创建表示对数函数结果的新张量
        out = Tensor(out_data, (self,), 'log')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 对数函数的导数是1/x
            self.grad = self.grad + (1.0 / self.data) * g
        # 设置反向传播函数
        out._backward = _backward
        return out

    def softmax(self):
        """Softmax激活函数
        
        对输入的每一行计算 softmax:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x - max(x)))
        
        数值稳定性：减去最大值防止溢出
        
        返回:
            应用 softmax 后的 Tensor
        """
        # 确保输入是二维数组（batch, features）
        if self.data.ndim == 1:
            # 一维输入 (features,)
            max_val = np.max(self.data)
            exp_vals = np.exp(self.data - max_val)
            out_data = exp_vals / np.sum(exp_vals)
        elif self.data.ndim == 2:
            # 二维输入 (batch, features)
            max_vals = np.max(self.data, axis=1, keepdims=True)
            exp_vals = np.exp(self.data - max_vals)
            out_data = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            raise ValueError("Softmax 要求输入为一维或二维数组")
        
        out = Tensor(out_data, (self,), 'softmax')
        
        def _backward():
            """Softmax 反向传播
            
            softmax 的 Jacobian 矩阵为:
            J[i][j] = s[i] * (delta_ij - s[j])
            
            对于 batch 输入，每个样本独立计算
            """
            g = out.grad
            
            if self.data.ndim == 2:
                # 二维输入 (batch, features)
                s = out.data  # softmax 输出
                # 计算梯度: dL/dx = s * (g - sum(g * s))
                sum_gs = np.sum(g * s, axis=1, keepdims=True)
                grad = s * (g - sum_gs)
                self.grad = self.grad + grad
            else:
                # 一维输入 (features,)
                s = out.data
                sum_gs = np.sum(g * s)
                grad = s * (g - sum_gs)
                self.grad = self.grad + grad
        
        out._backward = _backward
        return out
    
    def cross_entropy_loss(self, targets):
        """交叉熵损失函数
        
        参数:
            targets: 真实标签的 one-hot 编码 Tensor，形状为 (Batch, num_classes)
        
        返回:
            平均交叉熵损失（标量 Tensor）
        
        公式: L = -1/N * sum(sum(targets * log(softmax_output)))
        """
        # 先计算 softmax
        probs = self.softmax()
        
        # 计算交叉熵: -sum(targets * log(probs + epsilon))
        # 添加小常数防止 log(0)
        epsilon = 1e-15
        batch_size = probs.data.shape[0]
        
        # clip 防止 log(0)
        probs_clipped = np.clip(probs.data, epsilon, 1 - epsilon)
        
        # 计算每个样本的交叉熵
        log_probs = np.log(probs_clipped)
        sample_losses = -np.sum(targets.data * log_probs, axis=1)
        
        # 平均损失
        avg_loss = np.mean(sample_losses)
        
        # 创建损失 Tensor
        out = Tensor(avg_loss, (probs,), 'cross_entropy')
        
        def _backward():
            """交叉熵 + softmax 的反向传播
            
            组合导数: dL/dz = probs - targets
            这是因为 softmax + cross_entropy 有优美的简化形式
            """
            g = out.grad  # 标量梯度
            # dL/dz = (probs - targets) * grad_from_upstream / batch_size
            probs.grad = probs.grad + g * (probs.data - targets.data) / batch_size
        
        out._backward = _backward
        return out
    
    def backward(self, retain_graph=False):
        """反向传播，可选择是否保留计算图
        
        参数:
            retain_graph: 是否保留计算图用于多次反向传播，默认为 False
        """
        # 拓扑排序列表，用于按正确顺序执行反向传播
        topo = []
        # 已访问节点集合，避免重复访问
        visited = set()
        
        # 构建拓扑排序
        def build(v):
            if v not in visited:
                visited.add(v)
                # 递归访问前驱节点
                for child in v._prev:
                    build(child)
                # 添加当前节点到拓扑排序列表
                topo.append(v)
        build(self)
        
        # 初始化最终梯度为1.0
        self.grad = np.ones_like(self.data, dtype=np.float64)

        # 按逆序执行反向传播
        for node in reversed(topo):
            node._backward()
            # 梯度检查
            try:
                node._validate_grad_shape()
            except ValueError as e:
                raise ValueError(
                    f"反向传播时梯度形状验证失败 (节点: {node._op}, 标签: {node.label}):\n{e}"
                )
        
        # 释放计算图（除非明确要求保留）
        if not retain_graph:
            for node in topo:
                node._prev = set()
                node._backward = lambda: None
                node._retain_graph = False


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