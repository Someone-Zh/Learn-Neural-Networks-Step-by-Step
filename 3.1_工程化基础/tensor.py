# 张量类实现自动微分功能
import math

class Tensor:
    def __init__(self, data, _prev=(), _op='', label=''):
        # 存储张量的实际数据
        self.data = data
        # 初始化梯度：标量为0，列表/矩阵为同形状零值
        if isinstance(data, (int, float)):
            self.grad = 0.0
        elif isinstance(data, list):
            if not data:
                self.grad = []
            elif isinstance(data[0], list):
                self.grad = [[0.0 for _ in row] for row in data]
            else:
                self.grad = [0.0 for _ in data]
        else:
            raise ValueError("数据必须是标量或列表")
            
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
        if isinstance(self.data, (int, float)):
            if isinstance(self.grad, list):
                raise ValueError(
                    f"梯度形状不匹配: 数据是标量 {type(self.data).__name__}, "
                    f"但梯度是列表 {type(self.grad).__name__}"
                )
        elif isinstance(self.data, list):
            if not isinstance(self.grad, list):
                raise ValueError(
                    f"梯度形状不匹配: 数据是列表, 但梯度是 {type(self.grad).__name__}"
                )
            if len(self.data) != len(self.grad):
                raise ValueError(
                    f"梯度形状不匹配: 数据长度 {len(self.data)}, 梯度长度 {len(self.grad)}"
                )
            if self.data and isinstance(self.data[0], list):
                if not self.grad or not isinstance(self.grad[0], list):
                    raise ValueError(
                        "梯度形状不匹配: 数据是二维列表, 但梯度不是"
                    )
                for i, (data_row, grad_row) in enumerate(zip(self.data, self.grad)):
                    if len(data_row) != len(grad_row):
                        raise ValueError(
                            f"梯度形状不匹配: 第 {i} 行数据长度 {len(data_row)}, "
                            f"梯度长度 {len(grad_row)}"
                        )
            elif self.data and not isinstance(self.data[0], list):
                if self.grad and isinstance(self.grad[0], list):
                    raise ValueError(
                        "梯度形状不匹配: 数据是一维列表, 但梯度是二维列表"
                    )
    
    def zero_grad(self, release_graph=True):
        """清零梯度，可选择是否释放计算图
        
        参数:
            release_graph: 是否释放计算图节点，默认为 True
        """
        if isinstance(self.data, (int, float)):
            self.grad = 0.0
        elif isinstance(self.data, list):
            if not self.data:
                self.grad = []
            elif isinstance(self.data[0], list):
                self.grad = [[0.0 for _ in row] for row in self.data]
            else:
                self.grad = [0.0 for _ in self.data]
        
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
        # 简化广播：仅支持 标量+矩阵 或 矩阵+矩阵(同形)
        if isinstance(self.data, list) and not isinstance(other.data, list):
            out_data = [[x + other.data for x in row] for row in self.data]
        elif isinstance(self.data, list) and isinstance(other.data, list):
            out_data = [[self.data[i][j] + other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))]
        else:
            out_data = self.data + other.data
            
        # 创建新的Tensor对象
        out = Tensor(out_data, (self, other), '+')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 处理梯度更新
            if isinstance(self.grad, list):
                if isinstance(other.data, list):
                    # 如果两个操作数都是矩阵，则各自更新梯度
                    self.grad = [[self.grad[i][j] + g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                    other.grad = [[other.grad[i][j] + g[i][j] for j in range(len(other.grad[0]))] for i in range(len(other.grad))]
                else:
                    # 如果other是标量，则self是矩阵，只更新对应部分
                    self.grad = [[self.grad[i][j] + g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                    total = sum(sum(row) for row in g)
                    other.grad += total
            else:
                # 如果self是标量，直接更新梯度
                self.grad += g
                other.grad += g
        # 设置反向传播函数
        out._backward = _backward
        return out

    def __mul__(self, other):
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        # 简化：标量*矩阵 或 标量*标量
        if isinstance(self.data, list) and not isinstance(other.data, list):
            out_data = [[x * other.data for x in row] for row in self.data]
        elif not isinstance(self.data, list) and isinstance(other.data, list):
            out_data = [[x * self.data for x in row] for row in other.data]
        elif not isinstance(self.data, list) and not isinstance(other.data, list):
            out_data = self.data * other.data
        else:
            # 矩阵逐元素乘法
            if len(self.data) != len(other.data):
                raise ValueError("矩阵逐元素乘法要求两个矩阵行数相同")
            if len(self.data[0]) != len(other.data[0]):
                raise ValueError("矩阵逐元素乘法要求两个矩阵列数相同")
            out_data = [[self.data[i][j] * other.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))]

        # 创建新的Tensor对象
        out = Tensor(out_data, (self, other), '*')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 处理梯度更新
            if isinstance(self.grad, list):
                if not isinstance(other.data, list):
                    # self是矩阵，other是标量的情况
                    self.grad = [[self.grad[i][j] + other.data * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                    total = sum(self.data[i][j] * g[i][j] for i in range(len(self.data)) for j in range(len(self.data[0])))
                    other.grad += total
                elif not isinstance(self.data, list):
                    # self是标量，other是矩阵的情况
                    other.grad = [[other.grad[i][j] + self.data * g[i][j] for j in range(len(other.grad[0]))] for i in range(len(other.grad))]
                    total = sum(other.data[i][j] * g[i][j] for i in range(len(other.data)) for j in range(len(other.data[0])))
                    self.grad += total
                else:
                    # 两个都是矩阵的情况（逐元素乘法）
                    self.grad = [[self.grad[i][j] + other.data[i][j] * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                    other.grad = [[other.grad[i][j] + self.data[i][j] * g[i][j] for j in range(len(other.grad[0]))] for i in range(len(other.grad))]
            else:
                # 两个都是标量的情况
                self.grad += other.data * g
                other.grad += self.data * g
        # 设置反向传播函数
        out._backward = _backward
        return out

    def __pow__(self, power):
        # 验证幂次必须是数字
        assert isinstance(power, (int, float))
        # 计算幂运算结果
        if isinstance(self.data, list):
            out_data = [[x ** power for x in row] for row in self.data]
        else:
            out_data = self.data ** power
            
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), f'**{power}')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 计算局部导数并更新梯度
            if isinstance(self.data, list):
                # 对于矩阵情况，逐元素计算梯度
                local_grad = [[power * (self.data[i][j] ** (power - 1)) for j in range(len(self.data[0]))] for i in range(len(self.data))]
                self.grad = [[self.grad[i][j] + local_grad[i][j] * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
            else:
                # 对于标量情况，直接计算梯度
                local_grad = power * (self.data ** (power - 1))
                self.grad += local_grad * g
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
        if isinstance(other.data, (int, float)):
            if other.data == 0:
                raise ValueError("除数不能为零")
        elif isinstance(other.data, list):
            for row in other.data:
                if isinstance(row, list):
                    for x in row:
                        if x == 0:
                            raise ValueError("除数不能为零")
                else:
                    if row == 0:
                        raise ValueError("除数不能为零")
        return self * (other ** -1)

    def matmul(self, other):
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        # 获取两个张量的数据
        A, B = self.data, other.data
        
        # 验证输入类型
        if not (isinstance(A, list) and isinstance(B, list)):
            raise ValueError("矩阵乘法要求两个操作数都是列表形式的矩阵")
        
        # 简单处理向量转矩阵
        if A and not isinstance(A[0], list):
            A = [A]
        if B and not isinstance(B[0], list):
            B = [[x] for x in B]
        
        # 验证矩阵非空
        if not A or not B or not A[0] or not B[0]:
            raise ValueError("矩阵不能为空")
        
        # 获取矩阵维度信息
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        # 验证矩阵维度是否匹配
        if cols_A != rows_B:
            raise ValueError(f"矩阵维度不匹配: 第一个矩阵的列数({cols_A}) != 第二个矩阵的行数({rows_B})")

        # 执行矩阵乘法运算
        out_data = [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]
        # 创建新的Tensor对象
        out = Tensor(out_data, (self, other), '@')

        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 计算A的梯度：dA = g @ B.T
            B_T = [[B[j][i] for j in range(rows_B)] for i in range(cols_B)]
            dA = [[sum(g[i][k] * B_T[k][j] for k in range(cols_B)) for j in range(cols_A)] for i in range(rows_A)]
            
            # 计算B的梯度：dB = A.T @ g
            A_T = [[A[j][i] for j in range(rows_A)] for i in range(cols_A)]
            dB = [[sum(A_T[i][k] * g[k][j] for k in range(rows_A)) for j in range(cols_B)] for i in range(rows_B)]

            # 更新A的梯度
            if isinstance(self.grad, list) and isinstance(self.grad[0], list):
                for i in range(rows_A):
                    for j in range(cols_A): 
                        self.grad[i][j] += dA[i][j]
            # 更新B的梯度
            if isinstance(other.grad, list) and isinstance(other.grad[0], list):
                for i in range(rows_B):
                    for j in range(cols_B):
                        other.grad[i][j] += dB[i][j]
        # 设置反向传播函数
        out._backward = _backward
        return out

    def sum(self):
        # 计算张量所有元素的总和
        if isinstance(self.data, list):
            total = sum(sum(row) for row in self.data)
        else:
            total = self.data
        # 创建表示总和的新张量
        out = Tensor(total, (self,), 'sum')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 将梯度分配给原张量的所有元素
            if isinstance(self.grad, list):
                if isinstance(self.grad[0], list):
                    # 如果是二维矩阵，将梯度加到每个元素上
                    self.grad = [[self.grad[i][j] + g for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                else:
                    # 如果是一维数组，将梯度加到每个元素上
                    self.grad = [self.grad[i] + g for i in range(len(self.grad))]
            else:
                # 如果是标量，直接更新梯度
                self.grad += g
        # 设置反向传播函数
        out._backward = _backward
        return out

    def exp(self):
        # 计算自然指数函数e^x
        if isinstance(self.data, list):
            if not self.data:
                out_data = []
            elif isinstance(self.data[0], list):
                out_data = [[math.exp(x) for x in row] for row in self.data]
            else:
                out_data = [math.exp(x) for x in self.data]
        else:
            out_data = math.exp(self.data)
        # 创建表示指数函数结果的新张量
        out = Tensor(out_data, (self,), 'exp')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 指数函数的导数是其自身
            if isinstance(self.grad, list):
                if not self.data:
                    pass  # 空列表无需处理
                elif isinstance(self.data[0], list):
                    # 对于二维矩阵，应用链式法则
                    local = [[out.data[i][j] for j in range(len(out.data[0]))] for i in range(len(out.data))]
                    self.grad = [[self.grad[i][j] + local[i][j] * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                else:
                    # 对于一维数组，应用链式法则
                    local = [out.data[i] for i in range(len(out.data))]
                    self.grad = [self.grad[i] + local[i] * g[i] for i in range(len(self.grad))]
            else:
                # 对于标量，应用链式法则
                self.grad += out.data * g
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    def relu(self):
        """ReLU激活函数"""
        if isinstance(self.data, list):
            if not self.data:
                out_data = []
            elif isinstance(self.data[0], list):
                out_data = [[max(0, x) for x in row] for row in self.data]
            else:
                out_data = [max(0, x) for x in self.data]
        else:
            out_data = max(0, self.data)
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), 'relu')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # ReLU的导数是：x>0时为1，否则为0
            if isinstance(self.grad, list):
                if not self.data:
                    pass  # 空列表无需处理
                elif isinstance(self.data[0], list):
                    # 对于二维矩阵，应用链式法则
                    local = [[1.0 if self.data[i][j] > 0 else 0.0 for j in range(len(self.data[0]))] for i in range(len(self.data))]
                    self.grad = [[self.grad[i][j] + local[i][j] * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                else:
                    # 对于一维数组，应用链式法则
                    local = [1.0 if self.data[i] > 0 else 0.0 for i in range(len(self.data))]
                    self.grad = [self.grad[i] + local[i] * g[i] for i in range(len(self.grad))]
            else:
                # 对于标量，应用链式法则
                self.grad += (1.0 if self.data > 0 else 0.0) * g
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    def tanh(self):
        """tanh激活函数"""
        if isinstance(self.data, list):
            if not self.data:
                out_data = []
            elif isinstance(self.data[0], list):
                out_data = [[math.tanh(x) for x in row] for row in self.data]
            else:
                out_data = [math.tanh(x) for x in self.data]
        else:
            out_data = math.tanh(self.data)
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), 'tanh')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # tanh的导数是：1 - tanh^2(x)
            if isinstance(self.grad, list):
                if not self.data:
                    pass  # 空列表无需处理
                elif isinstance(self.data[0], list):
                    # 对于二维矩阵，应用链式法则
                    local = [[1.0 - out.data[i][j] ** 2 for j in range(len(out.data[0]))] for i in range(len(out.data))]
                    self.grad = [[self.grad[i][j] + local[i][j] * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                else:
                    # 对于一维数组，应用链式法则
                    local = [1.0 - out.data[i] ** 2 for i in range(len(out.data))]
                    self.grad = [self.grad[i] + local[i] * g[i] for i in range(len(self.grad))]
            else:
                # 对于标量，应用链式法则
                self.grad += (1.0 - out.data ** 2) * g
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    def sigmoid(self):
        """sigmoid激活函数"""
        def sigmoid_func(x):
            return 1.0 / (1.0 + math.exp(-x))
        
        if isinstance(self.data, list):
            if not self.data:
                out_data = []
            elif isinstance(self.data[0], list):
                out_data = [[sigmoid_func(x) for x in row] for row in self.data]
            else:
                out_data = [sigmoid_func(x) for x in self.data]
        else:
            out_data = sigmoid_func(self.data)
        # 创建新的Tensor对象
        out = Tensor(out_data, (self,), 'sigmoid')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # sigmoid的导数是：sigmoid(x) * (1 - sigmoid(x))
            if isinstance(self.grad, list):
                if not self.data:
                    pass  # 空列表无需处理
                elif isinstance(self.data[0], list):
                    # 对于二维矩阵，应用链式法则
                    local = [[out.data[i][j] * (1.0 - out.data[i][j]) for j in range(len(out.data[0]))] for i in range(len(out.data))]
                    self.grad = [[self.grad[i][j] + local[i][j] * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                else:
                    # 对于一维数组，应用链式法则
                    local = [out.data[i] * (1.0 - out.data[i]) for i in range(len(out.data))]
                    self.grad = [self.grad[i] + local[i] * g[i] for i in range(len(self.grad))]
            else:
                # 对于标量，应用链式法则
                self.grad += (out.data * (1.0 - out.data)) * g
        # 设置反向传播函数
        out._backward = _backward
        return out

    def log(self):
        # 计算自然对数函数ln(x)
        if isinstance(self.data, list):
            if not self.data:
                out_data = []
            elif isinstance(self.data[0], list):
                # 检查是否有非正数
                for row in self.data:
                    for x in row:
                        if x <= 0:
                            raise ValueError("对数函数要求输入值大于0")
                out_data = [[math.log(x) for x in row] for row in self.data]
            else:
                # 一维数组
                for x in self.data:
                    if x <= 0:
                        raise ValueError("对数函数要求输入值大于0")
                out_data = [math.log(x) for x in self.data]
        else:
            if self.data <= 0:
                raise ValueError("对数函数要求输入值大于0")
            out_data = math.log(self.data)
        # 创建表示对数函数结果的新张量
        out = Tensor(out_data, (self,), 'log')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 对数函数的导数是1/x
            if isinstance(self.grad, list):
                if not self.data:
                    pass  # 空列表无需处理
                elif isinstance(self.data[0], list):
                    # 对于二维矩阵，应用链式法则
                    local = [[1.0 / self.data[i][j] for j in range(len(self.data[0]))] for i in range(len(self.data))]
                    self.grad = [[self.grad[i][j] + local[i][j] * g[i][j] for j in range(len(self.grad[0]))] for i in range(len(self.grad))]
                else:
                    # 对于一维数组，应用链式法则
                    local = [1.0 / self.data[i] for i in range(len(self.data))]
                    self.grad = [self.grad[i] + local[i] * g[i] for i in range(len(self.grad))]
            else:
                # 对于标量，应用链式法则
                self.grad += (1.0 / self.data) * g
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
        # 确保输入是二维列表（batch, features）
        if isinstance(self.data, list) and self.data:
            if isinstance(self.data[0], list):
                # 二维输入 (batch, features)
                out_data = []
                for row in self.data:
                    # 数值稳定性：减去最大值
                    max_val = max(row)
                    exp_vals = [math.exp(x - max_val) for x in row]
                    sum_exp = sum(exp_vals)
                    out_data.append([e / sum_exp for e in exp_vals])
            else:
                # 一维输入 (features,)
                max_val = max(self.data)
                exp_vals = [math.exp(x - max_val) for x in self.data]
                sum_exp = sum(exp_vals)
                out_data = [e / sum_exp for e in exp_vals]
        else:
            raise ValueError("Softmax 要求输入为非空列表")
        
        out = Tensor(out_data, (self,), 'softmax')
        
        def _backward():
            """Softmax 反向传播
            
            softmax 的 Jacobian 矩阵为:
            J[i][j] = s[i] * (delta_ij - s[j])
            
            对于 batch 输入，每个样本独立计算
            """
            g = out.grad
            
            if isinstance(self.data, list) and isinstance(self.data[0], list):
                # 二维输入 (batch, features)
                for b in range(len(self.data)):
                    s = out.data[b]  # softmax 输出
                    n = len(s)
                    # 计算梯度: dL/dx = sum_j (dL/ds_j * ds_j/dx_i)
                    # ds_j/dx_i = s_j * (delta_ij - s_i)
                    for i in range(n):
                        grad_sum = 0.0
                        for j in range(n):
                            if i == j:
                                grad_sum += g[b][j] * s[j] * (1 - s[i])
                            else:
                                grad_sum += g[b][j] * (-s[j] * s[i])
                        self.grad[b][i] += grad_sum
            else:
                # 一维输入 (features,)
                s = out.data
                n = len(s)
                for i in range(n):
                    grad_sum = 0.0
                    for j in range(n):
                        if i == j:
                            grad_sum += g[j] * s[j] * (1 - s[i])
                        else:
                            grad_sum += g[j] * (-s[j] * s[i])
                    self.grad[i] += grad_sum
        
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
        batch_size = len(probs.data)
        num_classes = len(probs.data[0])
        
        # 计算每个样本的交叉熵
        losses = []
        for b in range(batch_size):
            sample_loss = 0.0
            for c in range(num_classes):
                # targets 是 one-hot，只有正确类别为 1
                p = probs.data[b][c]
                # clip 防止 log(0)
                p = max(epsilon, min(1 - epsilon, p))
                sample_loss -= targets.data[b][c] * math.log(p)
            losses.append(sample_loss)
        
        # 平均损失
        avg_loss = sum(losses) / batch_size
        
        # 创建损失 Tensor
        out = Tensor(avg_loss, (probs,), 'cross_entropy')
        
        def _backward():
            """交叉熵 + softmax 的反向传播
            
            组合导数: dL/dz = probs - targets
            这是因为 softmax + cross_entropy 有优美的简化形式
            """
            g = out.grad  # 标量梯度
            for b in range(batch_size):
                for c in range(num_classes):
                    # dL/dz = (probs - targets) * grad_from_upstream
                    probs.grad[b][c] += g * (probs.data[b][c] - targets.data[b][c]) / batch_size
        
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
        if isinstance(self.data, (int, float)):
            self.grad = 1.0
        elif isinstance(self.data, list):
             if isinstance(self.data[0], list):
                 self.grad = [[1.0 for _ in row] for row in self.data]
             else:
                 self.grad = [1.0 for _ in self.data]

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