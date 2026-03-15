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
            raise NotImplementedError("矩阵乘以矩阵需要使用.matmul()方法")

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
    def __sub__(self, other): return self + (-other)
    # 除法运算：通过乘法和负指数实现
    def __truediv__(self, other): return self * (other ** -1)

    def matmul(self, other):
        # 获取两个张量的数据
        A, B = self.data, other.data
        # 简单处理向量转矩阵
        if isinstance(A, list) and A and not isinstance(A[0], list): A = [A]
        if isinstance(B, list) and B and not isinstance(B[0], list): B = [[x] for x in B]
        
        # 获取矩阵维度信息
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        # 验证矩阵维度是否匹配
        assert cols_A == rows_B, f"矩阵维度不匹配 {cols_A} != {rows_B}"

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
                    for j in range(cols_A): self.grad[i][j] += dA[i][j]
            # 更新B的梯度
            if isinstance(other.grad, list) and isinstance(other.grad[0], list):
                for i in range(rows_B):
                    for j in range(cols_B): other.grad[i][j] += dB[i][j]
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
            out_data = [[math.exp(x) for x in row] for row in self.data]
        else:
            out_data = math.exp(self.data)
        # 创建表示指数函数结果的新张量
        out = Tensor(out_data, (self,), 'exp')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 指数函数的导数是其自身
            if isinstance(self.grad, list):
                if isinstance(self.data[0], list):
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

    def log(self):
        # 计算自然对数函数ln(x)
        if isinstance(self.data, list):
            out_data = [[math.log(x) for x in row] for row in self.data]
        else:
            out_data = math.log(self.data)
        # 创建表示对数函数结果的新张量
        out = Tensor(out_data, (self,), 'log')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 对数函数的导数是1/x
            if isinstance(self.grad, list):
                if isinstance(self.data[0], list):
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

    def backward(self):
        # 拓扑排序列表，用于按正确顺序执行反向传播
        topo = []
        # 已访问节点集合，避免重复访问
        visited = set()
        
        # 构建拓扑排序
        def build(v):
            if v not in visited:
                visited.add(v)
                # 递归访问前驱节点
                for child in v._prev: build(child)
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