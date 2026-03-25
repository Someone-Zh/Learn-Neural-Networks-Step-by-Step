import numpy as np

class TensorActivation:
    def exp(self):
        # 使用NumPy计算自然指数函数e^x
        out_data = np.exp(self.data)
        # 创建表示指数函数结果的新张量
        out = self.__class__(out_data, (self,), 'exp')
        
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
        out = self.__class__(out_data, (self,), 'relu')
        
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
        out = self.__class__(out_data, (self,), 'tanh')
        
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
        out = self.__class__(out_data, (self,), 'sigmoid')
        
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
        out = self.__class__(out_data, (self,), 'log')
        
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
        
        out = self.__class__(out_data, (self,), 'softmax')
        
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