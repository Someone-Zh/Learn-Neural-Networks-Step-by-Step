import numpy as np

class TensorBasic:
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
        out = self.__class__(out_data, (self, other), '@')

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

    def sum(self, axis=None, keepdims=False):
        # 使用NumPy计算总和
        total = np.sum(self.data, axis=axis, keepdims=keepdims)
        # 创建表示总和的新张量
        out = self.__class__(total, (self,), 'sum')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 将梯度分配给原张量的所有元素
            # 如果指定了axis，需要广播梯度到原始形状
            if axis is None:
                self.grad = self.grad + np.ones_like(self.data) * g
            else:
                # 计算需要广播的形状
                broadcast_shape = list(self.data.shape)
                if isinstance(axis, int):
                    broadcast_shape[axis] = 1
                else:
                    for ax in axis:
                        broadcast_shape[ax] = 1
                # 广播梯度
                broadcasted_g = np.broadcast_to(g, self.data.shape)
                self.grad = self.grad + broadcasted_g
        # 设置反向传播函数
        out._backward = _backward
        return out