import numpy as np
import torch

class TensorBasic:
    def matmul(self, other):
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None and other.torch_tensor is not None:
            A = self.torch_tensor
            B = other.torch_tensor
            
            if A.dim() < 1 or B.dim() < 1:
                raise ValueError("矩阵乘法要求两个操作数都是数组")
            
            # 处理一维数组
            orig_A_shape = A.shape
            orig_B_shape = B.shape
            
            if A.dim() == 1:
                A = A.reshape(1, -1)
            if B.dim() == 1:
                B = B.reshape(-1, 1)
            
            # 执行矩阵乘法
            result = torch.matmul(A, B)
            
            out_data = result.detach().cpu().numpy()
            out = self.__class__(out_data, (self, other), '@')
            out.torch_tensor = result
            out.use_pytorch = True
            out.device = self.device
            
            def _backward():
                if out.torch_tensor.grad is not None:
                    g = out.torch_tensor.grad
                    
                    # 计算梯度
                    if A.dim() == 3 and B.dim() == 2:
                        dA = torch.matmul(g, B.t())
                        dB = torch.matmul(A.transpose(1, 2), g).sum(dim=0)
                    elif A.dim() == 2 and B.dim() == 2:
                        dA = torch.matmul(g, B.t())
                        dB = torch.matmul(A.t(), g)
                    elif A.dim() == 3 and B.dim() == 3:
                        dA = torch.matmul(g, B.transpose(1, 2))
                        dB = torch.matmul(A.transpose(1, 2), g).sum(dim=0)
                    else:
                        dA = torch.matmul(g, B.t())
                        dB = torch.matmul(A.t(), g)
                    
                    # 更新梯度
                    if self.torch_tensor.grad is None:
                        self.torch_tensor.grad = torch.zeros_like(self.torch_tensor)
                    if other.torch_tensor.grad is None:
                        other.torch_tensor.grad = torch.zeros_like(other.torch_tensor)
                    
                    # 恢复原始形状
                    if len(orig_A_shape) == 1:
                        dA = dA.flatten()
                    if len(orig_B_shape) == 1:
                        dB = dB.flatten()
                    
                    self.torch_tensor.grad += dA
                    other.torch_tensor.grad += dB
                    
                    # 同步数据
                    self.data = self.torch_tensor.detach().cpu().numpy()
                    other.data = other.torch_tensor.detach().cpu().numpy()
                    if self.torch_tensor.grad is not None:
                        self.grad = self.torch_tensor.grad.cpu().numpy()
                    if other.torch_tensor.grad is not None:
                        other.grad = other.torch_tensor.grad.cpu().numpy()
            
            out._backward = _backward
            return out
        
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
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None:
            # 转换axis参数
            if axis is None:
                result_torch = torch.sum(self.torch_tensor)
            else:
                result_torch = torch.sum(self.torch_tensor, dim=axis, keepdim=keepdims)
            
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self,), 'sum')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            
            def _backward():
                if out.torch_tensor.grad is not None:
                    g = out.torch_tensor.grad
                    
                    # 广播梯度到原始形状
                    if axis is None:
                        expanded_g = g.expand_as(self.torch_tensor)
                    else:
                        # 需要手动广播
                        shape = list(self.torch_tensor.shape)
                        if isinstance(axis, int):
                            if keepdims:
                                shape[axis] = 1
                            else:
                                shape.pop(axis)
                        else:
                            for ax in sorted(axis, reverse=True):
                                if keepdims:
                                    shape[ax] = 1
                                else:
                                    shape.pop(ax)
                        
                        # 重塑梯度以匹配原始形状
                        g_reshaped = g.reshape(shape)
                        expanded_g = g_reshaped.expand_as(self.torch_tensor)
                    
                    if self.torch_tensor.grad is None:
                        self.torch_tensor.grad = torch.zeros_like(self.torch_tensor)
                    self.torch_tensor.grad += expanded_g
                    
                    # 同步数据
                    self.data = self.torch_tensor.detach().cpu().numpy()
                    if self.torch_tensor.grad is not None:
                        self.grad = self.torch_tensor.grad.cpu().numpy()
            
            out._backward = _backward
            return out
        
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