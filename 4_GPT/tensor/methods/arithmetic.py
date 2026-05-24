import numpy as np
import torch
from ..utils import _unbroadcast

class TensorArithmetic:
    def __add__(self, other):
        # 确保other是Tensor类型
        other = self._ensure_tensor(other)
        
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None and other.torch_tensor is not None:
            result_torch = self.torch_tensor + other.torch_tensor
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self, other), '+')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            return out
        
        # 使用NumPy广播机制进行加法
        out_data = self.data + other.data
            
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self, other), '+')

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
        
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None and other.torch_tensor is not None:
            result_torch = self.torch_tensor * other.torch_tensor
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self, other), '*')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            return out
        
        # 使用NumPy进行逐元素乘法
        out_data = self.data * other.data

        # 创建新的Tensor对象
        out = self.__class__(out_data, (self, other), '*')

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
        
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None:
            result_torch = self.torch_tensor ** power
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self,), f'**{power}')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            return out
        
        # 使用NumPy计算幂运算结果
        out_data = self.data ** power
            
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self,), f'**{power}')

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
    def __neg__(self): 
        if self.use_pytorch and self.torch_tensor is not None:
            result_torch = -self.torch_tensor
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self,), 'neg')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            return out
        return self * -1
    
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
        
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None and other.torch_tensor is not None:
            if torch.any(other.torch_tensor == 0):
                raise ValueError("除数不能为零")
            result_torch = self.torch_tensor / other.torch_tensor
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self, other), '/')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            return out
        
        # 检查除数是否为零
        if np.any(other.data == 0):
            raise ValueError("除数不能为零")
        return self * (other ** -1)