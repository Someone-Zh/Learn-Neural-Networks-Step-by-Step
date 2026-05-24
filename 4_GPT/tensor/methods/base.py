import numpy as np
import torch

class TensorBase:
    def __init__(self, data, _prev=(), _op='', label=''):
        # 存储张量的实际数据，转换为NumPy数组
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        elif isinstance(data, (int, float)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, self.__class__):
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
        
        # PyTorch GPU 支持
        self.use_pytorch = False
        self.torch_tensor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
        
        # 如果使用 PyTorch 后端，也清零 PyTorch 梯度
        if self.use_pytorch and self.torch_tensor is not None:
            if self.torch_tensor.grad is not None:
                self.torch_tensor.grad.zero_()
        
        # 释放计算图
        if release_graph:
            self._prev = set()
            self._backward = lambda: None
            self._retain_graph = False

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"

    # 确保输入参数为Tensor类型，如果不是则进行转换
    def _ensure_tensor(self, other):
        return other if isinstance(other, self.__class__) else self.__class__(other)