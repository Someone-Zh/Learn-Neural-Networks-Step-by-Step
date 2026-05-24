import numpy as np
import torch

class TensorNormalization:
    def layer_norm(self, eps=1e-5):
        """层归一化
        
        参数:
            eps: 防止除零的小值
            
        返回:
            归一化后的张量
        """
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None:
            # 使用PyTorch的layer_norm
            normalized_shape = self.torch_tensor.shape[-1:]
            result_torch = torch.nn.functional.layer_norm(
                self.torch_tensor, 
                normalized_shape, 
                eps=eps
            )
            
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self,), 'layer_norm')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            
            def _backward():
                if out.torch_tensor.grad is not None:
                    g = out.torch_tensor.grad
                    
                    # 计算梯度
                    N = self.torch_tensor.shape[-1]
                    mean = torch.mean(self.torch_tensor, dim=-1, keepdim=True)
                    var = torch.var(self.torch_tensor, dim=-1, keepdim=True, unbiased=False)
                    std = torch.sqrt(var + eps)
                    
                    # 计算dl/dx
                    dx = (1 / std) * (g - torch.mean(g, dim=-1, keepdim=True) - 
                                     (self.torch_tensor - mean) / (std ** 2) * 
                                     torch.mean(g * (self.torch_tensor - mean), dim=-1, keepdim=True))
                    
                    # 更新梯度
                    if self.torch_tensor.grad is None:
                        self.torch_tensor.grad = torch.zeros_like(self.torch_tensor)
                    self.torch_tensor.grad += dx
                    
                    # 同步数据
                    self.data = self.torch_tensor.detach().cpu().numpy()
                    if self.torch_tensor.grad is not None:
                        self.grad = self.torch_tensor.grad.cpu().numpy()
            
            out._backward = _backward
            return out
        
        # 计算均值和方差
        mean = np.mean(self.data, axis=-1, keepdims=True)
        var = np.var(self.data, axis=-1, keepdims=True)
        
        # 归一化
        out_data = (self.data - mean) / np.sqrt(var + eps)
        
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self,), 'layer_norm')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            # 计算梯度
            N = self.data.shape[-1]
            std = np.sqrt(var + eps)
            
            # 计算dl/dx
            dx = (1 / std) * (g - np.mean(g, axis=-1, keepdims=True) - 
                             (self.data - mean) / (std ** 2) * np.mean(g * (self.data - mean), axis=-1, keepdims=True))
            
            # 更新梯度
            self.grad = self.grad + dx
        # 设置反向传播函数
        out._backward = _backward
        return out

    def softmax(self):
        """Softmax激活函数
        
        返回:
            经过softmax处理的张量
        """
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None:
            # 使用PyTorch的softmax
            result_torch = torch.softmax(self.torch_tensor, dim=-1)
            
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self,), 'softmax')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            
            def _backward():
                if out.torch_tensor.grad is not None:
                    g = out.torch_tensor.grad
                    
                    # 计算梯度
                    out_data_torch = out.torch_tensor
                    dx = out_data_torch * (g - torch.sum(g * out_data_torch, dim=-1, keepdim=True))
                    
                    # 更新梯度
                    if self.torch_tensor.grad is None:
                        self.torch_tensor.grad = torch.zeros_like(self.torch_tensor)
                    self.torch_tensor.grad += dx
                    
                    # 同步数据
                    self.data = self.torch_tensor.detach().cpu().numpy()
                    if self.torch_tensor.grad is not None:
                        self.grad = self.torch_tensor.grad.cpu().numpy()
            
            out._backward = _backward
            return out
        
        # 数值稳定的softmax实现
        exp_data = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_data = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self,), 'softmax')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            # 计算梯度
            out_data = out.data
            dx = out_data * (g - np.sum(g * out_data, axis=-1, keepdims=True))
            
            # 更新梯度
            self.grad = self.grad + dx
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    def dropout(self, p=0.5, training=True):
        """Dropout层
        
        参数:
            p:  dropout概率，默认为0.5
            training: 是否处于训练模式，默认为True
            
        返回:
            应用dropout后的张量
        """
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None:
            if training:
                # 使用PyTorch的dropout
                result_torch = torch.nn.functional.dropout(self.torch_tensor, p=p, training=True)
            else:
                result_torch = self.torch_tensor
            
            out_data = result_torch.detach().cpu().numpy()
            out = self.__class__(out_data, (self,), 'dropout')
            out.torch_tensor = result_torch
            out.use_pytorch = True
            out.device = self.device
            
            def _backward():
                if out.torch_tensor.grad is not None:
                    g = out.torch_tensor.grad
                    
                    # PyTorch autograd会自动处理dropout的反向传播
                    if self.torch_tensor.grad is None:
                        self.torch_tensor.grad = torch.zeros_like(self.torch_tensor)
                    self.torch_tensor.grad += g
                    
                    # 同步数据
                    self.data = self.torch_tensor.detach().cpu().numpy()
                    if self.torch_tensor.grad is not None:
                        self.grad = self.torch_tensor.grad.cpu().numpy()
            
            out._backward = _backward
            return out
        
        if training:
            # 创建dropout掩码
            mask = np.random.binomial(1, 1 - p, size=self.data.shape).astype(np.float64)
            # 应用掩码并缩放
            out_data = self.data * mask / (1 - p)
        else:
            # 测试模式下不应用dropout
            out_data = self.data
        
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self,), 'dropout')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            if training:
                # 应用相同的掩码到梯度
                dx = g * mask / (1 - p)
            else:
                dx = g
            
            # 更新梯度
            self.grad = self.grad + dx
        # 设置反向传播函数
        out._backward = _backward
        return out