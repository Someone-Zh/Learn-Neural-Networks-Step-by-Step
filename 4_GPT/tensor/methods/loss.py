import numpy as np
import torch

class TensorLoss:
    def cross_entropy_loss(self, targets):
        """交叉熵损失函数
        
        参数:
            targets: 真实标签的 one-hot 编码 Tensor，形状为 (Batch, num_classes)
        
        返回:
            平均交叉熵损失（标量 Tensor）
        
        公式: L = -1/N * sum(sum(targets * log(softmax_output)))
        """
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None and targets.torch_tensor is not None:
            # 先计算 softmax
            probs_torch = torch.softmax(self.torch_tensor, dim=-1)
            
            # 计算交叉熵
            epsilon = 1e-15
            batch_size = probs_torch.shape[0]
            
            # clip 防止 log(0)
            probs_clipped = torch.clamp(probs_torch, epsilon, 1 - epsilon)
            
            # 计算每个样本的交叉熵
            log_probs = torch.log(probs_clipped)
            sample_losses = -torch.sum(targets.torch_tensor * log_probs, dim=1)
            
            # 平均损失
            avg_loss = torch.mean(sample_losses)
            
            # 创建损失 Tensor
            out_data = avg_loss.detach().cpu().numpy()
            out = self.__class__(out_data, (self,), 'cross_entropy')
            out.torch_tensor = avg_loss
            out.use_pytorch = True
            out.device = self.device
            
            def _backward():
                if out.torch_tensor.grad is not None:
                    g = out.torch_tensor.grad
                    # dL/dz = (probs - targets) * grad_from_upstream / batch_size
                    if self.torch_tensor.grad is None:
                        self.torch_tensor.grad = torch.zeros_like(self.torch_tensor)
                    self.torch_tensor.grad += g * (probs_torch - targets.torch_tensor) / batch_size
                    
                    # 同步数据
                    self.data = self.torch_tensor.detach().cpu().numpy()
                    if self.torch_tensor.grad is not None:
                        self.grad = self.torch_tensor.grad.cpu().numpy()
            
            out._backward = _backward
            return out
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
        out = self.__class__(avg_loss, (probs,), 'cross_entropy')
        
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