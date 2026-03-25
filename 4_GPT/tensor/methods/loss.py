import numpy as np

class TensorLoss:
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