import numpy as np
import torch

class TensorAttention:
    def multi_head_attention(self, q, k, v, num_heads, mask=None):
        """多头自注意力机制
        
        参数:
            q: 查询张量，形状为 (batch_size, seq_len, d_model)
            k: 键张量，形状为 (batch_size, seq_len, d_model)
            v: 值张量，形状为 (batch_size, seq_len, d_model)
            num_heads: 注意力头数
            mask: 注意力掩码，形状为 (batch_size, 1, seq_len, seq_len) 或 (batch_size, num_heads, seq_len, seq_len)
            
        返回:
            注意力输出，形状为 (batch_size, seq_len, d_model)
        """
        # 确保输入是Tensor类型
        q = self._ensure_tensor(q)
        k = self._ensure_tensor(k)
        v = self._ensure_tensor(v)
        
        # 获取形状信息
        batch_size, seq_len, d_model = q.data.shape
        d_k = d_model // num_heads
        
        # 验证维度
        if d_model % num_heads != 0:
            raise ValueError("d_model必须能被num_heads整除")
        
        # 如果使用 PyTorch 后端
        if self.use_pytorch and self.torch_tensor is not None and q.torch_tensor is not None and k.torch_tensor is not None and v.torch_tensor is not None:
            # 线性投影
            wq, wk, wv = torch.chunk(self.torch_tensor, 3, dim=1)
            
            # 计算Q, K, V
            q_proj = torch.matmul(q.torch_tensor, wq)
            k_proj = torch.matmul(k.torch_tensor, wk)
            v_proj = torch.matmul(v.torch_tensor, wv)
            
            # 重塑为多头
            q_reshaped = q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
            k_reshaped = k_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
            v_reshaped = v_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
            
            # 计算注意力分数
            attn_scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / np.sqrt(d_k)
            
            # 应用注意力掩码
            if mask is not None:
                mask_torch = mask.torch_tensor if hasattr(mask, 'torch_tensor') else torch.tensor(mask, device=self.device)
                attn_scores = attn_scores + mask_torch
            
            # 计算注意力权重
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            # 计算注意力输出
            attn_output = torch.matmul(attn_weights, v_reshaped)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
            
            out_data = attn_output.detach().cpu().numpy()
            out = self.__class__(out_data, (self, q, k, v), 'multi_head_attention')
            out.torch_tensor = attn_output
            out.use_pytorch = True
            out.device = self.device
            
            def _backward():
                if out.torch_tensor.grad is not None:
                    g = out.torch_tensor.grad
                    
                    # 反向传播
                    dv = torch.matmul(attn_weights.transpose(-2, -1), 
                                     g.reshape(batch_size, num_heads, seq_len, d_k).transpose(1, 2))
                    dv = dv.transpose(1, 2).reshape(batch_size, seq_len, d_model)
                    
                    d_attn_weights = torch.matmul(g.reshape(batch_size, num_heads, seq_len, d_k), 
                                                 v_reshaped.transpose(-2, -1))
                    
                    d_attn_scores = d_attn_weights * attn_weights - \
                                   d_attn_weights.sum(dim=-1, keepdim=True) * attn_weights
                    
                    dk = torch.matmul(q_reshaped.transpose(-2, -1), d_attn_scores) / np.sqrt(d_k)
                    dk = dk.transpose(1, 2).reshape(batch_size, seq_len, d_model)
                    
                    dq = torch.matmul(d_attn_scores, k_reshaped.transpose(-2, -1)) / np.sqrt(d_k)
                    dq = dq.transpose(1, 2).reshape(batch_size, seq_len, d_model)
                    
                    dwq = torch.matmul(q.torch_tensor.transpose(1, 2), dq).sum(dim=0)
                    dwk = torch.matmul(k.torch_tensor.transpose(1, 2), dk).sum(dim=0)
                    dwv = torch.matmul(v.torch_tensor.transpose(1, 2), dv).sum(dim=0)
                    dW = torch.cat([dwq, dwk, dwv], dim=1)
                    
                    if self.torch_tensor.grad is None:
                        self.torch_tensor.grad = torch.zeros_like(self.torch_tensor)
                    if q.torch_tensor.grad is None:
                        q.torch_tensor.grad = torch.zeros_like(q.torch_tensor)
                    if k.torch_tensor.grad is None:
                        k.torch_tensor.grad = torch.zeros_like(k.torch_tensor)
                    if v.torch_tensor.grad is None:
                        v.torch_tensor.grad = torch.zeros_like(v.torch_tensor)
                    
                    self.torch_tensor.grad += dW
                    q.torch_tensor.grad += dq
                    k.torch_tensor.grad += dk
                    v.torch_tensor.grad += dv
                    
                    # 同步数据
                    self.data = self.torch_tensor.detach().cpu().numpy()
                    q.data = q.torch_tensor.detach().cpu().numpy()
                    k.data = k.torch_tensor.detach().cpu().numpy()
                    v.data = v.torch_tensor.detach().cpu().numpy()
                    if self.torch_tensor.grad is not None:
                        self.grad = self.torch_tensor.grad.cpu().numpy()
                    if q.torch_tensor.grad is not None:
                        q.grad = q.torch_tensor.grad.cpu().numpy()
                    if k.torch_tensor.grad is not None:
                        k.grad = k.torch_tensor.grad.cpu().numpy()
                    if v.torch_tensor.grad is not None:
                        v.grad = v.torch_tensor.grad.cpu().numpy()
            
            out._backward = _backward
            return out
        
        # 线性投影（简化版，实际应该有可学习的权重）
        # 这里我们假设self是权重矩阵
        if self.data.shape != (d_model, d_model * 3):
            raise ValueError("权重矩阵形状应为 (d_model, d_model * 3)")
        
        # 分割权重
        wq, wk, wv = np.split(self.data, 3, axis=1)
        
        # 计算Q, K, V
        q_proj = np.matmul(q.data, wq)
        k_proj = np.matmul(k.data, wk)
        v_proj = np.matmul(v.data, wv)
        
        # 重塑为多头
        q_reshaped = q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        k_reshaped = k_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        v_reshaped = v_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        
        # 计算注意力分数
        attn_scores = np.matmul(q_reshaped, k_reshaped.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        # 应用注意力掩码
        if mask is not None:
            mask = mask.data if hasattr(mask, 'data') else mask
            attn_scores = attn_scores + mask
        
        # 计算注意力权重
        attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 计算注意力输出
        attn_output = np.matmul(attn_weights, v_reshaped)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # 创建新的Tensor对象
        out = self.__class__(attn_output, (self, q, k, v), 'multi_head_attention')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            # 反向传播
            # 计算v的梯度
            attn_weights_T = attn_weights.transpose(0, 1, 3, 2)
            dv = np.matmul(attn_weights_T, g.reshape(batch_size, num_heads, seq_len, d_k).transpose(0, 2, 1, 3))
            dv = dv.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
            
            # 计算注意力权重的梯度
            d_attn_weights = np.matmul(g.reshape(batch_size, num_heads, seq_len, d_k), v_reshaped.transpose(0, 1, 3, 2))
            
            # 计算注意力分数的梯度
            d_attn_scores = d_attn_weights * attn_weights - d_attn_weights.sum(axis=-1, keepdims=True) * attn_weights
            
            # 计算q的梯度
            k_reshaped_T = k_reshaped.transpose(0, 1, 3, 2)
            dq = np.matmul(d_attn_scores, k_reshaped_T) / np.sqrt(d_k)
            dq = dq.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
            
            # 计算k的梯度
            q_reshaped_T = q_reshaped.transpose(0, 1, 3, 2)
            dk = np.matmul(q_reshaped_T, d_attn_scores) / np.sqrt(d_k)
            dk = dk.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
            
            # 计算权重梯度
            dwq = np.matmul(q.data.transpose(0, 2, 1), dq).sum(axis=0)
            dwk = np.matmul(k.data.transpose(0, 2, 1), dk).sum(axis=0)
            dwv = np.matmul(v.data.transpose(0, 2, 1), dv).sum(axis=0)
            dW = np.concatenate([dwq, dwk, dwv], axis=1)
            
            # 更新梯度
            self.grad = self.grad + dW
            q.grad = q.grad + dq
            k.grad = k.grad + dk
            v.grad = v.grad + dv
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    @staticmethod
    def create_causal_mask(seq_len, batch_size=1):
        """创建因果注意力掩码
        
        参数:
            seq_len: 序列长度
            batch_size: 批次大小，默认为1
            
        返回:
            因果掩码，形状为 (batch_size, 1, seq_len, seq_len)
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(np.float64)
        mask = mask * -1e10  # 将上三角部分设置为负无穷
        mask = np.expand_dims(mask, axis=0)  # (1, seq_len, seq_len)
        mask = np.expand_dims(mask, axis=1)  # (1, 1, seq_len, seq_len)
        mask = np.repeat(mask, batch_size, axis=0)  # (batch_size, 1, seq_len, seq_len)
        
        # 转换为Tensor
        from ..core import Tensor
        return Tensor(mask)