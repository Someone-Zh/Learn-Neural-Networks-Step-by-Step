import numpy as np

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