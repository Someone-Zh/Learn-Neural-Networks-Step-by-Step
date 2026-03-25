import numpy as np

class TensorEmbedding:
    def embedding(self, indices):
        """Embedding层实现
        
        从嵌入矩阵中根据索引提取嵌入向量
        
        参数:
            indices: 索引张量，形状为 (*)
            
        返回:
            嵌入向量张量，形状为 (*, embedding_dim)
        """
        # 确保indices是Tensor类型
        indices = self._ensure_tensor(indices)
        
        # 验证输入形状
        if self.data.ndim != 2:
            raise ValueError("Embedding层要求权重矩阵为二维形状 (num_embeddings, embedding_dim)")
        
        # 提取嵌入向量
        out_data = self.data[indices.data.astype(int)]
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self, indices), 'embedding')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            # 创建权重矩阵的梯度
            dW = np.zeros_like(self.data)
            
            # 使用numpy的高级索引来累加梯度
            np.add.at(dW, indices.data.astype(int), g)
            
            # 更新权重矩阵的梯度
            self.grad = self.grad + dW
        # 设置反向传播函数
        out._backward = _backward
        return out

    @classmethod
    def create_embedding(cls, num_embeddings, embedding_dim):
        """创建Embedding权重矩阵
        
        参数:
            num_embeddings: 嵌入词汇表大小
            embedding_dim: 嵌入维度
            
        返回:
            初始化的嵌入权重矩阵
        """
        # 使用Xavier初始化
        scale = np.sqrt(1.0 / embedding_dim)
        weights = np.random.uniform(-scale, scale, (num_embeddings, embedding_dim))
        return cls(weights)

    def rotate_embedding(self, seq_len):
        """旋转位置编码
        
        参数:
            seq_len: 序列长度
            
        返回:
            应用旋转编码后的嵌入
        """
        # 获取形状信息
        batch_size, _, d_model = self.data.shape
        
        # 计算旋转矩阵
        theta = np.zeros((seq_len, d_model // 2))
        for i in range(seq_len):
            for j in range(d_model // 2):
                theta[i, j] = i / (10000 ** (2 * j / d_model))
        
        # 创建旋转矩阵
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 扩展到批次维度
        cos_theta = np.expand_dims(cos_theta, axis=0)  # (1, seq_len, d_model//2)
        sin_theta = np.expand_dims(sin_theta, axis=0)  # (1, seq_len, d_model//2)
        
        # 分割嵌入为偶数和奇数部分
        x_even = self.data[..., ::2]  # 偶数索引
        x_odd = self.data[..., 1::2]  # 奇数索引
        
        # 应用旋转
        x_rotated_even = x_even * cos_theta - x_odd * sin_theta
        x_rotated_odd = x_even * sin_theta + x_odd * cos_theta
        
        # 合并回原始形状
        out_data = np.stack([x_rotated_even, x_rotated_odd], axis=-1).reshape(batch_size, seq_len, d_model)
        
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self,), 'rotate_embedding')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            # 分割梯度为偶数和奇数部分
            g_even = g[..., ::2]
            g_odd = g[..., 1::2]
            
            # 反向传播旋转操作
            dx_even = g_even * cos_theta + g_odd * sin_theta
            dx_odd = -g_even * sin_theta + g_odd * cos_theta
            
            # 合并回原始形状
            dx = np.stack([dx_even, dx_odd], axis=-1).reshape(batch_size, seq_len, d_model)
            
            # 更新梯度
            self.grad = self.grad + dx
        # 设置反向传播函数
        out._backward = _backward
        return out
    
    @staticmethod
    def positional_encoding(seq_len, d_model, batch_size=1):
        """生成正弦余弦位置编码
        
        参数:
            seq_len: 序列长度
            d_model: 模型维度
            batch_size: 批次大小，默认为1
            
        返回:
            位置编码张量，形状为 (batch_size, seq_len, d_model)
        """
        # 创建位置索引
        position = np.arange(seq_len)[np.newaxis, :, np.newaxis]  # (1, seq_len, 1)
        
        # 创建维度索引
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model//2,)
        
        # 计算正弦和余弦
        pe = np.zeros((1, seq_len, d_model))
        pe[..., 0::2] = np.sin(position * div_term)
        pe[..., 1::2] = np.cos(position * div_term)
        
        # 扩展到批次维度
        pe = np.repeat(pe, batch_size, axis=0)
        
        # 转换为Tensor
        from ..core import Tensor
        return Tensor(pe)