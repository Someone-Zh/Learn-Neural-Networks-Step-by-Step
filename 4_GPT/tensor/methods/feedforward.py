import numpy as np

class TensorFeedForward:
    def feed_forward(self, x, hidden_dim):
        """前馈神经网络
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            hidden_dim: 隐藏层维度
            
        返回:
            前馈网络输出，形状为 (batch_size, seq_len, d_model)
        """
        # 确保输入是Tensor类型
        x = self._ensure_tensor(x)
        
        # 获取形状信息
        batch_size, seq_len, d_model = x.data.shape
        
        # 验证权重矩阵形状
        if self.data.shape != (d_model + hidden_dim, d_model):
            raise ValueError(f"权重矩阵形状应为 ({d_model + hidden_dim}, {d_model})")
        
        # 分割权重
        w1 = self.data[:d_model, :]
        w2 = self.data[d_model:, :]
        
        # 前向传播
        hidden = np.matmul(x.data, w1)
        hidden = np.maximum(0, hidden)  # ReLU激活
        out_data = np.matmul(hidden, w2)
        
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self, x), 'feed_forward')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            # 反向传播
            # 计算w2的梯度
            dw2 = np.matmul(hidden.transpose(0, 2, 1), g).sum(axis=0)
            
            # 计算隐藏层的梯度
            dhidden = np.matmul(g, w2.T)
            dhidden = dhidden * (hidden > 0)  # ReLU导数
            
            # 计算w1的梯度
            dw1 = np.matmul(x.data.transpose(0, 2, 1), dhidden).sum(axis=0)
            
            # 计算x的梯度
            dx = np.matmul(dhidden, w1.T)
            
            # 更新梯度
            self.grad = self.grad + np.concatenate([dw1, dw2], axis=0)
            x.grad = x.grad + dx
        
        out._backward = _backward
        return out

class TransformerBlock:
    """Transformer块
    
    包含多头自注意力、层归一化和前馈网络
    """
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        """
        初始化Transformer块
        
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            hidden_dim: 前馈网络隐藏层维度
            dropout: dropout概率
        """
        from ..core import Tensor
        
        # 初始化权重
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 注意力权重
        self.attn_weights = Tensor(np.random.randn(d_model, d_model * 3) * np.sqrt(1.0 / d_model))
        
        # 前馈网络权重
        self.ffn_weights = Tensor(np.random.randn(d_model + hidden_dim, d_model) * np.sqrt(1.0 / d_model))
        
        # 层归一化参数
        self.ln1_gamma = Tensor(np.ones(d_model))
        self.ln1_beta = Tensor(np.zeros(d_model))
        self.ln2_gamma = Tensor(np.ones(d_model))
        self.ln2_beta = Tensor(np.zeros(d_model))
    
    def forward(self, x, mask=None, training=True):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
            mask: 注意力掩码
            training: 是否处于训练模式
            
        返回:
            输出张量，形状为 (batch_size, seq_len, d_model)
        """
        from ..core import Tensor
        
        # 多头自注意力
        attn_output = self.attn_weights.multi_head_attention(x, x, x, self.num_heads, mask)
        
        # Dropout
        if training:
            attn_output = attn_output.dropout(self.dropout)
        
        # 残差连接和层归一化
        x = x + attn_output
        x = x.layer_norm()
        
        # 前馈网络
        ffn_output = self.ffn_weights.feed_forward(x, self.hidden_dim)
        
        # Dropout
        if training:
            ffn_output = ffn_output.dropout(self.dropout)
        
        # 残差连接和层归一化
        x = x + ffn_output
        x = x.layer_norm()
        
        return x
    
    def parameters(self):
        """
        获取所有可训练参数
        
        返回:
            参数列表
        """
        return [
            self.attn_weights,
            self.ffn_weights,
            self.ln1_gamma,
            self.ln1_beta,
            self.ln2_gamma,
            self.ln2_beta
        ]
    def feed_forward(self, hidden_dim):
        """前馈神经网络
        
        参数:
            hidden_dim: 隐藏层维度
            
        返回:
            前馈网络输出
        """
        # 获取形状信息
        batch_size, seq_len, d_model = self.data.shape
        
        # 验证权重矩阵形状
        if self.data.shape != (d_model + hidden_dim, hidden_dim + d_model):
            raise ValueError("权重矩阵形状应为 (d_model + hidden_dim, hidden_dim + d_model)")
        
        # 分割权重
        w1 = self.data[:d_model, :hidden_dim]
        w2 = self.data[hidden_dim:, hidden_dim:]
        
        # 前向传播
        hidden = np.maximum(0, np.matmul(self.data, w1))  # ReLU激活
        out_data = np.matmul(hidden, w2)
        
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self,), 'feed_forward')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            # 反向传播
            dhidden = np.matmul(g, w2.T) * (hidden > 0)  # ReLU的导数
            dx = np.matmul(dhidden, w1.T)
            
            # 计算权重梯度
            dw2 = np.matmul(hidden.transpose(0, 2, 1), g).sum(axis=0)
            dw1 = np.matmul(self.data.transpose(0, 2, 1), dhidden).sum(axis=0)
            dW = np.zeros_like(self.data)
            dW[:d_model, :hidden_dim] = dw1
            dW[hidden_dim:, hidden_dim:] = dw2
            
            # 更新梯度
            self.grad = self.grad + dW
            self.grad = self.grad + dx
        # 设置反向传播函数
        out._backward = _backward
        return out

    def moe_forward(self, expert_weights_list, gate_weights):
        """混合专家模式前向传播
        
        参数:
            expert_weights_list: 专家网络权重列表
            gate_weights: 门控网络权重
            
        返回:
            混合专家模式输出
        """
        # 确保输入是Tensor类型
        gate_weights = self._ensure_tensor(gate_weights)
        
        # 获取形状信息
        batch_size, seq_len, d_model = self.data.shape
        num_experts = len(expert_weights_list)
        
        # 计算门控分数
        gate_logits = np.matmul(self.data, gate_weights.data)
        gate_probs = np.exp(gate_logits - np.max(gate_logits, axis=-1, keepdims=True))
        gate_probs = gate_probs / np.sum(gate_probs, axis=-1, keepdims=True)
        
        # 计算每个专家的输出
        expert_outputs = []
        for expert_weights in expert_weights_list:
            # 确保专家权重是Tensor类型
            expert_weights = self._ensure_tensor(expert_weights)
            # 前向传播
            hidden = np.maximum(0, np.matmul(self.data, expert_weights.data[:d_model, :]))
            output = np.matmul(hidden, expert_weights.data[d_model:, :])
            expert_outputs.append(output)
        
        # 加权组合专家输出
        out_data = np.zeros_like(self.data)
        for i, expert_output in enumerate(expert_outputs):
            out_data += gate_probs[..., i:i+1] * expert_output
        
        # 创建新的Tensor对象
        out = self.__class__(out_data, (self, gate_weights) + tuple(expert_weights_list), 'moe_forward')
        
        def _backward():
            # 获取输出张量的梯度
            g = out.grad
            
            # 计算门控网络梯度
            dgate = np.zeros_like(gate_probs)
            for i, expert_output in enumerate(expert_outputs):
                dgate[..., i] = np.sum(g * expert_output, axis=-1)
            dgate = gate_probs * (dgate - np.sum(gate_probs * dgate, axis=-1, keepdims=True))
            dgate_weights = np.matmul(self.data.transpose(0, 2, 1), dgate).sum(axis=0)
            
            # 计算专家网络梯度
            dexpert_weights_list = []
            for i, expert_weights in enumerate(expert_weights_list):
                # 计算专家输出的梯度
                dexpert = g * gate_probs[..., i:i+1]
                # 计算隐藏层梯度
                dhidden = np.matmul(dexpert, expert_weights.data[d_model:, :].T) * (np.maximum(0, np.matmul(self.data, expert_weights.data[:d_model, :])) > 0)
                # 计算专家权重梯度
                dw1 = np.matmul(self.data.transpose(0, 2, 1), dhidden).sum(axis=0)
                dw2 = np.matmul(np.maximum(0, np.matmul(self.data, expert_weights.data[:d_model, :])).transpose(0, 2, 1), dexpert).sum(axis=0)
                dexpert_weights = np.zeros_like(expert_weights.data)
                dexpert_weights[:d_model, :] = dw1
                dexpert_weights[d_model:, :] = dw2
                dexpert_weights_list.append(dexpert_weights)
            
            # 计算输入梯度
            dx = np.matmul(dgate, gate_weights.data.T)
            for i, expert_weights in enumerate(expert_weights_list):
                dhidden = np.matmul(g * gate_probs[..., i:i+1], expert_weights.data[d_model:, :].T) * (np.maximum(0, np.matmul(self.data, expert_weights.data[:d_model, :])) > 0)
                dx += np.matmul(dhidden, expert_weights.data[:d_model, :].T)
            
            # 更新梯度
            self.grad = self.grad + dx
            gate_weights.grad = gate_weights.grad + dgate_weights
            for i, expert_weights in enumerate(expert_weights_list):
                expert_weights.grad = expert_weights.grad + dexpert_weights_list[i]
        # 设置反向传播函数
        out._backward = _backward
        return out

    @classmethod
    def create_transformer_block(cls, d_model, num_heads, hidden_dim):
        """创建Transformer Block的权重
        
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            hidden_dim: 前馈网络隐藏层维度
            
        返回:
            包含所有权重的字典
        """
        # 初始化注意力权重
        attn_weights = np.random.randn(d_model, d_model * 3) * np.sqrt(2.0 / d_model)
        
        # 初始化前馈网络权重
        ff_weights = np.random.randn(d_model + hidden_dim, hidden_dim + d_model) * np.sqrt(2.0 / d_model)
        
        return {
            'attn_weights': cls(attn_weights),
            'ff_weights': cls(ff_weights)
        }

    @classmethod
    def create_moe_block(cls, d_model, hidden_dim, num_experts):
        """创建混合专家模式的权重
        
        参数:
            d_model: 模型维度
            hidden_dim: 专家网络隐藏层维度
            num_experts: 专家数量
            
        返回:
            包含门控网络权重和专家网络权重的字典
        """
        # 初始化门控网络权重
        gate_weights = np.random.randn(d_model, num_experts) * np.sqrt(2.0 / d_model)
        
        # 初始化专家网络权重
        expert_weights_list = []
        for _ in range(num_experts):
            expert_weights = np.random.randn(d_model + hidden_dim, hidden_dim + d_model) * np.sqrt(2.0 / d_model)
            expert_weights_list.append(cls(expert_weights))
        
        return {
            'gate_weights': cls(gate_weights),
            'expert_weights_list': expert_weights_list
        }