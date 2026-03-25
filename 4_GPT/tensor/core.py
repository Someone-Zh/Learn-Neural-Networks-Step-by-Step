# 张量类实现自动微分功能 - NumPy加速版本
import numpy as np
from .methods import (
    TensorBase,
    TensorArithmetic,
    TensorBasic,
    TensorEmbedding,
    TensorNormalization,
    TensorAttention,
    TensorFeedForward,
    TensorActivation,
    TensorLoss,
    TensorBackward
)

class Tensor:
    """张量类，支持自动微分和各种数学运算
    
    采用组合设计模式，集成了多个功能模块：
    - TensorBase: 基础功能，如构造函数、梯度验证等
    - TensorArithmetic: 算术运算，如加减乘除等
    - TensorBasic: 基本矩阵运算，如矩阵乘法、求和等
    - TensorEmbedding: 嵌入操作，如嵌入查找、旋转编码等
    - TensorNormalization: 归一化操作，如层归一化、softmax等
    - TensorAttention: 注意力机制，如多头自注意力等
    - TensorFeedForward: 前馈网络和混合专家模式
    - TensorActivation: 激活函数，如ReLU、tanh、sigmoid等
    - TensorLoss: 损失函数，如交叉熵损失
    - TensorBackward: 反向传播功能
    """
    def __init__(self, data, _prev=(), _op='', label=''):
        # 初始化基础功能
        self.base = TensorBase(data, _prev, _op, label)
        
        # 组合各个功能模块
        self.arithmetic = TensorArithmetic()
        self.basic = TensorBasic()
        self.embedding = TensorEmbedding()
        self.normalization = TensorNormalization()
        self.attention = TensorAttention()
        self.feedforward = TensorFeedForward()
        self.activation = TensorActivation()
        self.loss = TensorLoss()
        self.backward_module = TensorBackward()
        
        # 代理属性
        self.data = self.base.data
        self.grad = self.base.grad
        self._backward = self.base._backward
        self._prev = self.base._prev
        self._op = self.base._op
        self.label = self.base.label
        self._retain_graph = self.base._retain_graph
    
    # 代理方法
    def _validate_grad_shape(self):
        return self.base._validate_grad_shape()
    
    def zero_grad(self, release_graph=True):
        return self.base.zero_grad(release_graph)
    
    def _ensure_tensor(self, other):
        return other if isinstance(other, self.__class__) else self.__class__(other)
    
    # 算术运算
    def __add__(self, other):
        other = self._ensure_tensor(other)
        out_data = self.data + other.data
        out = self.__class__(out_data, (self, other), '+')
        
        def _backward():
            g = out.grad
            from ..utils import _unbroadcast
            self.grad = self.grad + _unbroadcast(g, self.data.shape)
            other.grad = other.grad + _unbroadcast(g, other.data.shape)
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = self._ensure_tensor(other)
        out_data = self.data * other.data
        out = self.__class__(out_data, (self, other), '*')
        
        def _backward():
            g = out.grad
            from ..utils import _unbroadcast
            self.grad = self.grad + _unbroadcast(g * other.data, self.data.shape)
            other.grad = other.grad + _unbroadcast(g * self.data, other.data.shape)
        
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out_data = self.data ** power
        out = self.__class__(out_data, (self,), f'**{power}')
        
        def _backward():
            g = out.grad
            local_grad = power * (self.data ** (power - 1))
            self.grad = self.grad + local_grad * g
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        other = self._ensure_tensor(other)
        return self + (-other)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __rtruediv__(self, other):
        return self._ensure_tensor(other) * (self ** -1)
    
    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        if np.any(other.data == 0):
            raise ValueError("除数不能为零")
        return self * (other ** -1)
    
    # 基本矩阵运算
    def matmul(self, other):
        other = self._ensure_tensor(other)
        A, B = self.data, other.data
        
        if A.ndim < 1 or B.ndim < 1:
            raise ValueError("矩阵乘法要求两个操作数都是数组")
        
        if A.ndim == 1:
            A = A.reshape(1, -1)
        if B.ndim == 1:
            B = B.reshape(-1, 1)
        
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"矩阵维度不匹配: 第一个矩阵的列数({A.shape[1]}) != 第二个矩阵的行数({B.shape[0]})")

        out_data = np.matmul(A, B)
        out = self.__class__(out_data, (self, other), '@')

        def _backward():
            g = out.grad
            orig_A_shape = A.shape
            orig_B_shape = B.shape
            
            dA = np.matmul(g, B.T)
            dB = np.matmul(A.T, g)

            if self.data.ndim == 1:
                dA = dA.flatten()
            if other.data.ndim == 1:
                dB = dB.flatten()
            
            self.grad = self.grad + dA
            other.grad = other.grad + dB
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        total = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = self.__class__(total, (self,), 'sum')
        
        def _backward():
            g = out.grad
            if axis is None:
                self.grad = self.grad + np.ones_like(self.data) * g
            else:
                broadcast_shape = list(self.data.shape)
                if isinstance(axis, int):
                    broadcast_shape[axis] = 1
                else:
                    for ax in axis:
                        broadcast_shape[ax] = 1
                broadcasted_g = np.broadcast_to(g, self.data.shape)
                self.grad = self.grad + broadcasted_g
        
        out._backward = _backward
        return out
    
    # 嵌入操作
    def embedding(self, indices):
        indices = self._ensure_tensor(indices)
        
        if self.data.ndim != 2:
            raise ValueError("Embedding层要求权重矩阵为二维形状 (num_embeddings, embedding_dim)")
        
        out_data = self.data[indices.data.astype(int)]
        out = self.__class__(out_data, (self, indices), 'embedding')
        
        def _backward():
            g = out.grad
            dW = np.zeros_like(self.data)
            np.add.at(dW, indices.data.astype(int), g)
            self.grad = self.grad + dW
        
        out._backward = _backward
        return out
    
    @classmethod
    def create_embedding(cls, num_embeddings, embedding_dim):
        scale = np.sqrt(1.0 / embedding_dim)
        weights = np.random.uniform(-scale, scale, (num_embeddings, embedding_dim))
        return cls(weights)
    
    def rotate_embedding(self, seq_len):
        batch_size, _, d_model = self.data.shape
        
        theta = np.zeros((seq_len, d_model // 2))
        for i in range(seq_len):
            for j in range(d_model // 2):
                theta[i, j] = i / (10000 ** (2 * j / d_model))
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        cos_theta = np.expand_dims(cos_theta, axis=0)
        sin_theta = np.expand_dims(sin_theta, axis=0)
        
        x_even = self.data[..., ::2]
        x_odd = self.data[..., 1::2]
        
        x_rotated_even = x_even * cos_theta - x_odd * sin_theta
        x_rotated_odd = x_even * sin_theta + x_odd * cos_theta
        
        out_data = np.stack([x_rotated_even, x_rotated_odd], axis=-1).reshape(batch_size, seq_len, d_model)
        
        out = self.__class__(out_data, (self,), 'rotate_embedding')
        
        def _backward():
            g = out.grad
            g_even = g[..., ::2]
            g_odd = g[..., 1::2]
            
            dx_even = g_even * cos_theta + g_odd * sin_theta
            dx_odd = -g_even * sin_theta + g_odd * cos_theta
            
            dx = np.stack([dx_even, dx_odd], axis=-1).reshape(batch_size, seq_len, d_model)
            self.grad = self.grad + dx
        
        out._backward = _backward
        return out
    
    # 归一化操作
    def layer_norm(self, eps=1e-5):
        mean = np.mean(self.data, axis=-1, keepdims=True)
        var = np.var(self.data, axis=-1, keepdims=True)
        
        out_data = (self.data - mean) / np.sqrt(var + eps)
        
        out = self.__class__(out_data, (self,), 'layer_norm')
        
        def _backward():
            g = out.grad
            N = self.data.shape[-1]
            std = np.sqrt(var + eps)
            
            dx = (1 / std) * (g - np.mean(g, axis=-1, keepdims=True) - 
                             (self.data - mean) / (std ** 2) * np.mean(g * (self.data - mean), axis=-1, keepdims=True))
            
            self.grad = self.grad + dx
        
        out._backward = _backward
        return out
    
    def softmax(self):
        exp_data = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_data = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
        
        out = self.__class__(out_data, (self,), 'softmax')
        
        def _backward():
            g = out.grad
            out_data = out.data
            dx = out_data * (g - np.sum(g * out_data, axis=-1, keepdims=True))
            
            self.grad = self.grad + dx
        
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
    
    def dropout(self, p=0.5, training=True):
        """Dropout层
        
        参数:
            p:  dropout概率，默认为0.5
            training: 是否处于训练模式，默认为True
            
        返回:
            应用dropout后的张量
        """
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
    
    # 前馈网络
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
    
    # 注意力机制
    def multi_head_attention(self, q, k, v, num_heads, mask=None):

        q = self._ensure_tensor(q)
        k = self._ensure_tensor(k)
        v = self._ensure_tensor(v)
        
        batch_size, seq_len, d_model = q.data.shape
        d_k = d_model // num_heads
        
        if d_model % num_heads != 0:
            raise ValueError("d_model必须能被num_heads整除")
        
        if self.data.shape != (d_model, d_model * 3):
            raise ValueError("权重矩阵形状应为 (d_model, d_model * 3)")
        
        wq, wk, wv = np.split(self.data, 3, axis=1)
        
        q_proj = np.matmul(q.data, wq)
        k_proj = np.matmul(k.data, wk)
        v_proj = np.matmul(v.data, wv)
        
        q_reshaped = q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        k_reshaped = k_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        v_reshaped = v_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        
        attn_scores = np.matmul(q_reshaped, k_reshaped.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        attn_weights = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        attn_output = np.matmul(attn_weights, v_reshaped)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        out = self.__class__(attn_output, (self, q, k, v), 'multi_head_attention')
        
        def _backward():
            g = out.grad
            
            attn_weights_T = attn_weights.transpose(0, 1, 3, 2)
            dv = np.matmul(attn_weights_T, g.reshape(batch_size, num_heads, seq_len, d_k).transpose(0, 2, 1, 3))
            dv = dv.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
            
            q_reshaped_T = q_reshaped.transpose(0, 1, 3, 2)
            dk = np.matmul(q_reshaped_T, g.reshape(batch_size, num_heads, seq_len, d_k))
            dk = dk.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
            
            k_reshaped_T = k_reshaped.transpose(0, 1, 3, 2)
            dq = np.matmul(g.reshape(batch_size, num_heads, seq_len, d_k), k_reshaped_T)
            dq = dq.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
            
            dwq = np.matmul(q.data.transpose(0, 2, 1), dq).sum(axis=0)
            dwk = np.matmul(k.data.transpose(0, 2, 1), dk).sum(axis=0)
            dwv = np.matmul(v.data.transpose(0, 2, 1), dv).sum(axis=0)
            dW = np.concatenate([dwq, dwk, dwv], axis=1)
            
            self.grad = self.grad + dW
            q.grad = q.grad + dq
            k.grad = k.grad + dk
            v.grad = v.grad + dv
        
        out._backward = _backward
        return out
    
    # 激活函数
    def exp(self):
        out_data = np.exp(self.data)
        out = self.__class__(out_data, (self,), 'exp')
        
        def _backward():
            g = out.grad
            self.grad = self.grad + out.data * g
        
        out._backward = _backward
        return out
    
    def relu(self):
        out_data = np.maximum(0, self.data)
        out = self.__class__(out_data, (self,), 'relu')
        
        def _backward():
            g = out.grad
            local = (self.data > 0).astype(np.float64)
            self.grad = self.grad + local * g
        
        out._backward = _backward
        return out
    
    def tanh(self):
        out_data = np.tanh(self.data)
        out = self.__class__(out_data, (self,), 'tanh')
        
        def _backward():
            g = out.grad
            local = 1.0 - out.data ** 2
            self.grad = self.grad + local * g
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out_data = 1.0 / (1.0 + np.exp(-self.data))
        out = self.__class__(out_data, (self,), 'sigmoid')
        
        def _backward():
            g = out.grad
            local = out.data * (1.0 - out.data)
            self.grad = self.grad + local * g
        
        out._backward = _backward
        return out
    
    def log(self):
        if np.any(self.data <= 0):
            raise ValueError("对数函数要求输入值大于0")
        out_data = np.log(self.data)
        out = self.__class__(out_data, (self,), 'log')
        
        def _backward():
            g = out.grad
            self.grad = self.grad + (1.0 / self.data) * g
        
        out._backward = _backward
        return out
    
    # 反向传播
    def backward(self, retain_graph=False):
        topo = []
        visited = set()
        
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        
        self.grad = np.ones_like(self.data, dtype=np.float64)

        for node in reversed(topo):
            node._backward()
            try:
                node._validate_grad_shape()
            except ValueError as e:
                raise ValueError(
                    f"反向传播时梯度形状验证失败 (节点: {node._op}, 标签: {node.label}):\n{e}"
                )
        
        if not retain_graph:
            for node in topo:
                node._prev = set()
                node._backward = lambda: None
                node._retain_graph = False
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"

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

class GPT:
    """GPT模型
    
    包含嵌入层、多个Transformer块和输出层
    """
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout=0.1):
        """
        初始化GPT模型
        
        参数:
            vocab_size: 词汇表大小
            d_model: 模型维度
            num_heads: 注意力头数
            hidden_dim: 前馈网络隐藏层维度
            num_layers: Transformer块数量
            dropout: dropout概率
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 嵌入层
        self.embedding = Tensor.create_embedding(vocab_size, d_model)
        
        # Transformer块
        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append(TransformerBlock(d_model, num_heads, hidden_dim, dropout))
        
        # 输出层
        self.output_weights = Tensor(np.random.randn(d_model, vocab_size) * np.sqrt(1.0 / d_model))
    
    def forward(self, x, training=True):
        """
        前向传播
        
        参数:
            x: 输入张量，形状为 (batch_size, seq_len)
            training: 是否处于训练模式
            
        返回:
            输出张量，形状为 (batch_size, seq_len, vocab_size)
        """
        # 嵌入
        x = self.embedding.embedding(x)
        
        # 添加位置编码
        batch_size, seq_len, _ = x.data.shape
        pe = TensorEmbedding.positional_encoding(seq_len, self.d_model, batch_size)
        x = x + pe
        
        # Dropout
        if training:
            x = x.dropout(self.dropout)
        
        # 创建因果掩码
        mask = TensorAttention.create_causal_mask(seq_len, batch_size)
        
        # 经过多个Transformer块
        for block in self.blocks:
            x = block.forward(x, mask, training)
        
        # 输出层
        output = x.matmul(self.output_weights)
        
        return output
    
    def parameters(self):
        """
        获取所有可训练参数
        
        返回:
            参数列表
        """
        params = [self.embedding, self.output_weights]
        for block in self.blocks:
            params.extend(block.parameters())
        return params
    
    def save(self, path):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """
        加载模型
        
        参数:
            path: 加载路径
            
        返回:
            GPT模型实例
        """
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)