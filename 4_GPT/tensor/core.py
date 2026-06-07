# 张量类实现自动微分功能 - PyTorch GPU 加速
import numpy as np
import torch
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

# 全局配置：设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 向后兼容：USE_PYTORCH_BACKEND 始终为 True
USE_PYTORCH_BACKEND = True


class Tensor:
    """张量类，支持自动微分和各种数学运算 - PyTorch 后端"""
    def __init__(self, data, _prev=(), _op='', label='', device=None):
        # 初始化基础功能（不存储data，由Tensor类管理torch_tensor）
        self.base = TensorBase(data, _prev, _op, label)
        
        # PyTorch GPU 支持
        self.use_pytorch = True
        self.device = device or DEVICE
        
        # 创建 torch tensor
        if isinstance(data, torch.Tensor):
            self.torch_tensor = data.to(self.device)
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if data.dtype in [np.int32, np.int64, np.int16, np.int8, np.float32, np.float64]:
                self.torch_tensor = torch.from_numpy(data.astype(np.float32)).to(self.device)
            else:
                self.torch_tensor = torch.from_numpy(data).to(self.device)
        
        # 只对浮点型张量设置 requires_grad（如果是非叶子节点，需要用 retain_grad）
        if self.torch_tensor.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            if self.torch_tensor.is_leaf:
                self.torch_tensor.requires_grad = True
            else:
                self.torch_tensor.retain_grad()
        
        # 初始化梯度
        self.grad = np.zeros(self.torch_tensor.shape, dtype=np.float64)
        
        # 组合各个功能模块
        self._arithmetic = TensorArithmetic()
        self._basic = TensorBasic()
        self._embedding = TensorEmbedding()
        self._normalization = TensorNormalization()
        self._attention = TensorAttention()
        self._feedforward = TensorFeedForward()
        self._activation = TensorActivation()
        self._loss = TensorLoss()
        self._backward_module = TensorBackward()
        
        # 代理属性
        self._backward = self.base._backward
        self._prev = self.base._prev
        self._op = self.base._op
        self.label = self.base.label
        self._retain_graph = self.base._retain_graph
    
    @property
    def data(self):
        """返回 torch_tensor 的 numpy 副本"""
        return self.torch_tensor.detach().cpu().numpy()
    
    @data.setter
    def data(self, value):
        """设置 torch_tensor（通过numpy数组）"""
        if isinstance(value, np.ndarray):
            self.torch_tensor = torch.from_numpy(value.astype(np.float32)).to(self.device)
        elif isinstance(value, torch.Tensor):
            self.torch_tensor = value.to(self.device)
        else:
            self.torch_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
    
    # 代理方法
    def _validate_grad_shape(self):
        return self.base._validate_grad_shape()
    
    def zero_grad(self, release_graph=True):
        self.grad = np.zeros(self.torch_tensor.shape, dtype=np.float64)
        if self.torch_tensor.grad is not None:
            self.torch_tensor.grad.zero_()
        if release_graph:
            self._prev = set()
            self._backward = lambda: None
            self._retain_graph = False
    
    def _ensure_tensor(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)
    
    # 算术运算
    def __add__(self, other):
        other = self._ensure_tensor(other)
        result_torch = self.torch_tensor + other.torch_tensor
        out = Tensor(result_torch, (self, other), '+')
        out.torch_tensor = result_torch
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                self.grad = self.grad + g
                other.grad = other.grad + g
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = self._ensure_tensor(other)
        result_torch = self.torch_tensor * other.torch_tensor
        out = Tensor(result_torch, (self, other), '*')
        out.torch_tensor = result_torch
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                self.grad = self.grad + g * other.data
                other.grad = other.grad + g * self.data
        
        out._backward = _backward
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        result_torch = self.torch_tensor ** power
        out = Tensor(result_torch, (self,), f'**{power}')
        out.torch_tensor = result_torch
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
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
        return self * (other ** -1)
    
    # 基本矩阵运算
    def matmul(self, other):
        """矩阵乘法，支持二维和三维批量操作"""
        other = self._ensure_tensor(other)
        result = torch.matmul(self.torch_tensor, other.torch_tensor)
        out = Tensor(result, (self, other), '@')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad
                dA = torch.matmul(g, other.torch_tensor.t())
                dB = torch.matmul(self.torch_tensor.t(), g)
                self.grad = self.grad + dA.cpu().numpy()
                other.grad = other.grad + dB.cpu().numpy()
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        result = torch.sum(self.torch_tensor, dim=axis, keepdim=keepdims)
        out = Tensor(result, (self,), 'sum')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
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
        
        if self.torch_tensor.dim() != 2:
            raise ValueError("Embedding层要求权重矩阵为二维形状 (num_embeddings, embedding_dim)")
        
        indices_long = indices.torch_tensor.long().reshape(-1)
        result = torch.index_select(self.torch_tensor, 0, indices_long)
        result = result.reshape(*indices.torch_tensor.shape, -1)
        
        out = Tensor(result, (self, indices), 'embedding')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad
                dW = torch.zeros_like(self.torch_tensor)
                indices_flat = indices_long.reshape(-1)
                g_flat = g.reshape(-1, g.shape[-1])
                dW.index_add_(0, indices_flat, g_flat)
                self.grad = self.grad + dW.cpu().numpy()
        
        out._backward = _backward
        return out
    
    @classmethod
    def create_embedding(cls, num_embeddings, embedding_dim):
        scale = np.sqrt(1.0 / embedding_dim)
        weights = torch.randn(num_embeddings, embedding_dim) * scale
        return cls(weights)
    
    def rotate_embedding(self, seq_len):
        batch_size, _, d_model = self.data.shape
        
        positions = torch.arange(seq_len, device=self.device).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device).float() * 
                            -(np.log(10000.0) / d_model))
        angles = positions.unsqueeze(1) * div_term.unsqueeze(0)
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)
        
        x_even = self.torch_tensor[..., ::2]
        x_odd = self.torch_tensor[..., 1::2]
        x_rotated_even = x_even * cos_vals - x_odd * sin_vals
        x_rotated_odd = x_even * sin_vals + x_odd * cos_vals
        result = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).reshape(batch_size, seq_len, d_model)
        
        out = Tensor(result, (self,), 'rotate_embedding')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad
                g_even = g[..., ::2]
                g_odd = g[..., 1::2]
                dx_even = g_even * cos_vals + g_odd * sin_vals
                dx_odd = -g_even * sin_vals + g_odd * cos_vals
                dx = torch.stack([dx_even, dx_odd], dim=-1).reshape(batch_size, seq_len, d_model)
                self.grad = self.grad + dx.cpu().numpy()
        
        out._backward = _backward
        return out
    
    # 归一化操作
    def layer_norm(self, eps=1e-5):
        result = torch.nn.functional.layer_norm(self.torch_tensor, (self.torch_tensor.shape[-1],), eps=eps)
        out = Tensor(result, (self,), 'layer_norm')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                mean = np.mean(self.data, axis=-1, keepdims=True)
                var = np.var(self.data, axis=-1, keepdims=True)
                std = np.sqrt(var + eps)
                dx = (1 / std) * (g - np.mean(g, axis=-1, keepdims=True) - 
                                 (self.data - mean) / (std ** 2) * np.mean(g * (self.data - mean), axis=-1, keepdims=True))
                self.grad = self.grad + dx
        
        out._backward = _backward
        return out
    
    def softmax(self):
        result = torch.softmax(self.torch_tensor, dim=-1)
        out = Tensor(result, (self,), 'softmax')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                out_data = out.data
                dx = out_data * (g - np.sum(g * out_data, axis=-1, keepdims=True))
                self.grad = self.grad + dx
        
        out._backward = _backward
        return out
    
    def dropout(self, p=0.5, training=True):
        if training:
            result = torch.nn.functional.dropout(self.torch_tensor, p=p, training=True)
        else:
            result = self.torch_tensor
        out = Tensor(result, (self,), 'dropout')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                self.grad = self.grad + g
        
        out._backward = _backward
        return out
    
    # 前馈网络
    def feed_forward(self, x, hidden_dim):
        x = self._ensure_tensor(x)
        batch_size, seq_len, d_model = x.data.shape
        
        if self.torch_tensor.shape != (d_model + hidden_dim, d_model):
            raise ValueError(f"权重矩阵形状应为 ({d_model + hidden_dim}, {d_model})")
        
        w1 = self.torch_tensor[:d_model, :]
        w2 = self.torch_tensor[d_model:, :]
        hidden = torch.relu(torch.matmul(x.torch_tensor, w1))
        result = torch.matmul(hidden, w2)
        
        out = Tensor(result, (self, x), 'feed_forward')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad
                dw2 = torch.matmul(hidden.transpose(1, 2), g).sum(dim=0)
                dhidden = torch.matmul(g, w2.t())
                dhidden = dhidden * (hidden > 0).float()
                dw1 = torch.matmul(x.torch_tensor.transpose(1, 2), dhidden).sum(dim=0)
                dx = torch.matmul(dhidden, w1.t())
                self.grad = self.grad + torch.cat([dw1, dw2], dim=0).cpu().numpy()
                x.grad = x.grad + dx.cpu().numpy()
        
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
        
        if self.torch_tensor.shape != (d_model, d_model * 3):
            raise ValueError("权重矩阵形状应为 (d_model, d_model * 3)")
        
        wq, wk, wv = torch.chunk(self.torch_tensor, 3, dim=1)
        q_proj = torch.matmul(q.torch_tensor, wq)
        k_proj = torch.matmul(k.torch_tensor, wk)
        v_proj = torch.matmul(v.torch_tensor, wv)
        
        q_reshaped = q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        k_reshaped = k_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        v_reshaped = v_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        
        attn_scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            mask_torch = mask.torch_tensor if hasattr(mask, 'torch_tensor') else torch.tensor(mask, device=self.device)
            attn_scores = attn_scores + mask_torch
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_reshaped)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        
        out = Tensor(attn_output, (self, q, k, v), 'multi_head_attention')
        out.torch_tensor = attn_output
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad
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
                
                self.grad = self.grad + dW.cpu().numpy()
                q.grad = q.grad + dq.cpu().numpy()
                k.grad = k.grad + dk.cpu().numpy()
                v.grad = v.grad + dv.cpu().numpy()
        
        out._backward = _backward
        return out
    
    # 激活函数
    def exp(self):
        result = torch.exp(self.torch_tensor)
        out = Tensor(result, (self,), 'exp')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        return out
    
    def relu(self):
        result = torch.relu(self.torch_tensor)
        out = Tensor(result, (self,), 'relu')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                local = (self.data > 0).astype(np.float64)
                self.grad = self.grad + local * g
        
        out._backward = _backward
        return out
    
    def silu(self):
        result = torch.nn.functional.silu(self.torch_tensor)
        out = Tensor(result, (self,), 'silu')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                sigmoid_x = 1.0 / (1.0 + np.exp(-self.data))
                local = sigmoid_x * (1.0 + self.data * (1.0 - sigmoid_x))
                self.grad = self.grad + local * g
        
        out._backward = _backward
        return out
    
    def tanh(self):
        result = torch.tanh(self.torch_tensor)
        out = Tensor(result, (self,), 'tanh')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                local = 1.0 - out.data ** 2
                self.grad = self.grad + local * g
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        result = torch.sigmoid(self.torch_tensor)
        out = Tensor(result, (self,), 'sigmoid')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                local = out.data * (1.0 - out.data)
                self.grad = self.grad + local * g
        
        out._backward = _backward
        return out
    
    def log(self):
        result = torch.log(self.torch_tensor)
        out = Tensor(result, (self,), 'log')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.cpu().numpy()
                self.grad = self.grad + (1.0 / self.data) * g
        
        out._backward = _backward
        return out
    
    # RMS 归一化
    def rms_norm(self, eps=1e-5):
        """RMS 归一化
        
        参数:
            eps: 防止除零的小值
            
        返回:
            归一化后的张量
        """
        variance = self.torch_tensor.pow(2).mean(-1, keepdim=True)
        normalized = self.torch_tensor * torch.rsqrt(variance + eps)
        out = Tensor(normalized, (self,), 'rms_norm')
        out.torch_tensor = normalized
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad
                N = self.torch_tensor.shape[-1]
                rms = torch.sqrt(variance + eps)
                dx = (1.0 / rms) * (g - self.torch_tensor / (rms ** 2) * torch.mean(g * self.torch_tensor, dim=-1, keepdim=True))
                self.grad = self.grad + dx.cpu().numpy()
        
        out._backward = _backward
        return out
    
    # 形状操作
    def view(self, *shape):
        """改变张量形状
        
        参数:
            shape: 新的形状
            
        返回:
            新形状的张量
        """
        result = self.torch_tensor.view(*shape)
        out = Tensor(result, (self,), 'view')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.view(self.torch_tensor.shape)
                self.grad = self.grad + g.cpu().numpy()
        
        out._backward = _backward
        return out
    
    def reshape(self, *shape):
        """改变张量形状
        
        参数:
            shape: 新的形状
            
        返回:
            新形状的张量
        """
        result = self.torch_tensor.reshape(*shape)
        out = Tensor(result, (self,), 'reshape')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.reshape(self.torch_tensor.shape)
                self.grad = self.grad + g.cpu().numpy()
        
        out._backward = _backward
        return out
    
    def transpose(self, dim0, dim1):
        """转置张量的两个维度
        
        参数:
            dim0: 第一个维度
            dim1: 第二个维度
            
        返回:
            转置后的张量
        """
        result = self.torch_tensor.transpose(dim0, dim1)
        out = Tensor(result, (self,), 'transpose')
        out.torch_tensor = result
        out.torch_tensor.retain_grad()
        
        def _backward():
            if out.torch_tensor.grad is not None:
                g = out.torch_tensor.grad.transpose(dim0, dim1)
                self.grad = self.grad + g.cpu().numpy()
        
        out._backward = _backward
        return out
    
    # 反向传播
    def backward(self, retain_graph=False):
        if self.torch_tensor.grad_fn is not None or self.torch_tensor.requires_grad:
            if self.torch_tensor.numel() == 1:
                self.torch_tensor.backward(retain_graph=retain_graph)
            else:
                gradient = torch.ones_like(self.torch_tensor)
                self.torch_tensor.backward(gradient=gradient, retain_graph=retain_graph)
            
            if self.torch_tensor.grad is not None:
                self.grad = self.torch_tensor.grad.cpu().numpy()
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self._op})"


class TransformerBlock:
    """Transformer块"""
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.attn_weights = Tensor(torch.randn(d_model, d_model * 3, device=DEVICE) * np.sqrt(1.0 / d_model))
        self.ffn_weights = Tensor(torch.randn(d_model + hidden_dim, d_model, device=DEVICE) * np.sqrt(1.0 / d_model))
        self.ln1_gamma = Tensor(torch.ones(d_model, device=DEVICE))
        self.ln1_beta = Tensor(torch.zeros(d_model, device=DEVICE))
        self.ln2_gamma = Tensor(torch.ones(d_model, device=DEVICE))
        self.ln2_beta = Tensor(torch.zeros(d_model, device=DEVICE))
    
    def forward(self, x, mask=None, training=True):
        attn_output = self.attn_weights.multi_head_attention(x, x, x, self.num_heads, mask)
        if training:
            attn_output = attn_output.dropout(self.dropout)
        x = x + attn_output
        x = x.layer_norm()
        
        ffn_output = self.ffn_weights.feed_forward(x, self.hidden_dim)
        if training:
            ffn_output = ffn_output.dropout(self.dropout)
        x = x + ffn_output
        x = x.layer_norm()
        return x
    
    def parameters(self):
        return [self.attn_weights, self.ffn_weights, self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta]


class GPT:
    """GPT模型"""
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = Tensor.create_embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        self.output_weights = Tensor(torch.randn(d_model, vocab_size, device=DEVICE) * np.sqrt(1.0 / d_model))
    
    def forward(self, x, training=True):
        x = self.embedding.embedding(x)
        batch_size, seq_len, _ = x.data.shape
        pe = TensorEmbedding.positional_encoding(seq_len, self.d_model, batch_size)
        x = x + pe
        if training:
            x = x.dropout(self.dropout)
        mask = TensorAttention.create_causal_mask(seq_len, batch_size)
        for block in self.blocks:
            x = block.forward(x, mask, training)
        output = x.matmul(self.output_weights)
        return output
    
    def parameters(self):
        params = [self.embedding, self.output_weights]
        for block in self.blocks:
            params.extend(block.parameters())
        return params
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


def enable_pytorch_backend(device=None):
    global DEVICE
    if device is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = device
    print(f"PyTorch backend enabled. Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def disable_pytorch_backend():
    """禁用 PyTorch 后端（已弃用 - 始终使用 PyTorch）"""
    print("PyTorch backend is always enabled now.")


def get_backend_info():
    info = {
        'backend': 'PyTorch',
        'device': DEVICE,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
    return info


def to_torch_tensor(tensor_obj):
    if isinstance(tensor_obj, torch.Tensor):
        return tensor_obj
    return torch.from_numpy(tensor_obj.data.astype(np.float32)).to(tensor_obj.device)


def from_torch_tensor(torch_tensor, requires_grad=True):
    result = Tensor(torch_tensor.detach().cpu().numpy())
    if requires_grad and torch_tensor.requires_grad:
        result.use_pytorch = True
        result.torch_tensor = torch_tensor.clone().requires_grad_(True)
    return result
