import torch
import torch.nn.functional as F
from tensor.core import Tensor, USE_PYTORCH_BACKEND, DEVICE
from tensor.methods import TensorAttention
from optim.optimizer import Adam, ExponentialLR

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig:
    def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", int(hidden_size * 4))
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 4)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                 MiniMind Small Config (快速验证用)
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindSmallConfig:
    """小参数量模型配置，用于本地快速验证训练"""
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 64)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 2)
        self.use_moe = kwargs.get("use_moe", False)
        self.dropout = kwargs.get("dropout", 0.1)
        self.vocab_size = kwargs.get("vocab_size", 1000)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", False)
        self.num_attention_heads = kwargs.get("num_attention_heads", 2)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')
        self.intermediate_size = kwargs.get("intermediate_size", 256)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 256)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 10000)
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = None
        ### MoE specific configs (ignored if use_moe = False)
        self.num_experts = kwargs.get("num_experts", 2)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        # 使用 torch.ones 创建权重
        self.weight = Tensor(torch.ones(dim, device=DEVICE))

    def norm(self, x):
        # 使用 PyTorch 操作
        if isinstance(x, Tensor) and x.use_pytorch:
            variance = x.torch_tensor.pow(2).mean(-1, keepdim=True)
            normalized = x.torch_tensor * torch.rsqrt(variance + self.eps)
            return Tensor(normalized)
        else:
            # NumPy 后备方案
            rms = torch.sqrt(torch.mean(x.data ** 2, dim=-1, keepdim=True) + self.eps)
            normalized = Tensor(x.data / rms.cpu().numpy())
            return normalized

    def forward(self, x):
        normalized = self.norm(x)
        return self.weight * normalized

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    import math
    
    # 使用 torch 创建频率
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE)[: (dim // 2)] / dim))
    
    attn_factor = 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)
        
        if end / orig_max > 1.0:
            def inv_dim(b):
                return (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32, device=DEVICE) - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    
    t = torch.arange(end, dtype=torch.float32, device=DEVICE)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1) * attn_factor
    freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1) * attn_factor
    
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    # 扩展维度
    if cos.dim() == 3:
        cos = cos.unsqueeze(2)  # (batch, seq, 1, head_dim)
        sin = sin.unsqueeze(2)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(
        bs, slen, num_key_value_heads * n_rep, head_dim
    )

class Attention:
    def __init__(self, config: MiniMindConfig):
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        
        # 使用 torch 初始化权重
        self.q_proj = Tensor(torch.randn(config.hidden_size, config.num_attention_heads * self.head_dim, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.k_proj = Tensor(torch.randn(config.hidden_size, self.num_key_value_heads * self.head_dim, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.v_proj = Tensor(torch.randn(config.hidden_size, self.num_key_value_heads * self.head_dim, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.o_proj = Tensor(torch.randn(config.num_attention_heads * self.head_dim, config.hidden_size, device=DEVICE) * (1.0 / (config.num_attention_heads * self.head_dim))**0.5)
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dropout = config.dropout

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.torch_tensor.shape if x.use_pytorch else x.data.shape
        
        # 线性投影 - 使用 torch_tensor
        if x.use_pytorch:
            xq_torch = torch.matmul(x.torch_tensor, self.q_proj.torch_tensor)
            xk_torch = torch.matmul(x.torch_tensor, self.k_proj.torch_tensor)
            xv_torch = torch.matmul(x.torch_tensor, self.v_proj.torch_tensor)
            
            # 重塑
            xq_torch = xq_torch.view(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk_torch = xk_torch.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv_torch = xv_torch.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            
            # 转换为 Tensor 包装
            xq = Tensor(xq_torch)
            xk = Tensor(xk_torch)
            xv = Tensor(xv_torch)
        else:
            # NumPy 后备方案
            xq = x.matmul(self.q_proj)
            xk = x.matmul(self.k_proj)
            xv = x.matmul(self.v_proj)
            
            xq_data = xq.data.reshape(bsz, seq_len, self.n_local_heads, self.head_dim)
            xk_data = xk.data.reshape(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            xv_data = xv.data.reshape(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
            
            xq = Tensor(xq_data)
            xk = Tensor(xk_data)
            xv = Tensor(xv_data)
        
        # 层归一化
        xq = self.q_norm.forward(xq)
        xk = self.k_norm.forward(xk)
        
        # 应用旋转位置编码
        cos, sin = position_embeddings
        if x.use_pytorch:
            xq_torch, xk_torch = apply_rotary_pos_emb(xq.torch_tensor, xk.torch_tensor, cos, sin)
            xq = Tensor(xq_torch)
            xk = Tensor(xk_torch)
        else:
            xq_data, xk_data = apply_rotary_pos_emb(xq.data, xk.data, cos.cpu().numpy(), sin.cpu().numpy())
            xq = Tensor(xq_data)
            xk = Tensor(xk_data)
        
        # 处理past_key_value
        if past_key_value is not None:
            if x.use_pytorch:
                xk_torch = torch.cat([past_key_value[0], xk_torch], dim=1)
                xv_torch = torch.cat([past_key_value[1], xv_torch], dim=1)
                xk = Tensor(xk_torch)
                xv = Tensor(xv_torch)
            else:
                xk_data = np.concatenate([past_key_value[0], xk_data], axis=1)
                xv_data = np.concatenate([past_key_value[1], xv_data], axis=1)
                xk = Tensor(xk_data)
                xv = Tensor(xv_data)
        
        past_kv = (xk_torch, xv_torch) if use_cache and x.use_pytorch else (xk_data, xv_data) if use_cache else None
        
        # 转置维度并计算注意力
        if x.use_pytorch:
            xq_torch = xq.torch_tensor.transpose(1, 2)  # (bsz, heads, seq, head_dim)
            xk_torch = repeat_kv(xk.torch_tensor, self.n_rep).transpose(1, 2)
            xv_torch = repeat_kv(xv.torch_tensor, self.n_rep).transpose(1, 2)
            
            # 计算注意力分数
            scores = torch.matmul(xq_torch, xk_torch.transpose(-2, -1)) / (self.head_dim ** 0.5)
            
            # 应用因果掩码
            if self.is_causal:
                seq_len_kv = xk_torch.shape[2]
                causal_mask = torch.triu(torch.ones((seq_len, seq_len_kv), device=DEVICE), diagonal=1).bool()
                scores[:, :, -seq_len:, :].masked_fill_(causal_mask[:seq_len, :seq_len_kv], float('-inf'))
            
            # 应用注意力掩码
            if attention_mask is not None:
                scores += (1.0 - attention_mask.torch_tensor[:, None, None, :]) * -1e9
            
            # 计算注意力权重
            attn_weights = F.softmax(scores, dim=-1)
            
            # 应用dropout
            if self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training if hasattr(self, 'training') else False)
            
            # 计算注意力输出
            attn_output = torch.matmul(attn_weights, xv_torch)
            attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
            attn_output = Tensor(attn_output)
        else:
            # NumPy 后备方案
            xq_data = xq.data.transpose(0, 2, 1, 3)
            xk_data = repeat_kv(xk.data, self.n_rep).transpose(0, 2, 1, 3)
            xv_data = repeat_kv(xv.data, self.n_rep).transpose(0, 2, 1, 3)
            
            scores = np.matmul(xq_data, xk_data.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
            
            if self.is_causal:
                seq_len_kv = xk_data.shape[2]
                causal_mask = np.triu(np.ones((seq_len, seq_len_kv)), k=1).astype(float) * -1e10
                scores[:, :, -seq_len:, :] += causal_mask
            
            if attention_mask is not None:
                scores += (1.0 - attention_mask.data[:, np.newaxis, np.newaxis, :]) * -1e9
            
            attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
            
            if self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, size=attn_weights.shape).astype(float)
                attn_weights = attn_weights * mask / (1 - self.dropout)
            
            attn_output = np.matmul(attn_weights, xv_data)
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, seq_len, -1)
            attn_output = Tensor(attn_output)
        
        # 输出投影
        output = attn_output.matmul(self.o_proj)
        
        # 应用dropout
        if self.dropout > 0:
            output = output.dropout(self.dropout)
        
        return output, past_kv

class FeedForward:
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = Tensor(torch.randn(config.hidden_size, intermediate_size, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.down_proj = Tensor(torch.randn(intermediate_size, config.hidden_size, device=DEVICE) * (1.0 / intermediate_size)**0.5)
        self.up_proj = Tensor(torch.randn(config.hidden_size, intermediate_size, device=DEVICE) * (1.0 / config.hidden_size)**0.5)

    def forward(self, x):
        # 门控机制 - 使用 SiLU 激活函数
        gate = x.matmul(self.gate_proj).silu()
        up = x.matmul(self.up_proj)
        hidden = gate * up
        output = hidden.matmul(self.down_proj)
        return output

class MOEFeedForward:
    def __init__(self, config: MiniMindConfig):
        self.config = config
        self.gate = Tensor(torch.randn(config.hidden_size, config.num_experts, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.experts = [FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.torch_tensor.shape if x.use_pytorch else x.data.shape
        
        if x.use_pytorch:
            x_flat = x.torch_tensor.reshape(-1, hidden_dim)
            x_flat_tensor = Tensor(x_flat)
            
            # 计算门控分数
            scores = x_flat_tensor.matmul(self.gate).softmax()
            
            # 选择topk专家 - 使用 torch.topk
            topk_weight, topk_idx = torch.topk(scores.torch_tensor, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
            
            # 归一化topk概率
            if self.config.norm_topk_prob:
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            
            # 初始化输出
            y = torch.zeros_like(x_flat)
            
            # 专家前向传播
            for i, expert in enumerate(self.experts):
                mask = (topk_idx == i)
                if mask.any():
                    token_idx = torch.where(mask.any(dim=-1))[0]
                    weight = topk_weight[mask].reshape(-1, 1)
                    expert_input = Tensor(x_flat[token_idx])
                    expert_output = expert.forward(expert_input).torch_tensor
                    y[token_idx] += (expert_output * weight).squeeze()
            
            # 计算辅助损失
            self.aux_loss = 0
            if self.config.router_aux_loss_coef > 0:
                load = torch.mean(torch.eye(self.config.num_experts, device=DEVICE)[topk_idx].float(), dim=0)
                self.aux_loss = torch.sum(load * torch.mean(scores.torch_tensor, dim=0)) * self.config.num_experts * self.config.router_aux_loss_coef
            
            return Tensor(y.reshape(batch_size, seq_len, hidden_dim))
        else:
            # NumPy 后备方案
            x_flat = x.data.reshape(-1, hidden_dim)
            x_flat_tensor = Tensor(x_flat)
            
            scores = x_flat_tensor.matmul(self.gate).softmax()
            topk_weight, topk_idx = np.topk(scores.data, k=self.config.num_experts_per_tok, axis=-1, sorted=False)
            
            if self.config.norm_topk_prob:
                topk_weight = topk_weight / (topk_weight.sum(axis=-1, keepdims=True) + 1e-20)
            
            y = np.zeros_like(x_flat)
            
            for i, expert in enumerate(self.experts):
                mask = (topk_idx == i)
                if mask.any():
                    token_idx = np.where(mask.any(axis=-1))[0]
                    weight = topk_weight[mask].reshape(-1, 1)
                    expert_input = Tensor(x_flat[token_idx])
                    expert_output = expert.forward(expert_input).data
                    y[token_idx] += (expert_output * weight).squeeze()
            
            self.aux_loss = 0
            if self.config.router_aux_loss_coef > 0:
                load = np.mean(np.eye(self.config.num_experts)[topk_idx].astype(float), axis=0)
                self.aux_loss = np.sum(load * np.mean(scores.data, axis=0)) * self.config.num_experts * self.config.router_aux_loss_coef
            
            return Tensor(y.reshape(batch_size, seq_len, hidden_dim))

class MiniMindBlock:
    def __init__(self, layer_id: int, config: MiniMindConfig):
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        hidden_states, present_key_value = self.self_attn.forward(
            hidden_states, position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value

class MiniMindModel:
    def __init__(self, config: MiniMindConfig):
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = Tensor.create_embedding(config.vocab_size, config.hidden_size)
        self.dropout = config.dropout
        self.layers = [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # 预计算频率
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        batch_size, seq_length = input_ids.shape
        
        # 转换 input_ids 为 torch tensor（如果不是）
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
        
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # 嵌入
        hidden_states = self.embed_tokens.embedding(Tensor(input_ids))
        
        # 应用dropout
        if self.dropout > 0:
            hidden_states = hidden_states.dropout(self.dropout)
        
        # 位置编码 - 已经是 torch tensors
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )
        
        # 扩展到批次维度 - 使用 torch 操作
        position_embeddings = (
            position_embeddings[0].unsqueeze(0).expand(batch_size, -1, -1),
            position_embeddings[1].unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer.forward(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        hidden_states = self.norm.forward(hidden_states)
        
        # 计算辅助损失
        aux_loss = 0
        for l in self.layers:
            if hasattr(l.mlp, 'aux_loss'):
                aux_loss += l.mlp.aux_loss
        
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM:
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        self.model = MiniMindModel(self.config)
        # 使用 torch 初始化
        self.lm_head = Tensor(torch.randn(self.config.hidden_size, self.config.vocab_size, device=DEVICE) * (1.0 / self.config.hidden_size)**0.5)
        # 共享权重
        if self.model.embed_tokens.use_pytorch:
            self.model.embed_tokens.torch_tensor = self.lm_head.torch_tensor.T
        else:
            self.model.embed_tokens.data = self.lm_head.data.T

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None):
        # 转换 input_ids 和 labels
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
        
        hidden_states, past_key_values, aux_loss = self.model.forward(input_ids, attention_mask, past_key_values, use_cache)
        
        # 处理logits_to_keep
        if logits_to_keep > 0:
            hidden_states = Tensor(hidden_states.torch_tensor[:, -logits_to_keep:])
        
        # 计算logits
        logits = hidden_states.matmul(self.lm_head)
        
        loss = None
        if labels is not None:
            batch_size, seq_len, vocab_size = logits.torch_tensor.shape
            
            # 展平
            logits_flat = logits.torch_tensor.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            
            # 创建mask
            mask = (labels_flat != -100) & (labels_flat != 0)
            if mask.any():
                valid_logits = logits_flat[mask]
                valid_labels = labels_flat[mask]
                
                # 使用 PyTorch 交叉熵
                loss_value = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
                
                # 创建 Tensor 包装损失
                loss = Tensor(loss_value)
        
        return {
            'loss': loss,
            'aux_loss': aux_loss,
            'logits': logits,
            'past_key_values': past_key_values,
            'hidden_states': hidden_states
        }
    
    def generate(self, input_ids, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2):
        # 确保是 torch tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
        else:
            input_ids = input_ids.to(DEVICE)
        
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids, use_cache=True)
            logits = outputs['logits'].torch_tensor[:, -1, :]
            
            # 应用温度
            logits = logits / temperature
            
            # 应用top-k
            if top_k > 0:
                top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                min_top_k = top_k_values[:, -1:].unsqueeze(-1)
                logits = torch.where(logits < min_top_k, torch.full_like(logits, float('-inf')), logits)
            
            # 应用top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[:, 0] = False
                sorted_indices_to_remove = mask.scatter(-1, sorted_indices, mask)
                logits = logits.masked_fill(sorted_indices_to_remove, float('-inf'))
            
            # 采样 - 使用 multinomial
            probs = F.softmax(logits, dim=-1)
            # multinomial 需要 2D 输入 (batch, vocab) 或 1D (vocab)
            if probs.dim() > 2:
                probs = probs.squeeze(0)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 追加到输入
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查是否生成eos_token
            if next_token.item() == eos_token_id:
                break
        
        return input_ids

    def parameters(self):
        """
        获取所有可训练参数
        """
        params = [self.lm_head]
        params.append(self.model.embed_tokens)
        params.append(self.model.norm.weight)
        
        for layer in self.model.layers:
            params.append(layer.input_layernorm.weight)
            params.append(layer.post_attention_layernorm.weight)
            params.append(layer.self_attn.q_proj)
            params.append(layer.self_attn.k_proj)
            params.append(layer.self_attn.v_proj)
            params.append(layer.self_attn.o_proj)
            params.append(layer.self_attn.q_norm.weight)
            params.append(layer.self_attn.k_norm.weight)
            
            if isinstance(layer.mlp, FeedForward):
                params.append(layer.mlp.gate_proj)
                params.append(layer.mlp.down_proj)
                params.append(layer.mlp.up_proj)
            elif isinstance(layer.mlp, MOEFeedForward):
                params.append(layer.mlp.gate)
                for expert in layer.mlp.experts:
                    params.append(expert.gate_proj)
                    params.append(expert.down_proj)
                    params.append(expert.up_proj)
        
        return params

    def save(self, path):
        """
        保存模型参数
        """
        import pickle
        
        params_data = {}
        params_list = self.parameters()
        for idx, param in enumerate(params_list):
            if hasattr(param, 'torch_tensor') and param.torch_tensor is not None:
                # 保存 torch tensor（移到 CPU）
                params_data[f'param_{idx}'] = param.torch_tensor.cpu()
            elif hasattr(param, 'data'):
                params_data[f'param_{idx}'] = torch.from_numpy(param.data)
        
        save_dict = {
            'config': self.config,
            'params_data': params_data
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path):
        """
        加载模型参数
        """
        import pickle
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        config = save_dict['config']
        model = cls(config)
        
        params_list = model.parameters()
        params_data = save_dict['params_data']
        
        for idx, param in enumerate(params_list):
            if f'param_{idx}' in params_data:
                tensor_data = params_data[f'param_{idx}']
                if isinstance(tensor_data, torch.Tensor):
                    # 移动到正确设备
                    tensor_data = tensor_data.to(DEVICE)
                    if hasattr(param, 'torch_tensor'):
                        param.torch_tensor = tensor_data
                        param.data = tensor_data.cpu().numpy()
                else:
                    param.data = tensor_data
        
        return model