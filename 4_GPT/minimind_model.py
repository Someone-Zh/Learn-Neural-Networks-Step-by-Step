import numpy as np
from tensor.core import Tensor
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
        self.weight = Tensor(np.ones(dim))

    def norm(self, x):
        # 计算均方根值 (RMS): sqrt(mean(x^2))
        # 使用数值稳定的实现
        rms = np.sqrt(np.mean(x.data ** 2, axis=-1, keepdims=True) + self.eps)
        # 归一化
        normalized = Tensor(x.data / rms)
        return normalized

    def forward(self, x):
        # 应用权重缩放
        normalized = self.norm(x)
        return self.weight * normalized

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    import math
    freqs = 1.0 / (rope_base ** (np.arange(0, dim, 2)[: (dim // 2)].astype(float) / dim))
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
            ramp = np.clip((np.arange(dim // 2).astype(float) - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = np.arange(end)
    freqs = np.outer(t, freqs).astype(float)
    freqs_cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1) * attn_factor
    freqs_sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        return np.concatenate((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), axis=-1)
    
    # 扩展 cos 和 sin 的维度以匹配 q 和 k
    # q, k: (batch, seq, heads, head_dim)
    # cos, sin: (batch, seq, head_dim)
    # 需要将 cos, sin 扩展为 (batch, seq, 1, head_dim) 以便广播
    if cos.ndim == 3:
        cos = cos[:, :, np.newaxis, :]  # (batch, seq, 1, head_dim)
        sin = sin[:, :, np.newaxis, :]  # (batch, seq, 1, head_dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return np.repeat(x[:, :, :, np.newaxis, :], n_rep, axis=3).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)

class Attention:
    def __init__(self, config: MiniMindConfig):
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        
        # 初始化权重
        self.q_proj = Tensor(np.random.randn(config.hidden_size, config.num_attention_heads * self.head_dim) * np.sqrt(1.0 / config.hidden_size))
        self.k_proj = Tensor(np.random.randn(config.hidden_size, self.num_key_value_heads * self.head_dim) * np.sqrt(1.0 / config.hidden_size))
        self.v_proj = Tensor(np.random.randn(config.hidden_size, self.num_key_value_heads * self.head_dim) * np.sqrt(1.0 / config.hidden_size))
        self.o_proj = Tensor(np.random.randn(config.num_attention_heads * self.head_dim, config.hidden_size) * np.sqrt(1.0 / (config.num_attention_heads * self.head_dim)))
        
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dropout = config.dropout

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.data.shape
        
        # 线性投影
        xq = x.matmul(self.q_proj)
        xk = x.matmul(self.k_proj)
        xv = x.matmul(self.v_proj)
        
        # 重塑
        xq_data = xq.data.reshape(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk_data = xk.data.reshape(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv_data = xv.data.reshape(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        # 转换为Tensor
        xq = Tensor(xq_data)
        xk = Tensor(xk_data)
        xv = Tensor(xv_data)
        
        # 层归一化
        xq = self.q_norm.forward(xq)
        xk = self.k_norm.forward(xk)
        
        # 应用旋转位置编码
        cos, sin = position_embeddings
        xq_data, xk_data = apply_rotary_pos_emb(xq.data, xk.data, cos, sin)
        xq = Tensor(xq_data)
        xk = Tensor(xk_data)
        
        # 处理past_key_value
        if past_key_value is not None:
            xk_data = np.concatenate([past_key_value[0], xk_data], axis=1)
            xv_data = np.concatenate([past_key_value[1], xv.data], axis=1)
            xk = Tensor(xk_data)
            xv = Tensor(xv_data)
        past_kv = (xk_data, xv.data) if use_cache else None
        
        # 转置维度
        xq_data = xq.data.transpose(0, 2, 1, 3)
        xk_data = repeat_kv(xk.data, self.n_rep).transpose(0, 2, 1, 3)
        xv_data = repeat_kv(xv.data, self.n_rep).transpose(0, 2, 1, 3)
        
        # 计算注意力分数
        scores = np.matmul(xq_data, xk_data.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        # 应用因果掩码
        if self.is_causal:
            seq_len_kv = xk_data.shape[2]
            causal_mask = np.triu(np.ones((seq_len, seq_len_kv)), k=1).astype(float) * -1e10
            scores[:, :, -seq_len:, :] += causal_mask
        
        # 应用注意力掩码
        if attention_mask is not None:
            scores += (1.0 - attention_mask.data[:, np.newaxis, np.newaxis, :]) * -1e9
        
        # 计算注意力权重
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        # 应用dropout
        if self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, size=attn_weights.shape).astype(float)
            attn_weights = attn_weights * mask / (1 - self.dropout)
        
        # 计算注意力输出
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
        self.gate_proj = Tensor(np.random.randn(config.hidden_size, intermediate_size) * np.sqrt(1.0 / config.hidden_size))
        self.down_proj = Tensor(np.random.randn(intermediate_size, config.hidden_size) * np.sqrt(1.0 / intermediate_size))
        self.up_proj = Tensor(np.random.randn(config.hidden_size, intermediate_size) * np.sqrt(1.0 / config.hidden_size))

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
        self.gate = Tensor(np.random.randn(config.hidden_size, config.num_experts) * np.sqrt(1.0 / config.hidden_size))
        self.experts = [FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.data.shape
        x_flat = x.data.reshape(-1, hidden_dim)
        x_flat_tensor = Tensor(x_flat)
        
        # 计算门控分数
        scores = x_flat_tensor.matmul(self.gate).softmax()
        
        # 选择topk专家
        topk_weight, topk_idx = np.topk(scores.data, k=self.config.num_experts_per_tok, axis=-1, sorted=False)
        
        # 归一化topk概率
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(axis=-1, keepdims=True) + 1e-20)
        
        # 初始化输出
        y = np.zeros_like(x_flat)
        
        # 专家前向传播
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = np.where(mask.any(axis=-1))[0]
                weight = topk_weight[mask].reshape(-1, 1)
                expert_input = Tensor(x_flat[token_idx])
                expert_output = expert.forward(expert_input).data
                y[token_idx] += (expert_output * weight).squeeze()
        
        # 计算辅助损失
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
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        
        # 嵌入
        hidden_states = self.embed_tokens.embedding(Tensor(input_ids))
        
        # 应用dropout
        if self.dropout > 0:
            hidden_states = hidden_states.dropout(self.dropout)
        
        # 位置编码
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        
        # 扩展到批次维度
        position_embeddings = (
            np.expand_dims(position_embeddings[0], axis=0).repeat(batch_size, axis=0),
            np.expand_dims(position_embeddings[1], axis=0).repeat(batch_size, axis=0)
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
        self.lm_head = Tensor(np.random.randn(self.config.hidden_size, self.config.vocab_size) * np.sqrt(1.0 / self.config.hidden_size))
        # 共享权重
        self.model.embed_tokens.data = self.lm_head.data.T

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None):
        hidden_states, past_key_values, aux_loss = self.model.forward(input_ids, attention_mask, past_key_values, use_cache)
        
        # 处理logits_to_keep
        if logits_to_keep > 0:
            hidden_states = Tensor(hidden_states.data[:, -logits_to_keep:])
        
        # 计算logits
        logits = hidden_states.matmul(self.lm_head)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失 - 保持计算图连接
            batch_size, seq_len, vocab_size = logits.data.shape
            
            # 展平logits（保持Tensor连接）
            logits_flat_data = logits.data.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            
            # 创建mask忽略-100标签和padding标签(0)
            mask = (labels_flat != -100) & (labels_flat != 0)
            if mask.any():
                # 提取有效样本
                valid_logits_data = logits_flat_data[mask]
                valid_labels = labels_flat[mask]
                num_valid = len(valid_labels)
                
                # 计算log_softmax（数值稳定版本）
                max_logits = np.max(valid_logits_data, axis=-1, keepdims=True)
                shifted_logits = valid_logits_data - max_logits
                exp_logits = np.exp(shifted_logits)
                sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
                log_probs = shifted_logits - np.log(sum_exp)
                
                # 计算损失
                loss_value = -np.mean(log_probs[np.arange(num_valid), valid_labels])
                
                # 创建损失Tensor，连接到logits（这样梯度才能传回模型参数）
                loss = Tensor(loss_value, (logits,), 'cross_entropy')
                
                # 设置反向传播函数 - 梯度需要传回logits
                def _make_backward(valid_logits_data, valid_labels, mask, logits_flat_data, batch_size, seq_len, vocab_size):
                    def _backward():
                        if loss.grad is not None:
                            # 计算softmax概率
                            probs = np.exp(log_probs)
                            
                            # 创建one-hot编码
                            one_hot = np.zeros_like(valid_logits_data)
                            one_hot[np.arange(num_valid), valid_labels] = 1.0
                            
                            # 计算梯度: (softmax - one_hot) / n
                            grad_valid = (probs - one_hot) / num_valid
                            
                            # 将梯度放回原始位置
                            grad_flat = np.zeros_like(logits_flat_data)
                            grad_flat[mask] = grad_valid
                            
                            # 重塑回原始形状 (batch, seq_len, vocab_size)
                            grad_logits = grad_flat.reshape(batch_size, seq_len, vocab_size)
                            
                            # 初始化logits.grad如果为None
                            if logits.grad is None:
                                logits.grad = np.zeros_like(logits.data)
                            
                            # 传回梯度到logits
                            logits.grad = logits.grad + grad_logits * loss.grad
                    return _backward
                
                loss._backward = _make_backward(valid_logits_data, valid_labels, mask, logits_flat_data, batch_size, seq_len, vocab_size)
        
        return {
            'loss': loss,
            'aux_loss': aux_loss,
            'logits': logits,
            'past_key_values': past_key_values,
            'hidden_states': hidden_states
        }
    
    def generate(self, input_ids, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2):
        input_ids = input_ids.copy()
        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self.forward(input_ids, use_cache=True)
            logits = outputs['logits'].data[:, -1, :]
            
            # 应用温度
            logits = logits / temperature
            
            # 应用top-k
            if top_k > 0:
                top_k_values = np.sort(logits, axis=-1)[:, -top_k:]
                min_top_k = top_k_values[:, -1:]
                logits = np.where(logits < min_top_k, -float('inf'), logits)
            
            # 应用top-p
            if top_p < 1.0:
                sorted_logits = np.sort(logits, axis=-1)[:, ::-1]
                sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
                cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits), axis=-1, keepdims=True), axis=-1)
                mask = cumulative_probs > top_p
                mask[:, 0] = False  # 至少保留第一个token
                # 使用高级索引正确应用掩码
                for i in range(logits.shape[0]):
                    sorted_mask = mask[i]
                    sorted_idx = sorted_indices[i]
                    logits[i, sorted_idx[sorted_mask]] = -float('inf')
            
            # 采样
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
            next_token = np.random.choice(self.config.vocab_size, p=probs[0])
            
            # 追加到输入
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            
            # 检查是否生成eos_token
            if next_token == eos_token_id:
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
        保存模型（只保存参数数据）
        """
        import pickle
        
        # 收集所有参数的数据
        params_data = {}
        params_list = self.parameters()
        for idx, param in enumerate(params_list):
            if hasattr(param, 'data'):
                params_data[f'param_{idx}'] = param.data
        
        # 保存配置和参数
        save_dict = {
            'config': self.config,
            'params_data': params_data
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path):
        """
        加载模型
        """
        import pickle
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        # 创建新模型
        config = save_dict['config']
        model = cls(config)
        
        # 加载参数
        params_list = model.parameters()
        params_data = save_dict['params_data']
        
        for idx, param in enumerate(params_list):
            if f'param_{idx}' in params_data:
                param.data = params_data[f'param_{idx}']
        
        return model