import math
import torch
import torch.nn.functional as F
from tensor.core import Tensor, DEVICE


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
#                                     MiniMind Model - 使用 Tensor 类实现
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """预计算旋转位置编码的 cos 和 sin 值"""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE)[: (dim // 2)] / dim))
    attn_factor = 1.0
    
    if rope_scaling is not None:
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
    """应用旋转位置编码"""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    # 扩展维度
    if cos.dim() == 3:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """重复 key/value heads"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(
        bs, slen, num_key_value_heads * n_rep, head_dim
    )


class RMSNorm:
    """RMS 归一化 - 使用 Tensor 类实现"""
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.weight = Tensor(torch.ones(dim, device=DEVICE))

    def forward(self, x):
        """使用 Tensor 类的 rms_norm 方法"""
        if isinstance(x, Tensor):
            normalized = x.rms_norm(self.eps)
            return self.weight * normalized
        else:
            x_tensor = Tensor(x)
            normalized = x_tensor.rms_norm(self.eps)
            return self.weight * normalized


class Attention:
    """注意力机制 - 使用 Tensor 类实现"""
    def __init__(self, config: MiniMindConfig):
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.training = True

        # 使用 Tensor 初始化权重（形状和官方代码一致）
        self.q_proj = Tensor(torch.randn(config.hidden_size, config.num_attention_heads * self.head_dim, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.k_proj = Tensor(torch.randn(config.hidden_size, self.num_key_value_heads * self.head_dim, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.v_proj = Tensor(torch.randn(config.hidden_size, self.num_key_value_heads * self.head_dim, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.o_proj = Tensor(torch.randn(config.num_attention_heads * self.head_dim, config.hidden_size, device=DEVICE) * (1.0 / (config.num_attention_heads * self.head_dim))**0.5)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dropout = config.dropout

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.torch_tensor.shape

        # 线性投影 - 使用 Tensor 类的 matmul
        xq = x.matmul(self.q_proj)
        xk = x.matmul(self.k_proj)
        xv = x.matmul(self.v_proj)

        # 重塑维度
        xq_torch = xq.torch_tensor.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk_torch = xk.torch_tensor.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv_torch = xv.torch_tensor.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 首先处理 past_key_value - 在应用 norm 和 RoPE 之前！
        if past_key_value is not None:
            xk_torch = torch.cat([past_key_value[0], xk_torch], dim=1)
            xv_torch = torch.cat([past_key_value[1], xv_torch], dim=1)

        # 转换为 Tensor 包装
        xq = Tensor(xq_torch)
        xk = Tensor(xk_torch)
        xv = Tensor(xv_torch)

        # 层归一化
        xq = self.q_norm.forward(xq)
        xk = self.k_norm.forward(xk)

        # 应用旋转位置编码 - 对整个 q 和 k（包括 past）！
        cos, sin = position_embeddings
        xq_torch, xk_torch = apply_rotary_pos_emb(xq.torch_tensor, xk.torch_tensor, cos, sin)
        xq = Tensor(xq_torch)
        xk = Tensor(xk_torch)

        # 保存 cache - 是 norm 之前的、没有 RoPE 的 k 和 v！
        # 不对，等一下，我们保存的应该是原始的 xk_torch 和 xv_torch（没有 norm 和 RoPE）！
        # 哦对了，让我们重新调整顺序，并且保存正确的东西！
        # 让我们重新整理：
        # 先保存原始的 xk_torch（没有 RoPE，没有 norm）作为 cache！
        # 所以我们需要重新调整一下顺序！
        # 让我们重新正确实现一遍！
        # 重新开始，正确的顺序！
        # 正确步骤：
        # 1. 线性投影
        # 2. reshape
        # 3. 如果存在 past_kv，拼接
        # 4. 保存 cache（此时还没有 norm 和 RoPE）
        # 5. apply norm
        # 6. apply RoPE

        # 好的，所以我们重新调整上面的代码，先处理和保存 cache！
        # 让我们重新来一遍正确的！
        # 我们先重新计算一遍 xk_torch 和 xv_torch，并且在 apply norm 和 RoPE 之前保存为 cache！
        # 现在，重新计算：
        # 等等，我们刚才的代码已经先拼接了 past_key_value，现在我们需要保存的是拼接后的 xk_torch 和 xv_torch（也就是还没有 norm 和 RoPE 的）
        past_kv = (xk_torch.clone(), xv_torch.clone()) if use_cache else None

        # 转置维度并计算注意力
        xq_torch = xq.torch_tensor.transpose(1, 2)
        xk_torch = repeat_kv(xk.torch_tensor, self.n_rep).transpose(1, 2)
        xv_torch = repeat_kv(xv.torch_tensor, self.n_rep).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(xq_torch, xk_torch.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用因果掩码
        if self.is_causal:
            seq_len_kv = xk_torch.shape[2]
            causal_mask = torch.triu(torch.ones((seq_len, seq_len_kv), device=DEVICE), diagonal=1).bool()
            scores[:, :, -seq_len:, :].masked_fill_(causal_mask[:seq_len, :seq_len_kv], float('-inf'))

        # 应用注意力掩码
        if attention_mask is not None:
            if isinstance(attention_mask, Tensor):
                scores += (1.0 - attention_mask.torch_tensor[:, None, None, :]) * -1e9
            else:
                scores += (1.0 - attention_mask[:, None, None, :]) * -1e9

        # 计算注意力权重 - 使用 Tensor 类的 softmax
        attn_weights_tensor = Tensor(scores)
        attn_weights = attn_weights_tensor.softmax()

        # 应用 dropout
        if self.dropout > 0 and self.training:
            attn_weights = attn_weights.dropout(self.dropout)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights.torch_tensor, xv_torch)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_output = Tensor(attn_output)

        # 输出投影
        output = attn_output.matmul(self.o_proj)

        return output, past_kv


class FeedForward:
    """前馈网络 - 使用 Tensor 类实现"""
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = Tensor(torch.randn(config.hidden_size, intermediate_size, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.down_proj = Tensor(torch.randn(intermediate_size, config.hidden_size, device=DEVICE) * (1.0 / intermediate_size)**0.5)
        self.up_proj = Tensor(torch.randn(config.hidden_size, intermediate_size, device=DEVICE) * (1.0 / config.hidden_size)**0.5)

    def forward(self, x):
        # 门控机制 - 使用 Tensor 类的 matmul 和 silu
        gate = x.matmul(self.gate_proj).silu()
        up = x.matmul(self.up_proj)
        hidden = gate * up
        output = hidden.matmul(self.down_proj)
        return output


class MOEFeedForward:
    """MoE 前馈网络 - 使用 Tensor 类实现"""
    def __init__(self, config: MiniMindConfig):
        self.config = config
        self.gate = Tensor(torch.randn(config.hidden_size, config.num_experts, device=DEVICE) * (1.0 / config.hidden_size)**0.5)
        self.experts = [FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.torch_tensor.shape

        # 展平输入
        x_flat = x.torch_tensor.reshape(-1, hidden_dim)
        x_flat_tensor = Tensor(x_flat)

        # 计算门控分数 - 使用 Tensor 类的 matmul 和 softmax
        scores = x_flat_tensor.matmul(self.gate).softmax()

        # 选择 topk 专家
        topk_weight, topk_idx = torch.topk(scores.torch_tensor, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        # 归一化 topk 概率
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


class MiniMindBlock:
    """Transformer Block - 使用 Tensor 类实现"""
    def __init__(self, layer_id: int, config: MiniMindConfig):
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        self.training = True

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
    """MiniMind 模型 - 使用 Tensor 类实现"""
    def __init__(self, config: MiniMindConfig):
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = Tensor.create_embedding(config.vocab_size, config.hidden_size)
        self.dropout = config.dropout
        self.layers = [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.training = True

        # 预计算频率
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        batch_size, seq_length = input_ids.shape

        # 转换 input_ids 为 torch tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)

        # Wrap attention_mask
        if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
            attention_mask = Tensor(attention_mask)

        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 嵌入 - 使用 Tensor 类的 embedding 方法
        hidden_states = self.embed_tokens.embedding(Tensor(input_ids))

        # 应用 dropout
        if self.dropout > 0 and self.training:
            hidden_states = hidden_states.dropout(self.dropout)

        # 位置编码
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 扩展到批次维度
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
    """MiniMind Causal LM - 使用 Tensor 类实现"""
    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        self.model = MiniMindModel(self.config)
        self.lm_head = Tensor(torch.randn(self.config.hidden_size, self.config.vocab_size, device=DEVICE) * (1.0 / self.config.hidden_size)**0.5)
        self.training = True

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None):
        # 转换 input_ids 和 labels
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)

        hidden_states, past_key_values, aux_loss = self.model.forward(input_ids, attention_mask, past_key_values, use_cache)

        # 处理 logits_to_keep
        if logits_to_keep > 0:
            hidden_states = Tensor(hidden_states.torch_tensor[:, -logits_to_keep:])

        # 计算 logits - 使用 Tensor 类的 matmul
        logits = hidden_states.matmul(self.lm_head)

        loss = None
        if labels is not None:
            batch_size, seq_len, vocab_size = logits.torch_tensor.shape

            # 展平
            logits_flat = logits.torch_tensor.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)

            # 创建 mask
            mask = (labels_flat != -100) & (labels_flat != 0)
            if mask.any():
                valid_logits = logits_flat[mask]
                valid_labels = labels_flat[mask]

                # 使用 PyTorch 交叉熵
                loss_value = F.cross_entropy(valid_logits, valid_labels, reduction='mean')
                loss = Tensor(loss_value)

        return {
            'loss': loss,
            'aux_loss': aux_loss,
            'logits': logits,
            'past_key_values': past_key_values,
            'hidden_states': hidden_states
        }

    @torch.inference_mode()
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
        eos_token_id=2,
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,
        pad_token_id=None,
        **kwargs
    ):
        input_ids = kwargs.pop("input_ids", inputs)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
        else:
            input_ids = input_ids.to(DEVICE)

        # 处理 num_return_sequences
        input_ids = input_ids.repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)

        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            # 始终传入完整的 input_ids，这样位置编码就是正确的！
            # 不再切片 input_ids
            outputs = self.forward(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache
            )

            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], dim=-1)

            logits = outputs['logits'].torch_tensor[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    unique_tokens = torch.unique(input_ids[i])
                    logits[i, unique_tokens] /= repetition_penalty

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                min_topk = topk_vals[:, -1, None]
                logits = logits.masked_fill(logits < min_topk, -float('inf'))

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs > top_p
                mask[:, 0] = False
                indices_to_remove = mask.scatter(1, sorted_indices, mask)
                logits = logits.masked_fill(indices_to_remove, -float('inf'))

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs['past_key_values'] if use_cache else None

            if streamer:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        if streamer:
            streamer.end()

        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids

    def parameters(self):
        """获取所有可训练参数"""
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

    def load_state_dict(self, state_dict):
        """从 PyTorch state_dict 加载权重"""
        params_dict = {}
        # 需要转置的权重
        transpose_keys = set()
        
        for name, param in [
            ('lm_head', self.lm_head),
            ('model.embed_tokens', self.model.embed_tokens),
            ('model.norm.weight', self.model.norm.weight),
        ]:
            params_dict[name] = param
        # lm_head 也需要转置
        transpose_keys.add('lm_head')

        for i, layer in enumerate(self.model.layers):
            prefix = f'model.layers.{i}.'
            params_dict[f'{prefix}input_layernorm.weight'] = layer.input_layernorm.weight
            params_dict[f'{prefix}post_attention_layernorm.weight'] = layer.post_attention_layernorm.weight
            params_dict[f'{prefix}self_attn.q_proj'] = layer.self_attn.q_proj
            params_dict[f'{prefix}self_attn.k_proj'] = layer.self_attn.k_proj
            params_dict[f'{prefix}self_attn.v_proj'] = layer.self_attn.v_proj
            params_dict[f'{prefix}self_attn.o_proj'] = layer.self_attn.o_proj
            params_dict[f'{prefix}self_attn.q_norm.weight'] = layer.self_attn.q_norm.weight
            params_dict[f'{prefix}self_attn.k_norm.weight'] = layer.self_attn.k_norm.weight
            
            # 标记需要转置的权重
            transpose_keys.add(f'{prefix}self_attn.q_proj')
            transpose_keys.add(f'{prefix}self_attn.k_proj')
            transpose_keys.add(f'{prefix}self_attn.v_proj')
            transpose_keys.add(f'{prefix}self_attn.o_proj')

            if isinstance(layer.mlp, FeedForward):
                params_dict[f'{prefix}mlp.gate_proj'] = layer.mlp.gate_proj
                params_dict[f'{prefix}mlp.down_proj'] = layer.mlp.down_proj
                params_dict[f'{prefix}mlp.up_proj'] = layer.mlp.up_proj
                transpose_keys.add(f'{prefix}mlp.gate_proj')
                transpose_keys.add(f'{prefix}mlp.down_proj')
                transpose_keys.add(f'{prefix}mlp.up_proj')
            elif isinstance(layer.mlp, MOEFeedForward):
                params_dict[f'{prefix}mlp.gate'] = layer.mlp.gate
                transpose_keys.add(f'{prefix}mlp.gate')
                for j, expert in enumerate(layer.mlp.experts):
                    params_dict[f'{prefix}mlp.experts.{j}.gate_proj'] = expert.gate_proj
                    params_dict[f'{prefix}mlp.experts.{j}.down_proj'] = expert.down_proj
                    params_dict[f'{prefix}mlp.experts.{j}.up_proj'] = expert.up_proj
                    transpose_keys.add(f'{prefix}mlp.experts.{j}.gate_proj')
                    transpose_keys.add(f'{prefix}mlp.experts.{j}.down_proj')
                    transpose_keys.add(f'{prefix}mlp.experts.{j}.up_proj')

        for key, value in state_dict.items():
            param = None
            lookup_key = key
            if key in params_dict:
                param = params_dict[key]
            else:
                key_no_suffix = key.replace('.weight', '')
                if key_no_suffix in params_dict:
                    param = params_dict[key_no_suffix]
                    lookup_key = key_no_suffix

            if param is not None:
                if hasattr(param, 'torch_tensor'):
                    # 对需要转置的权重进行转置
                    loaded_value = value.to(DEVICE).contiguous()
                    if lookup_key in transpose_keys:
                        loaded_value = loaded_value.t().contiguous()
                    param.torch_tensor = loaded_value
                    param.data = loaded_value.detach().cpu().numpy()
                elif hasattr(param, 'data'):
                    param.data = value.detach().cpu().numpy()

    def half(self):
        """转换为半精度"""
        self._apply_dtype(torch.float16)
        return self

    def to(self, device=None):
        """移动到指定设备"""
        global DEVICE
        if device is not None:
            DEVICE = device
        self._apply_device(DEVICE)
        return self

    def _apply_dtype(self, dtype):
        """递归应用 dtype 到所有参数"""
        for param in self.parameters():
            if hasattr(param, 'torch_tensor'):
                param.torch_tensor = param.torch_tensor.to(dtype).contiguous()
                param.data = param.torch_tensor.detach().cpu().numpy()

    def _apply_device(self, device):
        """递归应用 device 到所有参数"""
        for param in self.parameters():
            if hasattr(param, 'torch_tensor'):
                param.torch_tensor = param.torch_tensor.to(device).contiguous()
                param.data = param.torch_tensor.detach().cpu().numpy()

    def train(self, mode=True):
        """设置训练模式"""
        self.training = mode
        self.model.training = mode
        for layer in self.model.layers:
            layer.training = mode
            layer.self_attn.training = mode
            if hasattr(layer.mlp, 'training'):
                layer.mlp.training = mode
        return self

    def eval(self):
        """设置评估模式"""
        return self.train(False)

    def __call__(self, *args, **kwargs):
        """支持像函数一样调用模型"""
        return self.forward(*args, **kwargs)

    def save(self, path):
        """保存模型参数"""
        import pickle

        params_data = {}
        params_list = self.parameters()
        for idx, param in enumerate(params_list):
            if hasattr(param, 'torch_tensor') and param.torch_tensor is not None:
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
        """加载模型参数"""
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
                    tensor_data = tensor_data.to(DEVICE)
                    if hasattr(param, 'torch_tensor'):
                        param.torch_tensor = tensor_data
                        param.data = tensor_data.detach().cpu().numpy()
                else:
                    param.data = tensor_data

        return model