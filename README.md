# 一步一步学Dl：dl 实战学习路径 
> **前置技能**：熟悉 Python 基础，了解基本线性代数。
> **核心原则**：**“逻辑手写，底层加速”**。
> - ❌ **禁止**：直接调用 `torch.nn.Linear`, `torch.nn.Transformer`, `torch.nn.LSTM` 等封装好的**网络层**。禁止使用 `transformers` 库。
> - ✅ **允许**：使用 `torch` 作为**基础算子库** (类似超级版的 NumPy)。使用 `torch.matmul`, `torch.softmax`, `torch.sum` 等基础函数。利用 PyTorch 的 `autograd` 自动计算梯度，利用 `.cuda()` 自动调用 RTX 2070 加速。
> **目标**：亲手搭建模型的每一层连接，亲手编写前向传播的数学公式，亲手写训练循环，但让 PyTorch 负责底层的矩阵加速和梯度推导。

---

## 📅 阶段概览

| 阶段 | 核心主题 | 关键组件 | 预计耗时 (RTX 2070) | 难度 |
| :--- | :--- | :--- | :--- | :---: |
| **Phase 1** | 数学基石 | 梯度下降、手动求导验证 | < 1 分钟 | ⭐ |
| **Phase 2** | 神经网络基础 | 手动实现 FCN, 激活函数公式 | ~30 秒 | ⭐⭐ |
| **Phase 3** | 工程化训练 | 手写 Loss, 手写 Optimizer 更新步 | ~1 分钟 | ⭐⭐ |
| **Phase 4** | 深度学习架构 | 手动拼凑 Conv, LSTM 门控单元 | ~5 分钟 | ⭐⭐⭐ |
| **Phase 5** | Transformer & LLM | 手写 Attention, Positional Encoding | ~10 分钟 | ⭐⭐⭐⭐ |
| **Bonus** | 语音扩展 | 手写 MFCC, CTC Loss | ~2 小时 | ⭐⭐⭐⭐⭐ |

---

## Phase 1: 数学基石 (验证梯度)
**目标**：在不依赖自动求导的情况下，手动推导并验证梯度，理解 `backward` 在做什么。

### 1.1 单参数拟合 (手动推导 vs 自动求导)
*   **任务**：
    1.  用 Python 原生 float 实现 $y=wx+b$ 的梯度下降 (完全手动)。
    2.  用 `torch.Tensor` 实现同样的逻辑，调用 `loss.backward()`。
    3.  **对比**：打印两者的梯度值，确认一致。
*   **代码重点**：
    ```python
    # 手动推导
    dw_manual = 2 * (pred - target) * x
    
    # PyTorch 自动
    x_t = torch.tensor([x], requires_grad=True)
    loss_t = (w_t * x_t - target)**2
    loss_t.backward()
    dw_auto = x_t.grad.item()
    
    assert abs(dw_manual - dw_auto) < 1e-6
    ```

---

## Phase 2: 神经网络基础 (拒绝 nn.Module 封装)
**目标**：理解层与层之间是如何通过矩阵运算连接的。

### 2.1 手写全连接层 (MyLinear)
*   **禁止**：`torch.nn.Linear`
*   **必须手写**：
    ```python
    class MyLinear:
        def __init__(self, in_features, out_features):
            # 手动初始化权重 (Xavier 初始化)
            self.weight = torch.randn(out_features, in_features) * (2/in_features)**0.5
            self.bias = torch.zeros(out_features)
            self.weight.requires_grad = True
            self.bias.requires_grad = True
            
        def forward(self, x):
            # 核心：手动写出矩阵乘法公式
            return torch.matmul(x, self.weight.t()) + self.bias
    ```
*   **激活函数**：不要直接用 `nn.ReLU()`，而是写 `def relu(x): return x * (x > 0)` 或 `torch.maximum(x, torch.zeros_like(x))`。

### 2.2 搭建简易 MLP
*   **任务**：实例化多个 `MyLinear`，串联起来，加上手写的 `relu`。
*   **测试**：拟合三维曲面 $z = \sin(x) + \cos(y)$。

---

## Phase 3: 训练引擎 (手写循环)
**目标**：掌握训练的本质流程。

### 3.1 手写优化器步骤
*   **禁止**：`torch.optim.Adam` (初期建议手写以理解原理，后期可用)。
*   **必须手写**：
    ```python
    # 假设 model 是你的网络，lr = 0.001
    for param in model.parameters():
        if param.grad is not None:
            with torch.no_grad():
                # 手动实现 SGD 更新公式
                param -= lr * param.grad
                # 清空梯度
                param.grad.zero_()
    ```
*   **进阶**：尝试手写 Adam 的动量更新逻辑 ($m_t, v_t$)。

### 3.2 数据加载
*   **任务**：读取文本/图片文件 -> 转为 Python List -> `torch.tensor(list).cuda()`。
*   **重点**：理解 `.cuda()` 如何将数据搬运到 RTX 2070 显存，之后的所有运算自动在 GPU 执行。

---

## Phase 4: 经典架构 (拆解黑盒)
**目标**：打开 CNN 和 RNN 的黑盒，看清内部连线。

### 4.1 手写 LeNet-5 (卷积层)
*   **禁止**：`torch.nn.Conv2d`
*   **挑战**：卷积的手写非常复杂且慢。
    *   *策略调整*：此阶段允许使用 `torch.nn.functional.conv2d` (函数式接口)，但你必须**手动管理权重**和**偏置**，并理解输入输出的维度变化 (H, W, Channel)。
    *   **核心学习**：理解 `padding`, `stride`, `kernel_size` 如何影响输出尺寸。手动计算输出尺寸公式。

### 4.2 手写 LSTM 单元
*   **禁止**：`torch.nn.LSTM`
*   **必须手写**：
    ```python
    def my_lstm_step(x, h_prev, c_prev, W_i, W_f, W_o, W_c):
        # 手动写出四个门的公式
        i_gate = torch.sigmoid(torch.matmul(x, W_i.t()) + ...)
        f_gate = torch.sigmoid(...)
        o_gate = torch.sigmoid(...)
        g_gate = torch.tanh(...)
        
        c_next = f_gate * c_prev + i_gate * g_gate
        h_next = o_gate * torch.tanh(c_next)
        return h_next, c_next
    ```
*   **测试**：用这个单步函数构建一个循环，处理序列数据。

---

## Phase 5: Transformer 与大模型 (核心战场)
**目标**：完全复现 Transformer 架构，这是本项目的最高潮。

### 5.1 手写自注意力 (Self-Attention)
*   **禁止**：`torch.nn.MultiheadAttention`
*   **必须手写**：
    ```python
    def self_attention(q, k, v, mask=None):
        d_k = q.size(-1)
        # 1. 计算 QK^T
        scores = torch.matmul(q, k.transpose(-2, -1))
        # 2. 缩放
        scores = scores / (d_k ** 0.5)
        # 3. Mask (因果掩码，防止看未来)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 4. Softmax (数值稳定版)
        attn = torch.softmax(scores, dim=-1)
        # 5. 加权求和
        output = torch.matmul(attn, v)
        return output
    ```

### 5.2 手写位置编码 (Positional Encoding)
*   **任务**：根据公式 $\sin(\frac{pos}{10000^{2i/d}})$ 生成矩阵，并加到 Embedding 上。
*   **重点**：理解为什么需要位置编码，以及奇偶维度的正弦余弦分布。

### 5.3 组装 Decoder-Only GPT
*   **架构**：
    1.  Token Embedding (查表)
    2.  + Positional Encoding
    3.  N 个 Block:
        *   LayerNorm -> Self-Attention -> Residual (+)
        *   LayerNorm -> MLP (两个 Linear + Gelu) -> Residual (+)
    4.  LayerNorm -> Linear (投影回词表) -> Softmax
*   **全部由你类组合而成**，不使用 `torch.nn.TransformerDecoder`。

### 5.4 训练小 GPT (NanoGPT)
*   **数据**：TinyStories 或 Shakespeare。
*   **Tokenizer**：手写一个简单的字符级或 BPE 分词器。
*   **训练循环**：
    ```python
    # 伪代码
    input_ids = batch.cuda()
    target_ids = batch.cuda()
    
    logits = my_gpt_model(input_ids) # 你的模型
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
    
    loss.backward() # PyTorch 自动帮你算几百万参数的梯度
    optimizer_step() # 你的手写更新或 torch.optim
    ```
*   **预期效果**：在 RTX 2070 上，训练一个 5M 参数的模型，几分钟内即可看到 Loss 下降，并能生成通顺的短句子。

---

## 🎙️ Bonus: 语音任务 (高难度)
*   **ASR/TTS**：依然遵循“逻辑手写”原则。
*   **FFT**：可以使用 `torch.fft.fft` (这是基础数学函数，不是神经网络层)。
*   **CTC Loss**：建议手写 CTC 的前向概率计算逻辑，或者深入研究 `torch.nn.CTCLoss` 的输入要求，手动对齐数据。

---

## 🛠️ 开发环境建议
1.  **安装**：
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # 确保安装的是 CUDA 版本，这样 .cuda() 才能调用你的 2070
    ```
2.  **验证 GPU**：
    ```python
    import torch
    print(torch.cuda.is_available()) # 应为 True
    print(torch.cuda.get_device_name(0)) # 应显示 GeForce RTX 2070
    ```
3.  **调试技巧**：
    *   随时使用 `print(tensor.shape)` 检查维度。
    *   使用 `tensor.device` 确认数据是否在 `cuda:0` 上。
    *   如果显存爆了 (OOM)，减小 `batch_size`。
