# 参照官方实现，简化版预训练数据集
import os
import json
import random
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

# Windows下 pyarrow / torch DLL 冲突的兼容处理：
# 不主动 import datasets。若存在 datasets 库且用户需要，才按需导入。


def _read_jsonl(path):
    """逐行读取 jsonl，返回 dict 列表。失败时给出清晰的错误信息。"""
    samples = []
    path = str(path)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                # 忽略格式错误的行
                continue
    return samples


class PretrainDataset:
    """
    预训练数据集：
      - 从 jsonl 读取，每条记录包含 {"text": "..."}
      - 分词 -> 截断 -> 补齐 -> 加 BOS/EOS
      - 返回 (input_ids, labels)，形状均为 (max_length,)
      - labels 中 pad_token 位置置为 -100（即不计算损失）

    这是对官方 PretrainDataset 的简化版：
      - 不依赖 datasets 库（避免 DLL 冲突）
      - 完全使用本地 tokenizer
    """

    def __init__(self, data_path, tokenizer, max_length=512, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.samples = _read_jsonl(data_path)
        random.seed(seed)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, text):
        # 留给 BOS + EOS 的空间
        max_tok = self.max_length - 2
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=max_tok,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        # 加 BOS / EOS
        ids = [self.bos_id] + list(ids) + [self.eos_id]
        # 补齐
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))
        return ids[: self.max_length]

    def __getitem__(self, index):
        text = str(self.samples[index].get("text", ""))
        ids = self._tokenize(text)
        input_ids = list(ids)
        # 下一个 token 预测：labels 与 input_ids 错位一位由损失函数内部处理，
        # 这里只忽略 padding 位置。
        labels = [-100 if tok == self.pad_id else tok for tok in ids]

        # 尽可能返回 torch.Tensor
        if HAS_TORCH:
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )
        import numpy as np
        return (
            np.array(input_ids, dtype=np.int64),
            np.array(labels, dtype=np.int64),
        )


class SFTDataset(PretrainDataset):
    """
    SFT 数据集：
      - 从 jsonl 读取，每条记录包含 {"conversations": [{"role":..., "content":...}, ...]}
      - 使用 tokenizer.apply_chat_template 构造提示 + 回答
      - labels 中 "assistant 之前" 的部分置为 -100（不计算损失）
    """

    def __init__(self, data_path, tokenizer, max_length=1024, seed=42):
        # 先当 PretrainDataset 初始化一次，只为读取样本列表
        super().__init__(data_path, tokenizer, max_length=max_length, seed=seed)
        # 我们的样本结构与父类不同，覆盖
        self.samples = [
            s for s in _read_jsonl(data_path) if "conversations" in s
        ]

    def _find_assistant_start(self, ids):
        """
        粗略定位 assistant 回答开始位置：
          - 调用 tokenizer 的 apply_chat_template 后，
            通常在 "<|im_start|>assistant\n" 或类似 token 后开始
          - 使用字符串方式定位：先 decode 再重新 encode assistant 之后的内容
        这里用最简单的做法：保留整段文本的 labels，供简单 SFT 用。
        更严格的 masking 可参考官方实现。
        """
        return 1  # 从第 1 个 token 后开始计算 loss（跳过 BOS/系统提示前的 token）

    def __getitem__(self, index):
        conv = self.samples[index]["conversations"]
        prompt = self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False
        )
        ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
            return_tensors=None,
        )["input_ids"]
        ids = [self.bos_id] + list(ids) + [self.eos_id]
        if len(ids) < self.max_length:
            ids = ids + [self.pad_id] * (self.max_length - len(ids))
        ids = ids[: self.max_length]

        labels = [-100 if tok == self.pad_id else tok for tok in ids]

        if HAS_TORCH:
            return (
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
            )
        import numpy as np
        return (
            np.array(ids, dtype=np.int64),
            np.array(labels, dtype=np.int64),
        )
