"""
================================================================================
                       MiniMind 预训练脚本（用户自实现版）
================================================================================

参考官方 trainer/train_pretrain.py 的做法，做了以下重构：
  1. 使用真实的 jsonl 预训练数据（dataset/lm_dataset.PretrainDataset）
  2. 支持梯度累积 gradient accumulation
  3. 余弦学习率 + warmup
  4. 支持检查点 save / resume
  5. 更清晰的日志打印（loss / lr / 进度）

限制：
  - 使用用户自实现的 Tensor / MiniMindForCausalLM / Adam（非官方 torch.nn）
  - 不做多卡分布式训练（简化）
  - 不做半精度混合训练（保留 float32，保证稳定）
"""

import os
import sys
import time
import math
import json
import random
import argparse

# —— 让脚本目录下的子包可被导入 ——
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import torch
from transformers import AutoTokenizer

from minimind_model import MiniMindConfig, MiniMindForCausalLM
from optim.optimizer import Adam, ExponentialLR
from dataset.lm_dataset import PretrainDataset


# =============================================================================
# 训练工具函数
# =============================================================================

def set_seed(seed: int):
    """固定随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_cosine(
    current_step: int,
    total_steps: int,
    learning_rate: float,
    min_lr_ratio: float = 0.1,
    warmup_steps: int = 0,
) -> float:
    """
    与官方保持一致：warmup + cosine decay
    前 warmup_steps 线性上升，之后按 cosine 衰减到 min_lr
    """
    if current_step < warmup_steps:
        # warmup: 线性上升
        return learning_rate * float(current_step + 1) / float(max(1, warmup_steps))
    # cosine decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = learning_rate * min_lr_ratio
    return min_lr + (learning_rate - min_lr) * cosine


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# 数据加载：手动 batch（因为我们的 Dataset 返回 torch tensor，不用 torch DataLoader
# 以避免在自实现框架里出现兼容问题）
# =============================================================================

def build_dataloader_iter(ds: PretrainDataset, batch_size: int, shuffle: bool = True):
    """
    生成器形式的“数据加载器”。
    每次 yield: (input_ids: (B, T), labels: (B, T))
    """
    n = len(ds)
    indices = list(range(n))
    if shuffle:
        random.shuffle(indices)

    for start in range(0, n, batch_size):
        batch_idx = indices[start : start + batch_size]
        if len(batch_idx) < batch_size:
            # 最后一个不足 batch 的 batch 直接丢弃，避免梯度尺度混乱
            break
        xs, ys = [], []
        for idx in batch_idx:
            x, y = ds[idx]
            xs.append(x)
            ys.append(y)
        if isinstance(xs[0], torch.Tensor):
            x_batch = torch.stack(xs, dim=0)
            y_batch = torch.stack(ys, dim=0)
        else:
            x_batch = np.stack(xs, axis=0)
            y_batch = np.stack(ys, axis=0)
        yield x_batch, y_batch


# =============================================================================
# 训练主循环
# =============================================================================

def train(args):
    set_seed(args.seed)

    # ---- 1. tokenizer ----
    log(f"加载 tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 2. 数据集 ----
    log(f"加载数据集: {args.data_path}")
    ds = PretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        seed=args.seed,
    )
    log(f"样本总数: {len(ds)}, 每轮 batches ≈ {len(ds) // args.batch_size}")

    # ---- 3. 模型 ----
    log("构建 MiniMindForCausalLM ...")
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_seq_len,
        use_moe=bool(args.use_moe),
        dropout=args.dropout,
    )
    model = MiniMindForCausalLM(config)
    model.train()

    # 统计参数量
    total_params = 0
    for p in model.parameters():
        total_params += int(np.prod(np.asarray(p.data).shape))
    log(f"模型总参数: {total_params:,}  ≈ {total_params / 1e6:.2f} M")

    # ---- 4. 优化器 ----
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # ---- 5. 检查点恢复 ----
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    ckpt_path = os.path.join(args.save_dir, f"{args.save_weight}.pt")
    latest_path = os.path.join(args.save_dir, f"{args.save_weight}_latest.pt")
    os.makedirs(args.save_dir, exist_ok=True)

    if args.from_resume and os.path.exists(latest_path):
        log(f"从检查点恢复: {latest_path}")
        try:
            ckpt = torch.load(latest_path, map_location="cpu")
            # 加载权重（使用模型已有的 load_state_dict）
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            # optimizer 自己的状态（Adam 的 m/v 是内部的，我们做简单恢复）
            if "optimizer_m" in ckpt and "optimizer_v" in ckpt:
                assert len(optimizer.m) == len(ckpt["optimizer_m"])
                optimizer.m = [
                    m.to(device=optimizer.m[i].device)
                    if isinstance(optimizer.m[i], torch.Tensor) and isinstance(m, torch.Tensor)
                    else m
                    for i, m in enumerate(ckpt["optimizer_m"])
                ]
                optimizer.v = [
                    v.to(device=optimizer.v[i].device)
                    if isinstance(optimizer.v[i], torch.Tensor) and isinstance(v, torch.Tensor)
                    else v
                    for i, v in enumerate(ckpt["optimizer_v"])
                ]
                optimizer.t = int(ckpt.get("optimizer_t", 0))
            start_epoch = int(ckpt.get("epoch", 0))
            global_step = int(ckpt.get("global_step", 0))
            best_loss = float(ckpt.get("best_loss", float("inf")))
            log(f"  -> 恢复成功: epoch={start_epoch}, step={global_step}, best_loss={best_loss:.4f}")
        except Exception as e:
            log(f"  -> 检查点加载失败，将从头训练: {e}")

    # ---- 6. 学习率与步数 ----
    batches_per_epoch = max(1, len(ds) // args.batch_size)
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = min(total_steps // 20, max(1, int(args.warmup_ratio * total_steps)))
    log(
        f"总训练步数: {total_steps}, warmup: {warmup_steps}, "
        f"batch_size={args.batch_size}, accumulation={args.accumulation_steps}"
    )

    # =========================================================================
    # 训练循环
    # =========================================================================
    log("开始训练 ...")
    train_start = time.time()
    running_loss = 0.0
    running_aux = 0.0
    running_n = 0

    for epoch in range(start_epoch, args.epochs):
        # 每轮重洗一次样本顺序
        if epoch != start_epoch:
            random.seed(args.seed + epoch)
            ds = PretrainDataset(
                data_path=args.data_path,
                tokenizer=tokenizer,
                max_length=args.max_seq_len,
                seed=args.seed + epoch,
            )

        dl = build_dataloader_iter(ds, args.batch_size, shuffle=True)
        batch_i = 0
        for x_batch, y_batch in dl:
            # —— 1. 前向 ——
            outputs = model.forward(x_batch, labels=y_batch)
            loss = outputs["loss"]
            aux_loss = outputs.get("aux_loss", 0.0)

            # loss 是自定义 Tensor；取标量值用于打印
            if hasattr(loss, "data"):
                loss_val = float(loss.data)
            else:
                loss_val = float(loss)
            if hasattr(aux_loss, "data"):
                aux_val = float(aux_loss.data)
            else:
                aux_val = float(aux_loss) if aux_loss is not None else 0.0

            # 梯度累积: 按 1/accumulation_steps 缩放；但自实现 Tensor 已经 mean 过了
            scaled_loss = loss
            if args.accumulation_steps > 1:
                # 若 loss 是 Tensor，除以 accumulation_steps
                if hasattr(loss, "__truediv__"):
                    scaled_loss = loss / args.accumulation_steps
                else:
                    scaled_loss = loss / args.accumulation_steps

            # —— 2. 反向（在累积步内，保持 retain_graph，否则每步会 free graph）——
            if hasattr(scaled_loss, "backward"):
                retain = (batch_i + 1) % args.accumulation_steps != 0
                # 自实现 Tensor 的 backward 接受 retain_graph 参数（如不存在就不传）
                try:
                    scaled_loss.backward(retain_graph=False)
                except TypeError:
                    scaled_loss.backward()
            else:
                # 兜底：PyTorch tensor 方式
                scaled_loss.backward()

            running_loss += loss_val
            running_aux += aux_val
            running_n += 1
            batch_i += 1
            global_step += 1

            # —— 3. 优化器 step（每 accumulation_steps 一次）——
            if (batch_i % args.accumulation_steps == 0) or (batch_i == batches_per_epoch):
                # 设置学习率
                lr = get_lr_cosine(
                    global_step, total_steps, args.learning_rate,
                    min_lr_ratio=args.min_lr_ratio, warmup_steps=warmup_steps,
                )
                optimizer.lr = lr

                optimizer.step()
                optimizer.zero_grad(release_graph=True)

            # —— 4. 日志打印 ——
            if global_step % args.log_interval == 0:
                avg_loss = running_loss / max(1, running_n)
                avg_aux = running_aux / max(1, running_n)
                elapsed = time.time() - train_start
                speed = global_step / max(elapsed, 1e-9)
                eta_sec = (total_steps - global_step) / max(speed, 1e-9)
                log(
                    f"Epoch {epoch + 1}/{args.epochs}  step {global_step}/{total_steps}  "
                    f"loss={avg_loss:.4f}  aux_loss={avg_aux:.4f}  "
                    f"lr={lr:.6e}  speed={speed:.2f} step/s  "
                    f"ETA={eta_sec / 60:.1f} min"
                )
                running_loss = 0.0
                running_aux = 0.0
                running_n = 0

        # —— 5. epoch 结束保存最新检查点 + 最佳检查点 ——
        if running_n > 0:
            avg_loss = running_loss / running_n
        else:
            avg_loss = float("inf")
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_loss=best_loss,
            tag="latest",
            extra={"args": vars(args), "avg_loss": float(avg_loss)},
        )
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                best_loss=best_loss,
                tag="best",
                extra={"args": vars(args), "avg_loss": float(avg_loss)},
            )
            log(f"  ✅ 发现更优模型，loss={best_loss:.4f} -> 保存到 {ckpt_path}")

    log(f"训练完成，总用时: {(time.time() - train_start) / 60:.1f} min")


# =============================================================================
# 检查点保存
# =============================================================================

def save_checkpoint(path, model, optimizer, epoch, global_step, best_loss, tag="", extra=None):
    """保存模型权重 + 优化器状态。"""
    # 使用模型自有的 state_dict（这里的 MiniMindForCausalLM 没有现成 state_dict，
    # 所以按 parameters 顺序导出 torch tensor）
    param_states = {}
    for idx, p in enumerate(model.parameters()):
        if hasattr(p, "torch_tensor"):
            param_states[f"param_{idx}"] = p.torch_tensor.detach().cpu().clone()
        elif hasattr(p, "data"):
            param_states[f"param_{idx}"] = torch.from_numpy(
                np.asarray(p.data, dtype=np.float32).copy()
            )

    # 优化器状态
    m_states = [
        (m.detach().cpu() if isinstance(m, torch.Tensor) else torch.tensor(np.asarray(m, dtype=np.float32)))
        for m in optimizer.m
    ]
    v_states = [
        (v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(np.asarray(v, dtype=np.float32)))
        for v in optimizer.v
    ]

    ckpt = {
        "model_state_dict": param_states,
        "optimizer_m": m_states,
        "optimizer_v": v_states,
        "optimizer_t": optimizer.t,
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "tag": tag,
        "extra": extra,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


# =============================================================================
# 参数解析
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MiniMind 预训练（自实现框架）")
    p.add_argument("--save_dir", type=str, default=os.path.join(HERE, "out"),
                   help="模型保存目录")
    p.add_argument("--save_weight", type=str, default="minimind_pretrain",
                   help="保存权重的前缀")
    p.add_argument("--data_path", type=str,
                   default=os.path.join(HERE, "dataset", "pretrain_t2t_mini.jsonl"),
                   help="预训练数据路径 (jsonl)")
    p.add_argument("--tokenizer_path", type=str,
                   default=os.path.join(HERE, "dataset"),
                   help="tokenizer 目录")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accumulation_steps", type=int, default=4,
                   help="梯度累积步数，等效 batch = batch_size * accumulation_steps")
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1,
                   help="cosine 最终 lr 与初始 lr 的比例")
    p.add_argument("--warmup_ratio", type=float, default=0.03,
                   help="warmup 步数占总步数的比例")
    p.add_argument("--max_seq_len", type=int, default=256,
                   help="最大截断长度 (中文 1 token ≈ 1.5~1.7 字符)")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_hidden_layers", type=int, default=4)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--num_key_value_heads", type=int, default=2)
    p.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--log_interval", type=int, default=50,
                   help="每多少个 step 打印一次日志")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from_resume", type=int, default=1, choices=[0, 1],
                   help="是否自动检测并加载检查点继续训练 (0=否, 1=是)")
    return p.parse_args()


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    log("=" * 60)
    log(f"MiniMind 预训练启动")
    log("=" * 60)
    log(f"  save_dir       = {args.save_dir}")
    log(f"  data_path      = {args.data_path}")
    log(f"  tokenizer_path = {args.tokenizer_path}")
    log(f"  epochs={args.epochs}, batch_size={args.batch_size}, "
        f"accumulation_steps={args.accumulation_steps}")
    log(f"  lr={args.learning_rate}, min_lr_ratio={args.min_lr_ratio}, "
        f"warmup_ratio={args.warmup_ratio}")
    log(f"  max_seq_len={args.max_seq_len}, hidden_size={args.hidden_size}, "
        f"num_hidden_layers={args.num_hidden_layers}, "
        f"num_attention_heads={args.num_attention_heads}, "
        f"num_key_value_heads={args.num_key_value_heads}, "
        f"use_moe={args.use_moe}, dropout={args.dropout}")
    log("=" * 60)

    train(args)
