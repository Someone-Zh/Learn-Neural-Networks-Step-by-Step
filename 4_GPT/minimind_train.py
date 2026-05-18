import numpy as np
import json
import os
from transformers import AutoTokenizer
from minimind_model import MiniMindConfig, MiniMindSmallConfig, MiniMindForCausalLM
from optim.optimizer import Adam, ExponentialLR
from tensor.core import Tensor

# Tokenizer加载函数
def load_tokenizer(tokenizer_path=None):
    """
    加载预定义的tokenizer
    
    参数:
        tokenizer_path: tokenizer文件所在目录路径，默认为当前目录下的dataset文件夹
    
    返回:
        加载的tokenizer对象
    """
    if tokenizer_path is None:
        tokenizer_path = os.path.join(os.path.dirname(__file__), 'dataset')
    
    print(f"⏳ 从 {tokenizer_path} 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✅ Tokenizer加载完成，词汇量: {tokenizer.vocab_size}")
    
    return tokenizer


# 数据加载函数
def load_last_n_samples_from_jsonl(file_path, n=100):
    """从JSONL文件读取最后n条样本"""
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        last_lines = lines[-n:] if len(lines) >= n else lines
        
        for line in last_lines:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    samples.append(data['text'])
            except json.JSONDecodeError:
                continue
    
    return samples


def prepare_batch_data(samples, tokenizer, batch_size, seq_len):
    """准备批次数据"""
    input_ids_list = []
    labels_list = []
    
    for sample in samples:
        # 使用tokenizer编码文本
        encoded = tokenizer(
            sample, 
            max_length=seq_len + 1,
            truncation=True,
            padding='max_length',
            return_tensors=None  # 返回列表而不是tensor
        )
        
        full_ids = encoded['input_ids']
        
        if len(full_ids) < 2:
            continue
        
        # 创建input_ids和labels（因果关系：预测下一个token）
        input_ids = full_ids[:-1]  # 去掉最后一个token作为输入
        labels = full_ids[1:]      # 去掉第一个token作为标签
        
        # 将labels中的padding token设置为-100（忽略损失计算）
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        labels = [l if l != pad_token_id else -100 for l in labels]
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    # 如果数据不足batch_size，重复采样
    while len(input_ids_list) < batch_size:
        idx = np.random.randint(0, len(input_ids_list))
        input_ids_list.append(input_ids_list[idx].copy())
        labels_list.append(labels_list[idx].copy())
    
    # 转换为numpy数组
    input_ids = np.array(input_ids_list[:batch_size], dtype=np.int64)
    labels = np.array(labels_list[:batch_size], dtype=np.int64)
    
    return input_ids, labels


# 数据准备（原有函数保留）
def prepare_data(batch_size, seq_len, vocab_size):
    """
    生成随机训练数据
    """
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    labels = np.copy(input_ids)
    # 向右移动一位作为标签
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # 忽略最后一个位置的标签
    return input_ids, labels

# 训练函数
def train():
    # 配置
    config = MiniMindConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=1024,
        vocab_size=10000,
        max_position_embeddings=1024,
        dropout=0.1,
        use_moe=False
    )
    
    # 创建模型
    model = MiniMindForCausalLM(config)
    
    # 优化器
    params = model.parameters()
    optimizer = Adam(params, lr=1e-4)
    scheduler = ExponentialLR(optimizer, initial_lr=1e-4, gamma=0.999)
    
    # 训练参数
    batch_size = 8
    seq_len = 128
    epochs = 1000
    log_interval = 10
    
    print("开始训练...")
    for epoch in range(epochs):
        # 生成数据
        input_ids, labels = prepare_data(batch_size, seq_len, config.vocab_size)
        
        # 前向传播
        outputs = model.forward(input_ids, labels=labels)
        loss = outputs['loss']
        aux_loss = outputs['aux_loss']
        total_loss = loss + aux_loss if aux_loss > 0 else loss
        
        # 反向传播
        loss_tensor = model.forward(input_ids, labels=labels)['logits']
        # 创建一个标量损失用于反向传播
        loss_scalar = Tensor(np.array([loss]))
        loss_scalar.backward()
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        # 打印日志
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Aux Loss: {aux_loss:.4f}, Total Loss: {total_loss:.4f}")
    
    # 保存模型
    model.save('minimind_model.pkl')
    print("模型保存成功！")

# 测试生成
def test_generation():
    # 加载模型
    model = MiniMindForCausalLM.load('minimind_model.pkl')
    
    # 测试输入
    input_ids = np.array([[1]])  # 假讽1是bos_token_id
    
    # 生成文本
    generated = model.generate(input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9)
    print("生成结果:", generated)


# 小模型训练函数（使用真实数据）
def train_small_model():
    """使用小数据集和小模型进行快速验证训练"""
    print("="*60)
    print("开始小模型快速验证训练")
    print("="*60)
    
    # 加载预定义tokenizer（先加载tokenizer以获取正确的vocab_size）
    print("\n加载预定义tokenizer...")
    tokenizer = load_tokenizer()
    
    # 配置中等大小模型（使用tokenizer的实际词汇量）
    # 相比原来的小模型，参数量增加了约 8-10 倍
    config = MiniMindSmallConfig(
        hidden_size=256,              # 从 64 增加到 256 (4倍)
        num_hidden_layers=4,          # 从 2 增加到 4 (2倍)
        num_attention_heads=8,        # 从 2 增加到 8 (4倍)
        num_key_value_heads=4,        # 从 1 增加到 4 (4倍)
        intermediate_size=1024,       # 从 256 增加到 1024 (4倍)
        vocab_size=tokenizer.vocab_size,  # 使用tokenizer的实际词汇量 (6400)
        max_position_embeddings=512,  # 从 256 增加到 512 (2倍)
        dropout=0.1,
        use_moe=False
    )
    
    print(f"\n模型配置:")
    print(f"  - Hidden Size: {config.hidden_size}")
    print(f"  - Num Layers: {config.num_hidden_layers}")
    print(f"  - Num Attention Heads: {config.num_attention_heads}")
    print(f"  - Vocab Size: {config.vocab_size}")
    print(f"  - Max Position Embeddings: {config.max_position_embeddings}")
    
    # 创建模型
    print("\n创建模型...")
    model = MiniMindForCausalLM(config)
    
    # 加载数据（增加数据量）
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'pretrain_t2t_mini.jsonl')
    print(f"\n从 {dataset_path} 加载数据...")
    
    if not os.path.exists(dataset_path):
        print(f"错误: 数据集文件不存在: {dataset_path}")
        return
    
    # 增加数据量：从最后100条增加到500条
    samples = load_last_n_samples_from_jsonl(dataset_path, n=500)
    print(f"成功加载 {len(samples)} 条样本")
    print(f"词汇表大小: {config.vocab_size}")
    
    # 优化器
    params = model.parameters()
    optimizer = Adam(params, lr=1e-3)  # 提高学习率以加速收敛
    scheduler = ExponentialLR(optimizer, initial_lr=1e-3, gamma=0.995)
    
    # 训练参数
    batch_size = 8           # 从 4 增加到 8
    seq_len = 128            # 从 64 增加到 128
    epochs = 200             # 从 100 增加到 200
    log_interval = 10        # 从 5 增加到 10
    
    print(f"\n训练参数:")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Sequence Length: {seq_len}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Learning Rate: 1e-3")
    print(f"  - Log Interval: {log_interval}")
    print(f"  - Data Samples: {len(samples)}")
    
    # 计算模型参数量
    total_params = sum(p.data.size for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"参数量/样本数比率: {total_params/len(samples):.1f}")
    if total_params/len(samples) > 10000:
        print(f"⚠️  警告: 参数量远大于样本数，建议增加数据或减少模型大小")
    
    # 训练历史记录
    loss_history = []
    best_loss = float('inf')
    patience = 20  # 早停耐心值
    patience_counter = 0
    
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # 准备批次数据
        input_ids, labels = prepare_batch_data(samples, tokenizer, batch_size, seq_len)
        
        # 调试信息：第一个epoch打印数据形状
        if epoch == 0:
            print(f"\n调试信息:")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  labels shape: {labels.shape}")
            print(f"  labels unique values: {np.unique(labels)}")
            valid_count = np.sum((labels != -100) & (labels != 0))
            print(f"  valid labels count: {valid_count}")
        
        # 前向传播
        outputs = model.forward(input_ids, labels=labels)
        loss = outputs['loss']
        aux_loss = outputs['aux_loss']
        
        # 计算总损失
        if loss is not None:
            if hasattr(loss, 'data'):
                loss_value = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
            else:
                loss_value = float(loss)
        else:
            loss_value = 0.0
        
        total_loss = loss_value + (float(aux_loss) if aux_loss > 0 else 0)
        loss_history.append(total_loss)
        
        # 反向传播
        if loss is not None and hasattr(loss, 'backward'):
            # 如果loss是Tensor，直接backward
            try:
                loss.backward()
            except Exception as e:
                print(f"反向传播出错: {e}")
                print("跳过本次更新")
                optimizer.zero_grad()
                continue
        elif loss is not None:
            # 否则创建一个标量损失用于反向传播
            loss_scalar = Tensor(np.array([loss_value]))
            loss_scalar.backward()
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        # 打印日志
        if (epoch + 1) % log_interval == 0:
            avg_loss = np.mean(loss_history[-log_interval:])
            current_lr = optimizer.lr
            print(f"Epoch [{epoch+1:3d}/{epochs}], Loss: {loss_value:.4f}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            
            # 检查是否改善
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= patience:
                print(f"\n⚠️  早停触发！连续{patience}个epoch没有改善")
                print(f"最佳平均损失: {best_loss:.4f}")
                break
    
    print("-" * 60)
    print("\n训练完成！")
    print(f"最终平均损失: {np.mean(loss_history[-10:]):.4f}")
    print(f"最佳平均损失: {best_loss:.4f}")
    print(f"训练了 {len(loss_history)} 个epoch")
    
    # 保存模型
    model_path = 'minimind_small_model.pkl'
    model.save(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 测试生成（使用多个提示词）
    print("\n" + "="*60)
    print("测试模型生成能力")
    print("="*60)
    
    test_prompts = [
        "你好",
        "人工智能",
        "The future of",
        "Python is"
    ]
    
    for prompt in test_prompts:
        test_small_generation(model, tokenizer, prompt)
        print()
    
    return model, tokenizer, loss_history


def test_small_generation(model, tokenizer, prompt="你好"):
    """测试模型的文本生成能力"""
    print(f"\n提示词: {prompt}")
    
    # 使用tokenizer编码提示词
    encoded = tokenizer(prompt, return_tensors='np')
    input_ids = encoded['input_ids']
    
    print(f"输入IDs shape: {input_ids.shape}")
    
    try:
        # 生成文本（增加生成长度）
        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=50,      # 从 20 增加到 50
            temperature=0.8, 
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # 使用tokenizer解码生成的文本
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\n生成结果:")
        print("-" * 60)
        print(generated_text)
        print("-" * 60)
        
        # 计算生成的token数量
        new_tokens = len(generated_ids[0]) - len(input_ids[0])
        print(f"\n生成了 {new_tokens} 个新token")
        
    except Exception as e:
        print(f"生成时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # 如果命令行参数包含 "small"，则运行小模型训练
    if len(sys.argv) > 1 and sys.argv[1] == "small":
        train_small_model()
    else:
        # 默认运行原有训练
        print("使用默认训练模式（随机数据）")
        print("如需使用小模型快速验证，请运行: python minimind_train.py small\n")
        train()
        test_generation()