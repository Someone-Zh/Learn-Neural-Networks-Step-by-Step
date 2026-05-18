"""
测试使用预定义tokenizer的训练流程
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from minimind_train import load_tokenizer, load_last_n_samples_from_jsonl, prepare_batch_data
import numpy as np

def test_tokenizer_loading():
    """测试tokenizer加载"""
    print("="*60)
    print("测试1: 加载预定义tokenizer")
    print("="*60)
    
    tokenizer = load_tokenizer()
    print(f"✓ Tokenizer加载成功")
    print(f"  - 词汇量: {tokenizer.vocab_size}")
    print(f"  - PAD token ID: {tokenizer.pad_token_id}")
    print(f"  - EOS token ID: {tokenizer.eos_token_id}")
    print(f"  - BOS token ID: {tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 'N/A'}")
    print()
    
    return tokenizer

def test_tokenizer_encoding(tokenizer):
    """测试tokenizer编码解码"""
    print("="*60)
    print("测试2: Tokenizer编码解码")
    print("="*60)
    
    test_text = "你好，世界！Hello World!"
    print(f"原始文本: {test_text}")
    
    # 编码
    encoded = tokenizer(test_text, return_tensors=None)
    print(f"编码后IDs: {encoded['input_ids'][:10]}... (显示前10个)")
    print(f"编码长度: {len(encoded['input_ids'])}")
    
    # 解码
    decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
    print(f"解码后文本: {decoded}")
    print()

def test_data_loading():
    """测试数据加载"""
    print("="*60)
    print("测试3: 加载数据集")
    print("="*60)
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset', 'pretrain_t2t_mini.jsonl')
    print(f"数据集路径: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"✗ 数据集文件不存在: {dataset_path}")
        return None
    
    samples = load_last_n_samples_from_jsonl(dataset_path, n=10)
    print(f"✓ 成功加载 {len(samples)} 条样本")
    print(f"  - 第一条样本: {samples[0][:50]}...")
    print()
    
    return samples

def test_batch_preparation(tokenizer, samples):
    """测试批次数据准备"""
    print("="*60)
    print("测试4: 准备批次数据")
    print("="*60)
    
    if samples is None:
        print("✗ 没有可用的样本数据")
        return
    
    batch_size = 4
    seq_len = 32
    
    input_ids, labels = prepare_batch_data(samples, tokenizer, batch_size, seq_len)
    
    print(f"✓ 批次数据准备成功")
    print(f"  - input_ids shape: {input_ids.shape}")
    print(f"  - labels shape: {labels.shape}")
    print(f"  - input_ids dtype: {input_ids.dtype}")
    print(f"  - labels dtype: {labels.dtype}")
    print(f"  - labels中-100的数量: {np.sum(labels == -100)}")
    print(f"  - labels中有效标签数量: {np.sum((labels != -100) & (labels != 0))}")
    print()
    
    # 显示第一个样本的详细信息
    print("第一个样本详情:")
    print(f"  input_ids: {input_ids[0]}")
    print(f"  labels: {labels[0]}")
    
    # 解码第一个样本
    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"  解码后的输入: {decoded_input[:50]}...")
    print()

def main():
    """运行所有测试"""
    print("\n开始测试使用预定义tokenizer的训练流程\n")
    
    try:
        # 测试1: 加载tokenizer
        tokenizer = test_tokenizer_loading()
        
        # 测试2: 编码解码
        test_tokenizer_encoding(tokenizer)
        
        # 测试3: 加载数据
        samples = test_data_loading()
        
        # 测试4: 准备批次数据
        test_batch_preparation(tokenizer, samples)
        
        print("="*60)
        print("所有测试完成！✓")
        print("="*60)
        print("\n现在可以运行小模型训练:")
        print("python minimind_train.py small")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
