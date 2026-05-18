"""
对比测试：预定义tokenizer vs 自定义字符级tokenizer
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer
import numpy as np

# 导入原来的SimpleCharTokenizer（从备份或注释中提取）
class SimpleCharTokenizer:
    """简单的字符级tokenizer，用于中文文本处理"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.pad_token = 0
        self.unk_token = 1
        self.bos_token = 2
        self.eos_token = 3
        
        self.char_to_id = {}
        self.id_to_char = {}
        
        self.char_to_id['<PAD>'] = self.pad_token
        self.char_to_id['<UNK>'] = self.unk_token
        self.char_to_id['<BOS>'] = self.bos_token
        self.char_to_id['<EOS>'] = self.eos_token
        
        self.id_to_char[self.pad_token] = '<PAD>'
        self.id_to_char[self.unk_token] = '<UNK>'
        self.id_to_char[self.bos_token] = '<BOS>'
        self.id_to_char[self.eos_token] = '<EOS>'
        
        self.next_id = 4
    
    def build_vocab(self, texts):
        """从文本列表构建词汇表"""
        char_count = {}
        for text in texts:
            for char in text:
                char_count[char] = char_count.get(char, 0) + 1
        
        sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
        
        for char, count in sorted_chars:
            if self.next_id >= self.vocab_size:
                break
            if char not in self.char_to_id:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
    
    def encode(self, text, max_length=None):
        """将文本编码为id序列"""
        ids = [self.bos_token]
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            else:
                ids.append(self.unk_token)
        ids.append(self.eos_token)
        
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [self.pad_token] * (max_length - len(ids))
        
        return ids
    
    def decode(self, ids):
        """将id序列解码为文本"""
        chars = []
        for id in ids:
            if id == self.eos_token:
                break
            if id not in [self.pad_token, self.bos_token]:
                chars.append(self.id_to_char.get(id, '<UNK>'))
        return ''.join(chars)


def compare_tokenizers():
    """对比两种tokenizer"""
    print("="*80)
    print("Tokenizer 对比测试")
    print("="*80)
    
    # 测试文本
    test_texts = [
        "Hello World!",
        "你好，世界！",
        "The quick brown fox jumps over the lazy dog",
        "人工智能正在改变我们的生活方式",
        "Python is a great programming language for machine learning"
    ]
    
    # 加载预定义tokenizer
    print("\n1. 加载预定义tokenizer...")
    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(__file__), 'dataset')
    )
    print(f"   ✓ 词汇量: {pretrained_tokenizer.vocab_size}")
    print(f"   ✓ PAD token: {pretrained_tokenizer.pad_token} (ID: {pretrained_tokenizer.pad_token_id})")
    print(f"   ✓ EOS token: {pretrained_tokenizer.eos_token} (ID: {pretrained_tokenizer.eos_token_id})")
    
    # 创建自定义tokenizer
    print("\n2. 创建自定义字符级tokenizer...")
    custom_tokenizer = SimpleCharTokenizer(vocab_size=1000)
    custom_tokenizer.build_vocab(test_texts)
    print(f"   ✓ 词汇量: {len(custom_tokenizer.char_to_id)}")
    
    print("\n" + "="*80)
    print("编码对比")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n测试文本 {i}: \"{text}\"")
        print("-" * 80)
        
        # 预定义tokenizer
        pretrained_encoded = pretrained_tokenizer(text, return_tensors=None)
        pretrained_ids = pretrained_encoded['input_ids']
        pretrained_decoded = pretrained_tokenizer.decode(pretrained_ids, skip_special_tokens=True)
        
        print(f"预定义tokenizer:")
        print(f"  Token数量: {len(pretrained_ids)}")
        print(f"  Token IDs: {pretrained_ids[:10]}{'...' if len(pretrained_ids) > 10 else ''}")
        print(f"  解码结果: \"{pretrained_decoded}\"")
        
        # 自定义tokenizer
        custom_ids = custom_tokenizer.encode(text, max_length=None)
        custom_decoded = custom_tokenizer.decode(custom_ids)
        
        print(f"\n自定义tokenizer:")
        print(f"  Token数量: {len(custom_ids)}")
        print(f"  Token IDs: {custom_ids[:10]}{'...' if len(custom_ids) > 10 else ''}")
        print(f"  解码结果: \"{custom_decoded}\"")
        
        # 对比
        reduction = ((len(custom_ids) - len(pretrained_ids)) / len(custom_ids) * 100) if len(custom_ids) > 0 else 0
        print(f"\n  📊 序列长度减少: {reduction:.1f}% ({len(custom_ids)} → {len(pretrained_ids)})")
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("\n预定义tokenizer的优势:")
    print("  ✓ 使用子词分词（BPE），能更好地处理常见词组")
    print("  ✓ 序列长度更短，训练效率更高")
    print("  ✓ 词汇量更大（6400 vs ~100），表达能力更强")
    print("  ✓ 已在大量数据上预训练，泛化能力更好")
    print("  ✓ 与Hugging Face生态系统兼容")
    print("\n自定义tokenizer的特点:")
    print("  • 简单直观，易于理解")
    print("  • 完全控制词汇表构建过程")
    print("  • 适合特定领域的定制化需求")
    print("  • 但序列长度较长，训练效率较低")
    print("="*80)


if __name__ == "__main__":
    compare_tokenizers()
