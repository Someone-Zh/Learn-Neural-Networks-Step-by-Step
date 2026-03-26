"""
词向量经典样例测试：国王 - 男人 + 女人 ≈ 女王
使用自定义Tensor框架实现Word2Vec风格的词嵌入训练
"""

import numpy as np
import sys
import os

# 将项目根目录添加到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensor.core import Tensor
from optim.optimizer import Adam


class Word2VecTrainer:
    """Word2Vec训练器"""
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01):
        """
        初始化训练器
        
        参数:
            vocab_size: 词汇表大小
            embedding_dim: 词向量维度
            learning_rate: 学习率
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 创建词嵌入矩阵 (使用Xavier初始化)
        scale = np.sqrt(1.0 / embedding_dim)
        self.target_embeddings = Tensor(np.random.uniform(-scale, scale, (vocab_size, embedding_dim)))
        self.context_embeddings = Tensor(np.random.uniform(-scale, scale, (vocab_size, embedding_dim)))
        
        # 优化器
        params = [self.target_embeddings, self.context_embeddings]
        self.optimizer = Adam(params, lr=learning_rate)
        
        # 词表
        self.word_to_idx = {}
        self.idx_to_word = {}
    
    def add_word(self, word):
        """添加词到词表"""
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
    
    def get_embedding(self, word):
        """获取词的嵌入向量（返回Tensor）"""
        idx = self.word_to_idx[word]
        # 直接使用numpy索引获取嵌入向量
        emb_data = self.target_embeddings.data[idx]
        return Tensor(emb_data)
    
    def get_embedding_numpy(self, word):
        """获取词的嵌入向量（返回numpy数组）"""
        idx = self.word_to_idx[word]
        return self.target_embeddings.data[idx]
    
    def cosine_similarity(self, vec1, vec2):
        """计算余弦相似度（接受numpy数组或Tensor）"""
        if isinstance(vec1, Tensor):
            v1 = vec1.data
        else:
            v1 = vec1
        if isinstance(vec2, Tensor):
            v2 = vec2.data
        else:
            v2 = vec2
            
        dot_product = np.sum(v1 * v2)
        norm1 = np.sqrt(np.sum(v1 ** 2))
        norm2 = np.sqrt(np.sum(v2 ** 2))
        return dot_product / (norm1 * norm2 + 1e-8)
    
    def find_nearest(self, vec, top_k=5):
        """找到最相似的词"""
        if isinstance(vec, Tensor):
            vec_data = vec.data
        else:
            vec_data = vec
            
        similarities = []
        for idx in range(min(self.vocab_size, len(self.word_to_idx))):
            other_vec = self.target_embeddings.data[idx]
            sim = self.cosine_similarity(vec_data, other_vec)
            similarities.append((self.idx_to_word[idx], sim))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_training_data():
    """创建训练数据 - 简单的上下文配对"""
    # 准备数据
    sentences = [
        "国王 管理 国家",
        "男人 治理 国家",
        "女人 统治 国家",
        "国王 是 君主",
        "男人 是 统治者",
        "女人 是 统治者",
        "国王 拥有 权力",
        "男人 拥有 权力",
        "女人 拥有 权力",
        "国王 很 尊贵",
        "男人 很 尊贵",
        "女人 很 尊贵",
        "国王 坐 在 王座",
        "男人 坐 在 椅子",
        "女人 坐 在 王座",
        "王后 是 女王",
        "女王 统治 王国",
        "王后 很 尊贵",
    ]
    
    # 收集所有词汇(构建词表)
    word_counts = {}
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    return sentences, word_counts


def train_word2vec():
    """训练词向量模型"""
    print("=" * 60)
    print("词向量训练 - 经典样例：国王 - 男人 + 女人 ≈ 女王")
    print("=" * 60)
    
    # 创建训练数据和词表
    sentences, word_counts = create_training_data()
    vocab = list(word_counts.keys())
    vocab_size = len(vocab)
    
    print(f"\n词汇表: {vocab}")
    print(f"词汇表大小: {vocab_size}")
    
    # 创建训练器
    embedding_dim = 16  # 使用较小的维度便于演示
    trainer = Word2VecTrainer(vocab_size, embedding_dim, learning_rate=0.1)
    
    # 添加词到训练器
    for word in vocab:
        trainer.add_word(word)
    
    # 准备训练数据：中心词-上下文配对
    window_size = 1
    training_pairs = []
    
    for sentence in sentences:
        words = sentence.split()
        for i, center_word in enumerate(words):
            # 获取上下文词
            context_indices = []
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    context_indices.append(trainer.word_to_idx[words[j]])
            
            for ctx_idx in context_indices:
                training_pairs.append((trainer.word_to_idx[center_word], ctx_idx))
    
    print(f"\n训练样本数: {len(training_pairs)}")
    
    # 训练
    epochs = 500
    print(f"\n开始训练 ({epochs} 轮)...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for center_idx, ctx_idx in training_pairs:
            # 获取中心词和上下文词的嵌入向量
            center_emb = trainer.target_embeddings.data[center_idx]  # (embedding_dim,)
            ctx_emb = trainer.context_embeddings.data[ctx_idx]  # (embedding_dim,)
            
            # 计算点积相似度
            similarity = np.dot(center_emb, ctx_emb)
            
            # 使用sigmoid将相似度转换为概率
            prob = 1.0 / (1.0 + np.exp(-similarity))
            
            # 目标：让相关词的对相似度接近1
            target = 1.0
            
            # 简单的损失函数 (MSE)
            loss = (prob - target) ** 2
            total_loss += loss
            
            # 手动计算梯度
            # d_loss/d_prob = 2 * (prob - target)
            # d_prob/d_similarity = prob * (1 - prob)
            # d_similarity/d_center_emb = ctx_emb
            # d_similarity/d_ctx_emb = center_emb
            
            d_loss_d_prob = 2 * (prob - target)
            d_prob_d_sim = prob * (1 - prob)
            d_sim_d_center = ctx_emb
            d_sim_d_ctx = center_emb
            
            grad_center = d_loss_d_prob * d_prob_d_sim * d_sim_d_center
            grad_ctx = d_loss_d_prob * d_prob_d_sim * d_sim_d_ctx
            
            # 累加梯度
            trainer.target_embeddings.grad[center_idx] += grad_center
            trainer.context_embeddings.grad[ctx_idx] += grad_ctx
        
        # 更新参数
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(training_pairs)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    print("\n训练完成!")
    return trainer, vocab


def test_word_analogy(trainer):
    """测试词向量类比：国王 - 男人 + 女人 ≈ 女王"""
    print("\n" + "=" * 60)
    print("测试词向量类比")
    print("=" * 60)
    
    # 检查词汇是否在词表中
    words_to_check = ["国王", "男人", "女人", "女王"]
    for word in words_to_check:
        if word in trainer.word_to_idx:
            print(f"[OK] '{word}' 在词表中 (索引: {trainer.word_to_idx[word]})")
        else:
            print(f"[X] '{word}' 不在词表中")
    
    if not all(word in trainer.word_to_idx for word in words_to_check):
        print("\n警告: 某些词不在训练词表中，跳过精确测试")
        return
    
    # 获取词向量（numpy数组）
    king = trainer.get_embedding_numpy("国王")
    man = trainer.get_embedding_numpy("男人")
    woman = trainer.get_embedding_numpy("女人")
    queen = trainer.get_embedding_numpy("女王")
    
    # 计算: 国王 - 男人 + 女人
    result = king - man + woman
    
    print(f"\n计算: 国王 - 男人 + 女人")
    
    # 找到最相似的词
    print("\n找到最相似的词:")
    nearest = trainer.find_nearest(result, top_k=5)
    for word, sim in nearest:
        print(f"  {word}: {sim:.4f}")
    
    # 计算与女王的相关度
    similarity_to_queen = trainer.cosine_similarity(result, queen)
    print(f"\n与 '女王' 的余弦相似度: {similarity_to_queen:.4f}")
    
    # 测试其他类比
    print("\n" + "-" * 40)
    print("其他词向量测试:")
    
    # 测试: 男人 - 女人 + 国王
    test1 = man - woman + king
    print(f"\n男人 - 女人 + 国王:")
    nearest1 = trainer.find_nearest(test1, top_k=3)
    for word, sim in nearest1:
        print(f"  {word}: {sim:.4f}")
    
    # 测试: 国王 - 王座 + 椅子
    if "王座" in trainer.word_to_idx and "椅子" in trainer.word_to_idx:
        throne = trainer.get_embedding_numpy("王座")
        chair = trainer.get_embedding_numpy("椅子")
        test2 = king - throne + chair
        print(f"\n国王 - 王座 + 椅子:")
        nearest2 = trainer.find_nearest(test2, top_k=3)
        for word, sim in nearest2:
            print(f"  {word}: {sim:.4f}")


def visualize_embeddings(trainer, vocab):
    """可视化词嵌入（打印嵌入向量）"""
    print("\n" + "=" * 60)
    print("词嵌入向量 (前3维)")
    print("=" * 60)
    
    for word in vocab:
        if word in trainer.word_to_idx:
            emb = trainer.get_embedding_numpy(word)
            vec_str = ", ".join([f"{v:.3f}" for v in emb[:3]])
            print(f"{word}: [{vec_str}...]")


def main():
    """主函数"""
    try:
        # 训练词向量
        trainer, vocab = train_word2vec()
        
        # 测试词向量类比
        test_word_analogy(trainer)
        
        # 可视化
        visualize_embeddings(trainer, vocab)
        
        print("\n" + "=" * 60)
        print("测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
