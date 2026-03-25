# 单元测试
import numpy as np
import unittest
from tensor.core import Tensor, GPT
from tensor.methods.embedding import TensorEmbedding
from tensor.methods.attention import TensorAttention
from optim.optimizer import Adam

class TestTensor(unittest.TestCase):
    """测试张量类"""
    
    def test_basic_operations(self):
        """测试基本运算"""
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        
        d = a * b
        self.assertEqual(d.data, 6.0)
        
        e = a ** 2
        self.assertEqual(e.data, 4.0)
    
    def test_matmul(self):
        """测试矩阵乘法"""
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a.matmul(b)
        expected = np.array([[19, 22], [43, 50]])
        np.testing.assert_array_equal(c.data, expected)
    
    def test_backward(self):
        """测试反向传播"""
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)
    
    def test_embedding(self):
        """测试嵌入操作"""
        embedding = Tensor.create_embedding(10, 5)
        indices = Tensor([0, 1, 2])
        output = embedding.embedding(indices)
        self.assertEqual(output.data.shape, (3, 5))
    
    def test_layer_norm(self):
        """测试层归一化"""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = x.layer_norm()
        mean = np.mean(output.data, axis=-1, keepdims=True)
        var = np.var(output.data, axis=-1, keepdims=True)
        np.testing.assert_allclose(mean, 0.0, atol=1e-6)
        np.testing.assert_allclose(var, 1.0, atol=1e-6)
    
    def test_multi_head_attention(self):
        """测试多头自注意力"""
        d_model = 8
        num_heads = 2
        batch_size = 2
        seq_len = 4
        
        # 创建注意力权重
        attn_weights = Tensor(np.random.randn(d_model, d_model * 3))
        
        # 创建输入
        x = Tensor(np.random.randn(batch_size, seq_len, d_model))
        
        # 测试注意力机制
        output = attn_weights.multi_head_attention(x, x, x, num_heads)
        self.assertEqual(output.data.shape, (batch_size, seq_len, d_model))
    
    def test_dropout(self):
        """测试Dropout"""
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output = x.dropout(p=0.5, training=True)
        # 确保输出形状正确
        self.assertEqual(output.data.shape, x.data.shape)

class TestGPT(unittest.TestCase):
    """测试GPT模型"""
    
    def test_gpt_forward(self):
        """测试GPT前向传播"""
        vocab_size = 100
        d_model = 16
        num_heads = 2
        hidden_dim = 32
        num_layers = 1
        dropout = 0.1
        
        # 创建模型
        model = GPT(vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout)
        
        # 创建输入
        batch_size = 2
        seq_len = 5
        x = Tensor(np.random.randint(0, vocab_size, size=(batch_size, seq_len)))
        
        # 测试前向传播
        output = model.forward(x, training=True)
        self.assertEqual(output.data.shape, (batch_size, seq_len, vocab_size))
    
    def test_gpt_parameters(self):
        """测试GPT参数"""
        vocab_size = 100
        d_model = 16
        num_heads = 2
        hidden_dim = 32
        num_layers = 1
        dropout = 0.1
        
        # 创建模型
        model = GPT(vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout)
        
        # 测试参数数量
        params = model.parameters()
        self.assertTrue(len(params) > 0)

class TestUtils(unittest.TestCase):
    """测试工具函数"""
    
    def test_positional_encoding(self):
        """测试位置编码"""
        seq_len = 10
        d_model = 8
        batch_size = 2
        
        pe = TensorEmbedding.positional_encoding(seq_len, d_model, batch_size)
        self.assertEqual(pe.data.shape, (batch_size, seq_len, d_model))
    
    def test_causal_mask(self):
        """测试因果掩码"""
        seq_len = 5
        batch_size = 2
        
        mask = TensorAttention.create_causal_mask(seq_len, batch_size)
        self.assertEqual(mask.data.shape, (batch_size, 1, seq_len, seq_len))
        # 确保上三角部分为负无穷
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                self.assertLess(mask.data[0, 0, i, j], -1e9)

if __name__ == '__main__':
    unittest.main()