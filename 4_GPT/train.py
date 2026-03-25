# 训练脚本
import numpy as np
from tensor.core import GPT, Tensor
from optim.optimizer import Adam, StepLR
from config import Config

class DataLoader:
    """数据加载器
    
    用于加载和处理训练数据
    """
    def __init__(self, data, batch_size, seq_len):
        """
        初始化数据加载器
        
        参数:
            data: 训练数据，一维数组
            batch_size: 批次大小
            seq_len: 序列长度
        """
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = len(data) // (batch_size * seq_len)
        self.data = data[:self.num_batches * batch_size * seq_len]
        self.data = self.data.reshape(batch_size, -1)
    
    def __iter__(self):
        """迭代器
        
        生成批次数据
        """
        for i in range(0, self.data.shape[1] - self.seq_len, self.seq_len):
            x = self.data[:, i:i+self.seq_len]
            y = self.data[:, i+1:i+self.seq_len+1]
            yield Tensor(x), Tensor(y)
    
    def __len__(self):
        """
        返回批次数量
        """
        return self.num_batches

def train_gpt(model, data_loader, optimizer, scheduler, epochs, device='cpu'):
    """
    训练GPT模型
    
    参数:
        model: GPT模型
        data_loader: 数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 训练轮数
        device: 设备
    """
    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y) in enumerate(data_loader):
            # 前向传播
            logits = model.forward(x, training=True)
            
            # 计算损失
            # 这里使用简单的均方误差损失
            loss = (logits - y.embedding).sum() ** 2
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.data
            
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {total_loss / (i+1):.4f}, LR: {optimizer.lr:.6f}")
        
        print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(data_loader):.4f}")

def generate_text(model, start_token, max_length):
    """
    生成文本
    
    参数:
        model: GPT模型
        start_token: 起始 token
        max_length: 最大长度
        
    返回:
        生成的 token 序列
    """
    tokens = [start_token]
    for _ in range(max_length - 1):
        # 创建输入张量
        input_tensor = Tensor(np.array([tokens]))
        
        # 前向传播
        logits = model.forward(input_tensor, training=False)
        
        # 采样下一个 token
        next_token = np.argmax(logits.data[0, -1])
        tokens.append(next_token)
    
    return tokens

def compute_perplexity(model, data_loader):
    """
    计算模型的困惑度
    
    参数:
        model: GPT模型
        data_loader: 数据加载器
        
    返回:
        困惑度
    """
    total_loss = 0
    total_tokens = 0
    
    for x, y in data_loader:
        # 前向传播
        logits = model.forward(x, training=False)
        
        # 计算损失
        loss = (logits - y.embedding).sum() ** 2
        total_loss += loss.data
        total_tokens += x.data.size
    
    # 计算困惑度
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

if __name__ == "__main__":
    # 加载配置
    config = Config(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1,
        batch_size=32,
        seq_len=10,
        epochs=10,
        lr=0.001,
        step_size=100,
        gamma=0.1
    )
    
    # 准备数据
    data = np.random.randint(0, config.vocab_size, size=10000)
    
    # 创建数据加载器
    data_loader = DataLoader(data, config.batch_size, config.seq_len)
    
    # 创建模型
    model = GPT(
        config.vocab_size,
        config.d_model,
        config.num_heads,
        config.hidden_dim,
        config.num_layers,
        config.dropout
    )
    
    # 创建优化器和学习率调度器
    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )
    
    scheduler = StepLR(
        optimizer,
        initial_lr=config.lr,
        step_size=config.step_size,
        gamma=config.gamma
    )
    
    # 训练模型
    train_gpt(model, data_loader, optimizer, scheduler, config.epochs)
    
    # 计算困惑度
    perplexity = compute_perplexity(model, data_loader)
    print(f"Perplexity: {perplexity:.4f}")
    
    # 生成文本
    start_token = 0
    max_length = 50
    generated_tokens = generate_text(model, start_token, max_length)
    print(f"Generated tokens: {generated_tokens}")
    
    # 保存模型
    model.save("gpt_model.pkl")
    print("Model saved successfully")
    
    # 加载模型
    loaded_model = GPT.load("gpt_model.pkl")
    print("Model loaded successfully")
    
    # 测试加载的模型
    loaded_generated_tokens = generate_text(loaded_model, start_token, max_length)
    print(f"Loaded model generated tokens: {loaded_generated_tokens}")
# 训练脚本
import numpy as np
from tensor.core import GPT, Tensor
from optim.optimizer import Adam

class DataLoader:
    """数据加载器
    
    用于加载和处理训练数据
    """
    def __init__(self, data, batch_size, seq_len):
        """
        初始化数据加载器
        
        参数:
            data: 训练数据，一维数组
            batch_size: 批次大小
            seq_len: 序列长度
        """
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = len(data) // (batch_size * seq_len)
        self.data = data[:self.num_batches * batch_size * seq_len]
        self.data = self.data.reshape(batch_size, -1)
    
    def __iter__(self):
        """迭代器
        
        生成批次数据
        """
        for i in range(0, self.data.shape[1] - self.seq_len, self.seq_len):
            x = self.data[:, i:i+self.seq_len]
            y = self.data[:, i+1:i+self.seq_len+1]
            yield Tensor(x), Tensor(y)
    
    def __len__(self):
        """
        返回批次数量
        """
        return self.num_batches

def train_gpt(model, data_loader, optimizer, epochs, device='cpu'):
    """
    训练GPT模型
    
    参数:
        model: GPT模型
        data_loader: 数据加载器
        optimizer: 优化器
        epochs: 训练轮数
        device: 设备
    """
    for epoch in range(epochs):
        total_loss = 0
        for i, (x, y) in enumerate(data_loader):
            # 前向传播
            logits = model.forward(x, training=True)
            
            # 计算损失
            # 这里使用简单的均方误差损失
            loss = (logits - y.embedding).sum() ** 2
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
            
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {total_loss / (i+1):.4f}")
        
        print(f"Epoch {epoch+1}, Total Loss: {total_loss / len(data_loader):.4f}")

def generate_text(model, start_token, max_length):
    """
    生成文本
    
    参数:
        model: GPT模型
        start_token: 起始 token
        max_length: 最大长度
        
    返回:
        生成的 token 序列
    """
    tokens = [start_token]
    for _ in range(max_length - 1):
        # 创建输入张量
        input_tensor = Tensor(np.array([tokens]))
        
        # 前向传播
        logits = model.forward(input_tensor, training=False)
        
        # 采样下一个 token
        next_token = np.argmax(logits.data[0, -1])
        tokens.append(next_token)
    
    return tokens

if __name__ == "__main__":
    # 示例：训练一个简单的GPT模型
    # 准备数据
    vocab_size = 1000
    data = np.random.randint(0, vocab_size, size=10000)
    
    # 创建数据加载器
    batch_size = 32
    seq_len = 10
    data_loader = DataLoader(data, batch_size, seq_len)
    
    # 创建模型
    d_model = 128
    num_heads = 4
    hidden_dim = 256
    num_layers = 2
    dropout = 0.1
    
    model = GPT(vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout)
    
    # 创建优化器
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    epochs = 10
    train_gpt(model, data_loader, optimizer, epochs)
    
    # 生成文本
    start_token = 0
    max_length = 50
    generated_tokens = generate_text(model, start_token, max_length)
    print(f"Generated tokens: {generated_tokens}")
    
    # 保存模型
    model.save("gpt_model.pkl")
    
    # 加载模型
    loaded_model = GPT.load("gpt_model.pkl")
    print("Model loaded successfully")