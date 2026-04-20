import numpy as np
from minimind_model import MiniMindConfig, MiniMindForCausalLM
from optim.optimizer import Adam, ExponentialLR
from tensor.core import Tensor

# 数据准备
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
    input_ids = np.array([[1]])  # 假设1是bos_token_id
    
    # 生成文本
    generated = model.generate(input_ids, max_new_tokens=50, temperature=0.7, top_p=0.9)
    print("生成结果:", generated)

if __name__ == "__main__":
    train()
    test_generation()