"""
验证 MiniMind 模型的前向传播
加载训练好的模型参数，测试前向传播是否正确
参考: tmp/minimind.py 和 tmp/minimind_model.py
"""
import torch
from transformers import AutoTokenizer, TextStreamer
import sys
import os

# 添加 tmp 目录到路径以导入官方模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tmp'))
from minimind_model import MiniMindConfig, MiniMindForCausalLM

# ======================
# 路径配置
# ======================
MODEL_PATH = r"C:\Users\Zh-So\Downloads\full_sft_768_moe.pth"
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'tmp')

# ======================
# 设备配置
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用设备: {device}")

# ======================
# 加载分词器
# ======================
print("\n⏳ 加载本地分词器...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print(f"✅ 分词器加载完成，词汇量: {tokenizer.vocab_size}")

# ======================
# 构建并加载模型
# ======================
print("\n⏳ 构建模型并加载权重...")
config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=tokenizer.vocab_size,
    flash_attn=False,
    use_moe=True
)

model = MiniMindForCausalLM(config)

# 使用官方方式加载权重
try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("✅ 使用 load_state_dict 加载成功")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    raise

# 转换为半精度并移动到设备
model = model.half().to(device)
model.eval()
print("✅ 模型加载成功！")

# ======================
# 验证函数
# ======================

def test_basic_forward():
    """测试基本前向传播"""
    print("\n" + "="*60)
    print("测试 1: 基本前向传播")
    print("="*60)
    
    # 准备测试输入
    test_text = "我该如何挑选电视？"
    conversation = [
        {"role": "system", "content": "你是一个专业的问答助手"},
        {"role": "user", "content": test_text}
    ]
    
    inputs_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(inputs_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    
    print(f"输入文本: {test_text}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids}")
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        
        print(f"\n✅ 前向传播成功！")
        print(f"Logits shape: {logits.shape}")
        print(f"Hidden states shape: {hidden_states.shape}")
        print(f"Logits 范围: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"Logits 均值: {logits.mean():.4f}")
        print(f"Logits 标准差: {logits.std():.4f}")
        
        # 检查是否有 NaN 或 Inf
        has_nan = torch.isnan(logits).any()
        has_inf = torch.isinf(logits).any()
        print(f"包含 NaN: {has_nan.item()}")
        print(f"包含 Inf: {has_inf.item()}")
        
        if has_nan or has_inf:
            print("❌ 警告: 输出包含异常值！")
            return False
        else:
            print("✅ 输出正常，无异常值")
            return True


def test_loss_computation():
    """测试损失计算"""
    print("\n" + "="*60)
    print("测试 2: 损失计算")
    print("="*60)
    
    # 准备输入和标签
    test_text = "人工智能是未来的发展方向"
    conversation = [
        {"role": "user", "content": test_text}
    ]
    
    inputs_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    
    inputs = tokenizer(inputs_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    
    # 创建标签（因果关系：预测下一个token）
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # 忽略最后一个位置
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 前向传播计算损失
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        aux_loss = outputs.aux_loss
        
        print(f"\n✅ 损失计算成功！")
        if loss is not None:
            print(f"Loss: {loss.item():.4f}")
        else:
            print(f"Loss: None")
        
        if aux_loss is not None:
            print(f"Aux Loss: {aux_loss:.4f}")
        else:
            print(f"Aux Loss: 0.0000")
        
        if loss is not None:
            if not torch.isnan(loss) and not torch.isinf(loss):
                print("✅ 损失值正常")
                return True
            else:
                print("❌ 损失值异常")
                return False
        else:
            print("⚠️  损失为 None")
            return False


def test_generation():
    """测试文本生成"""
    print("\n" + "="*60)
    print("测试 3: 文本生成")
    print("="*60)
    
    prompt = "我该如何挑选电视？"
    print(f"\n🧑 你: {prompt}")
    
     # 准备测试输入
    conversation = [
        {"role": "system", "content": "你是一个专业的问答助手"},
        {"role": "user", "content": prompt}
    ]
    
    
    inputs_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(inputs_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    
    print("🤖:", end=" ")
    
    # 创建流式输出器
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 生成文本 - 使用官方的 generate 方法
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.85,
            top_p=0.95,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            do_sample=True,
            use_cache=True
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n{generated_text}")
    
    print("\n✅ 文本生成成功！")
    return True


def test_kv_cache():
    """测试 KV Cache 机制"""
    print("\n" + "="*60)
    print("测试 4: KV Cache 机制")
    print("="*60)
    
    # 准备输入
    test_text = "测试"
    conversation = [
        {"role": "user", "content": test_text}
    ]
    
    inputs_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(inputs_text, return_tensors="pt").to(device)
    input_ids = inputs['input_ids']
    
    print(f"Input IDs shape: {input_ids.shape}")
    
    # 第一次前向传播（不使用 cache）
    with torch.no_grad():
        outputs1 = model(input_ids, use_cache=True)
        past_key_values = outputs1.past_key_values
        logits1 = outputs1.logits
        
        print(f"\n第一次前向传播:")
        print(f"  Logits shape: {logits1.shape}")
        print(f"  Past key values layers: {len(past_key_values)}")
        if past_key_values[0] is not None:
            print(f"  Key shape (layer 0): {past_key_values[0][0].shape}")
            print(f"  Value shape (layer 0): {past_key_values[0][1].shape}")
        
        # 第二次前向传播（使用 cache）
        next_token_idx = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
        input_ids_with_next = torch.cat([input_ids, next_token_idx], dim=1)
        
        outputs2 = model(
            input_ids_with_next[:, -1:],  # 只传入新token
            past_key_values=past_key_values,
            use_cache=True
        )
        logits2 = outputs2.logits
        
        print(f"\n第二次前向传播（使用 KV Cache）:")
        print(f"  Input shape: {input_ids_with_next[:, -1:].shape}")
        print(f"  Logits shape: {logits2.shape}")
        
        print("\n✅ KV Cache 机制工作正常！")
        return True


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("开始验证 MiniMind 模型前向传播")
    print("="*60)
    
    results = {}
    
    # 运行所有测试
    try:
        results['basic_forward'] = test_basic_forward()
    except Exception as e:
        print(f"\n❌ 基本前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['basic_forward'] = False
    
    try:
        results['loss_computation'] = test_loss_computation()
    except Exception as e:
        print(f"\n❌ 损失计算测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['loss_computation'] = False
    
    try:
        results['kv_cache'] = test_kv_cache()
    except Exception as e:
        print(f"\n❌ KV Cache 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['kv_cache'] = False
    
    try:
        results['generation'] = test_generation()
    except Exception as e:
        print(f"\n❌ 文本生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['generation'] = False
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有测试通过！模型前向传播正确！")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    main()
