import torch
from transformers import AutoTokenizer, TextStreamer

# ======================
# 路径
# ======================
MODEL_PATH    = r"4_GPT/dataset/full_sft_768.pth"
TOKENIZER_PATH = r"4_GPT/dataset/"

# ======================
# 设备
# ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用设备: {device}")

# ======================
# 分词器
# ======================
print("⏳ 加载本地分词器...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print(f"✅ 分词器加载完成，词汇量: {tokenizer.vocab_size}")

from minimind_model import MiniMindConfig, MiniMindForCausalLM

# ======================
# 模型
# ======================
print("⏳ 构建模型并加载权重...")
config = MiniMindConfig(
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=tokenizer.vocab_size,
    flash_attn=False,
    use_moe=False
)

model = MiniMindForCausalLM(config)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("✅ 模型加载成功！")

# ======================
# 对话函数（和官方完全一致）
# ======================
def chat(prompt):
    # ✅ 关键 1：对话模板（模型不乱说的核心）
    conversation = [{"role": "system", "content": "你是一个专业的问答助手。"},
                   {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(inputs, return_tensors="pt").to(device)

    print("🤖:", end=" ")

    # ✅ 关键 2：完全对齐官方生成参数
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            streamer=streamer,       # ✅ 关键3：流式输出
            temperature=0.85,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False  # 暂时保持 False，因为我们还没有完全修复 cache 逻辑，但核心功能已正常
        )

# ======================
# 测试
# ======================
if __name__ == "__main__":
    prompt = "给我段python读取文件的代码"
    print(f"\n🧑 你: {prompt}")
    chat(prompt)