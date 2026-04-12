import torch
from safetensors.torch import save_file

# 加载你的 pth 模型
model_data = torch.load("C:\\Users\\Zh-So\\Downloads\\full_sft_768.pth", map_location="cpu")

# 如果是整个模型对象，需要提取 state_dict
if "state_dict" in model_data:
    weights = model_data["state_dict"]
else:
    weights = model_data

# 保存为 safetensors
save_file(weights, "minimindv3.safetensors")
print("转换完成！")