import torch

# 1. 最关键的检查：PyTorch 能否找到 CUDA？
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# 2. 如果可用，检查有多少个 GPU 被识别
if torch.cuda.is_available():
    print(f"可用的 GPU 数量: {torch.cuda.device_count()}")

    # 3. 打印当前 GPU 的名称
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch 未找到可用的 CUDA 环境。")