import torch
import time

# 1. 加载模型
checkpoint_path = '../logs/rsl_rl/cartpole_direct/2025-11-11_10-21-55/exported/policy.pt'
device = torch.device('cpu') #('cuda' if torch.cuda.is_available() else 'cpu')

policy = torch.jit.load(checkpoint_path, map_location=device)
policy.to(device)
policy.eval()
# 打印模型结构确认
print(policy)

# 输入维度（由上次日志可知 62
x = torch.zeros((1, 59), device=device)

with torch.no_grad():
    for i in range(10):
        y = policy(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Output tensor:", y)

# 正式测量
N = 10000  # 重复次数
torch.set_grad_enabled(False)
start = time.perf_counter()
for _ in range(N):
    _ = policy(x)
end = time.perf_counter()

avg_time = (end - start) / N
freq = 1.0 / avg_time

print(f"平均每次推理耗时: {avg_time * 1000:.3f} ms")
print(f"等效推理频率: {freq:.1f} Hz")