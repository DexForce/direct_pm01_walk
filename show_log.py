import os
import re
import math
import matplotlib.pyplot as plt

LOG_FILE = "log.txt"
SKIP_LINES = 3000       # 跳过日志开头的前3000行
COLUMNS = 4             # 每行显示的子图数量
FIGSIZE_PER_ROW = 4.0   # 每行垂直空间

# 匹配 reward/penalty 行，例如：
# gait_phase_reward: -0.097 	 weighted: -0.005
pattern_reward = re.compile(r"([\w_]+):\s*[-+]?\d*\.?\d*\s+weighted:\s*([-+]?\d*\.?\d+)")
# 匹配 Mean episode length 行
pattern_episode = re.compile(r"Mean episode length:\s*([-+]?\d*\.?\d+)")

def parse_log(file_path, skip_lines=0):
    """解析日志文件，返回 {metric_name: [values]}"""
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        # 跳过前 skip_lines 行
        for _ in range(skip_lines):
            next(f, None)

        for line in f:
            # 奖励或惩罚行
            for name, val in pattern_reward.findall(line):
                try:
                    v = float(val)
                except ValueError:
                    continue
                data.setdefault(name, []).append(v)

            # 解析 Mean episode length
            match_ep = pattern_episode.search(line)
            if match_ep:
                try:
                    v = float(match_ep.group(1))
                    data.setdefault("mean_episode_length", []).append(v)
                except ValueError:
                    pass
    return data


def plot_metrics(data):
    """绘制所有 weighted 曲线，每个子图一个指标"""
    n = len(data)
    if n == 0:
        print("⚠️ 未找到任何 weighted 数据。")
        return

    rows = math.ceil(n / COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(16, FIGSIZE_PER_ROW * rows))
    fig.suptitle("IsaacLab Weighted Rewards / Penalties + Mean Episode Length", fontsize=14, y=0.995)
    axes = axes.flatten()

    for i, (name, values) in enumerate(sorted(data.items())):
        ax = axes[i]
        ax.plot(values, color="C0", linewidth=1.2)
        ax.set_title(name, fontsize=10, pad=6)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("Step Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        if len(values) > 0:
            ymin, ymax = min(values), max(values)
            pad = (ymax - ymin) * 0.2 + 1e-3
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_xlim(0, len(values))
        ax.tick_params(labelsize=8)

    # 清除多余的空子图
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3.2, w_pad=1.0)
    plt.show()


def main():
    if not os.path.exists(LOG_FILE):
        print("❌ 未找到 log.txt")
        return

    data = parse_log(LOG_FILE, skip_lines=SKIP_LINES)
    if not data:
        print("⚠️ 日志中未找到任何 weighted 或 mean_episode_length 数据。")
        return

    print(f"✅ 解析完成，共 {len(data)} 个指标，已跳过前 {SKIP_LINES} 行。")
    plot_metrics(data)


if __name__ == "__main__":
    main()
