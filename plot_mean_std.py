import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_seed_csvs(folder: str):
    # 读取某个log文件夹下所有seed的CSV日志（每个seed一个CSV）
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")
    # 每个CSV里通常包含：episode,timestep,reward
    dfs = [pd.read_csv(f) for f in files]
    return dfs, files


def align_on_common_timesteps(dfs):
    # ====== 关键点：对齐不同seed的时间轴 ======
    # 因为不同seed可能在某些timestep点记录不完全一致，
    # 这里取所有seed timesteps的交集（intersection），保证能安全对齐。
    common = set(dfs[0]["timestep"].tolist())
    for df in dfs[1:]:
        common &= set(df["timestep"].tolist())
    common = np.array(sorted(list(common)))

    # 将每个seed在这些“共同timestep”上的reward取出来，堆成矩阵
    rewards = []
    for df in dfs:
        # 以timestep为索引，按common顺序取reward
        s = df.set_index("timestep").loc[common]["reward"].to_numpy()
        rewards.append(s)

    # rewards shape: [num_seeds, T]
    rewards = np.vstack(rewards)
    return common, rewards


def plot_mean_std(log_folder: str, title: str, save_path: str = None):
    # 读取所有seed日志
    dfs, files = load_seed_csvs(log_folder)

    # 统一对齐时间步后，得到矩阵 rewards[seed, t]
    timesteps, rewards = align_on_common_timesteps(dfs)

    # 对seed维度求均值与标准差：用于画 mean ± std
    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)

    # 绘图：均值曲线 + 标准差阴影带
    plt.figure()
    plt.title(title)
    plt.plot(timesteps, mean)
    plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.grid(True)

    # 可选保存图片（用于LaTeX报告插图）
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    # 展示图像（本地运行时会弹出窗口）
    plt.show()

    # 打印实际使用了哪些CSV文件，方便复现实验与排查
    print("Used files:")
    for f in files:
        print(" -", f)


if __name__ == "__main__":
    # baseline PPO：多seed统计图
    plot_mean_std(
        log_folder="PPO_logs/CartPole-v1_BASE",
        title="CartPole-v1 Baseline PPO (mean ± std over 5 seeds)",
        save_path="PPO_figs/CartPole-v1_BASE/mean_std.png"
    )

    # extended PPO：GAE + advantage normalization 多seed统计图
    plot_mean_std(
        log_folder="PPO_logs/CartPole-v1_GAE",
        title="CartPole-v1 PPO + GAE (mean ± std over 5 seeds)",
        save_path="PPO_figs/CartPole-v1_GAE/mean_std.png"
    )
