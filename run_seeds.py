from train import train

if __name__ == "__main__":
    # 5个随机种子：用于多seed实验评估稳定性（mean ± std）
    seeds = [0, 1, 2, 3, 4]

    # ===================== Baseline PPO（对照组） =====================
    # 特点：
    # - 不使用GAE（use_gae=False）
    # - 优势估计：Monte Carlo return-to-go（在PPO.py里实现）
    # - 不做advantage normalization（normalize_adv=False）
    for s in seeds:
        train(
            random_seed=s,
            run_name="CartPole-v1_BASE",   # 输出目录名：PPO_logs/CartPole-v1_BASE, PPO_preTrained/CartPole-v1_BASE
            use_gae=False,
            gae_lambda=0.95,               # 对baseline无效，但为了接口统一保留
            normalize_adv=False
        )

    # ===================== Extended PPO（实验组） =====================
    # 特点（你的主要改动）：
    # - 使用GAE-λ 优势估计（use_gae=True）
    # - λ=0.95（常用的bias-variance折中）
    # - 启用advantage normalization（normalize_adv=True）提高训练稳定性
    for s in seeds:
        train(
            random_seed=s,
            run_name="CartPole-v1_GAE",    # 输出目录名：PPO_logs/CartPole-v1_GAE, PPO_preTrained/CartPole-v1_GAE
            use_gae=True,
            gae_lambda=0.95,
            normalize_adv=True
        )
