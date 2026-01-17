# PPO 在 CartPole-v1 上的复现与改进（GAE-λ）

本项目基于 **PPO-PyTorch** 代码框架，在 CartPole-v1 环境上复现了 Proximal Policy Optimization（PPO）算法，并在此基础上实现了以下改进：

- 使用 **Generalized Advantage Estimation（GAE-λ）**
- 引入 **Advantage Normalization**
- 采用 **5 个随机种子（multi-seed）** 进行稳定性与鲁棒性评估

该项目用于《Reinforcement Learning》课程的复现实验与分析报告。

------

## 一、项目目标

本项目的主要目标包括：

1. 复现标准 PPO（Baseline PPO）在 CartPole-v1 上的训练过程
2. 在 PPO 中引入 GAE-λ 优势估计方法
3. 对优势函数进行标准化以提高训练稳定性
4. 使用 5 个不同随机种子进行多次独立实验
5. 对比 Baseline PPO 与改进 PPO 在训练稳定性和鲁棒性上的差异

------

## 二、项目结构说明

```
.
├── PPO.py                # PPO 核心算法实现（baseline + GAE）
├── train.py              # 单次训练脚本（可配置 baseline / GAE）
├── run_seeds.py          # 多随机种子（5 seeds）训练入口
├── plot_mean_std.py      # 多 seed 结果的 mean ± std 绘图脚本
├── test.py               # 加载已训练模型进行测试评估
├── PPO_logs/             # 训练日志（CSV，默认不提交到 GitHub）
├── PPO_preTrained/       # 训练好的模型权重（默认不提交）
└── README.md             # 项目说明文档
```

说明：

- `PPO_logs/` 和 `PPO_preTrained/` 为程序运行时自动生成目录
- 所有实验结果均可通过重新运行代码复现

------

## 三、实验环境与超参数

- 环境：CartPole-v1（OpenAI Gym）
- 动作空间：离散（2 个动作）
- 最大训练步数：200,000
- 最大 episode 长度：400
- PPO 更新轮数（K）：80
- 折扣因子 γ：0.99
- Clip 参数 ε：0.2
- Actor 学习率：3e-4
- Critic 学习率：1e-3
- GAE-λ：0.95（仅在 GAE 模式下启用）
- 随机种子：{0, 1, 2, 3, 4}

------

## 四、Baseline PPO 与改进 PPO 的区别

### 1. Baseline PPO

- 使用 **Monte Carlo return-to-go** 计算优势函数
- 不进行 advantage normalization
- 对随机初始化较为敏感
- 个别随机种子下可能出现训练不稳定或提前收敛失败

### 2. PPO + GAE-λ + Advantage Normalization

- 使用 **GAE-λ（λ = 0.95）** 进行优势估计
- 在 PPO 更新前对优势函数进行标准化
- 有效降低优势估计的方差
- 在多随机种子下表现出更稳定、更一致的收敛行为

------

## 五、如何运行代码

### 1. 单次训练

```
python train.py
```

默认设置：

- 使用 GAE-λ
- 启用 advantage normalization
- 随机种子为 0

------

### 2. 多随机种子训练

```
python run_seeds.py
```

该脚本会自动完成：

- Baseline PPO（5 个随机种子）
- 改进 PPO（GAE，5 个随机种子）
- 每个 seed 的结果以 CSV 形式保存到 `PPO_logs/` 目录

------

### 3. 绘制多 seed mean ± std 曲线

```
python plot_mean_std.py
```

功能说明：

- 读取同一配置下多个 seed 的 CSV 日志
- 对齐 timestep
- 计算 mean 与 standard deviation
- 绘制 mean ± std 学习曲线

------

### 4. 测试已训练模型

```
python test.py
```

功能说明：

- 从 `PPO_preTrained/` 中加载指定 seed 的模型
- 运行多个测试 episode
- 输出每个 episode 的 reward 以及平均 reward

------

## 六、实验结论概述

- Baseline PPO 在大多数随机种子下能够解决 CartPole-v1，但稳定性有限
- 引入 GAE-λ 与 advantage normalization 后：
  - 所有随机种子均能稳定收敛
  - 训练过程更加平滑
  - 对随机初始化的敏感性显著降低

改进方法主要提升的是 **训练稳定性与鲁棒性**，而非最终性能上限。

------

## 七、参考资料

- Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, arXiv:1506.02438
- PPO-PyTorch：https://github.com/nikhilbarhate99/PPO-PyTorch
- CleanRL：https://github.com/vwxyzjn/cleanrl
