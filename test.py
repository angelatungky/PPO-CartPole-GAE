import time
import gym
import numpy as np
import torch

from PPO import PPO


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################
    env_name = "CartPole-v1"          # Gym环境ID（必须和训练一致）
    run_name = "CartPole-v1_GAE"      # 选择测试哪个训练输出的文件夹：BASE 或 GAE

    has_continuous_action_space = False
    max_ep_len = 400                  # 测试时每回合最多步数（应与训练设置一致）
    action_std = None                 # 离散动作空间不需要动作std（连续动作才用）

    render = True                     # 是否渲染窗口（服务器/无GUI环境建议False）
    frame_delay = 0.0                 # 渲染延迟：>0可放慢播放速度

    total_test_episodes = 10          # 测试回合数：输出每回合reward以及最终平均reward

    # PPO超参数（用于构造PPO对象；load参数后真正用到的是网络结构一致）
    # 注：这里的K_epochs/eps_clip/gamma等在测试阶段基本不会参与update，
    #     但PPO类初始化接口要求这些参数，因此保留并与训练一致是最安全的。
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99

    lr_actor = 0.0003
    lr_critic = 0.001
    #####################################################

    # 创建环境
    env = gym.make(env_name)

    # 状态维度：CartPole是4维
    state_dim = env.observation_space.shape[0]

    # 动作维度：离散动作空间（CartPole: 2个动作）
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # 初始化PPO agent（用于加载checkpoint并选择动作）
    # 注意：这里的use_gae/normalize_adv与训练时保持一致更规范，但实际上测试只用actor输出动作。
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=action_std,   # 离散动作空间传None也没问题（代码里不会用到）
        use_gae=True,                 # 与你训练CartPole-v1_GAE保持一致
        gae_lambda=0.95,
        normalize_adv=True
    )

    # ---------------- load checkpoint ----------------
    # 选择加载哪个seed训练出来的模型（0..4）
    random_seed = 0
    directory = f"PPO_preTrained/{run_name}/"
    checkpoint_path = directory + f"PPO_{run_name}_seed_{random_seed}.pth"

    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    print("--------------------------------------------------------------------------------------------")

    # ---------------- testing loop ----------------
    test_running_reward = 0.0

    # 测试多个episode：逐回合输出reward，并计算平均reward
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0.0

        # 设定测试时的环境seed：让不同episode初始状态略有变化，但仍可复现
        state, _ = env.reset(seed=random_seed + ep)

        for t in range(1, max_ep_len + 1):
            # 重要：select_action会把(s,a,logp,V)写入buffer（即使测试也会写）
            # 所以每个episode结束后要清空buffer，避免buffer无限增长占内存
            action = ppo_agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state

            ep_reward += reward

            # 渲染：本地演示用；无GUI环境建议关闭
            if render:
                env.render()
                if frame_delay > 0:
                    time.sleep(frame_delay)

            if done:
                break

        # 关键：清空buffer（因为测试阶段不需要累积轨迹用于update）
        # 否则select_action每步都append数据，buffer会不断变大
        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print(f"Episode: {ep}\t\tReward: {ep_reward:.2f}")

    env.close()

    print("============================================================================================")
    avg_test_reward = test_running_reward / total_test_episodes
    print(f"average test reward : {avg_test_reward:.2f}")
    print("============================================================================================")


if __name__ == "__main__":
    test()
