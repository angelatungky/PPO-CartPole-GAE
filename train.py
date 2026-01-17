import os
from datetime import datetime

import torch
import numpy as np
import gym

from PPO import PPO


################################### Training ###################################
def train(
    random_seed: int,
    run_name: str,
    use_gae: bool = True,
    gae_lambda: float = 0.95,
    normalize_adv: bool = True,
):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    gym_env_name = "CartPole-v1"  # Gym环境名称
    has_continuous_action_space = False  # CartPole是离散动作空间

    # 训练总步数与每回合最大步数
    max_ep_len = 400                    # 每个episode最多步数（你设置为400）
    max_training_timesteps = int(2e5)   # 总训练timesteps

    # 打印/记录/保存频率（以timesteps为单位）
    print_freq = max_ep_len * 10        # 每print_freq步打印一次平均reward
    log_freq = max_ep_len * 2           # 每log_freq步写一次CSV日志（平均reward）
    save_model_freq = int(1e5)          # 每save_model_freq步保存一次模型

    # （仅连续动作环境使用；离散环境不会用到，但保留不影响）
    action_std = 0.6
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = int(2.5e5)
    #####################################################

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4    # 每多少步做一次PPO update（收集一段rollout后更新）
    K_epochs = 80                       # 每次PPO update内部做多少个epoch优化

    eps_clip = 0.2                      # PPO clip系数
    gamma = 0.99                        # 折扣因子

    lr_actor = 3e-4                     # actor学习率
    lr_critic = 1e-3                    # critic学习率
    #####################################################

    print(f"Gym env name         : {gym_env_name}")
    print(f"Run name (folder)    : {run_name}")
    print(f"Random seed          : {random_seed}")
    # ====== 你做的扩展配置（核心对比：baseline vs GAE+norm） ======
    print(f"use_gae={use_gae}, gae_lambda={gae_lambda}, normalize_adv={normalize_adv}")

    # 创建环境
    env = gym.make(gym_env_name)

    # ---- 正确的随机种子设置（保证可复现） ----
    # 影响：网络初始化、numpy随机、环境reset、action采样
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    env.reset(seed=random_seed)
    env.action_space.seed(random_seed)

    # 状态/动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if has_continuous_action_space else env.action_space.n

    ###################### logging ######################
    # 日志文件夹：PPO_logs/<run_name>/
    log_dir = os.path.join("PPO_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 多seed实验：文件名包含seed，避免覆盖（这是你为multi-seed做的重要修改）
    log_f_name = os.path.join(log_dir, f"PPO_{run_name}_seed_{random_seed}.csv")
    print("logging at           : ", log_f_name)

    ################### checkpointing ###################
    # checkpoint文件夹：PPO_preTrained/<run_name>/
    model_dir = os.path.join("PPO_preTrained", run_name)
    os.makedirs(model_dir, exist_ok=True)

    # 多seed：checkpoint包含seed，避免覆盖
    checkpoint_path = os.path.join(model_dir, f"PPO_{run_name}_seed_{random_seed}.pth")
    print("save checkpoint path : ", checkpoint_path)

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per ep   : ", max_ep_len)
    print("save model freq        : ", save_model_freq)
    print("log freq               : ", log_freq)
    print("print freq             : ", print_freq)
    print("state dim              : ", state_dim)
    print("action dim             : ", action_dim)
    print("update_timestep        : ", update_timestep)
    print("K_epochs               : ", K_epochs)
    print("eps_clip               : ", eps_clip)
    print("gamma                  : ", gamma)
    print("lr_actor               : ", lr_actor)
    print("lr_critic              : ", lr_critic)
    print("--------------------------------------------------------------------------------------------")
    print("============================================================================================")

    # 初始化PPO agent（你在PPO.py里新增了use_gae/gae_lambda/normalize_adv参数）
    ppo_agent = PPO(
        state_dim, action_dim,
        lr_actor, lr_critic,
        gamma, K_epochs, eps_clip,
        has_continuous_action_space, action_std,
        use_gae=use_gae,
        gae_lambda=gae_lambda,
        normalize_adv=normalize_adv
    )

    # 记录训练时间
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at : ", start_time)
    print("============================================================================================")

    # 打开CSV日志文件
    with open(log_f_name, "w+") as log_f:
        log_f.write("episode,timestep,reward\n")

        # running统计：用于print的平均reward
        print_running_reward = 0.0
        print_running_episodes = 0

        # running统计：用于log的平均reward
        log_running_reward = 0.0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0

        # 训练主循环：直到达到max_training_timesteps
        while time_step <= max_training_timesteps:

            # 为了避免每个episode完全相同的初始状态，你这里用 seed + episode编号
            # 这样同一个random_seed下的不同episode仍然有变化，更符合训练随机性
            state, _ = env.reset(seed=random_seed + i_episode)

            current_ep_reward = 0.0

            for t in range(1, max_ep_len + 1):

                # 选择动作（policy_old采样，并写入buffer）
                action = ppo_agent.select_action(state)

                # Gym新API：返回terminated/truncated
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # 存入buffer：reward与done标记（PPO update时用于returns/GAE计算）
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                state = next_state
                time_step += 1
                current_ep_reward += reward

                # 每update_timestep步做一次PPO更新（用buffer中rollout做K_epochs次优化）
                if time_step % update_timestep == 0:
                    ppo_agent.update()

                # 连续动作环境才需要decay std（离散动作不会进入）
                if has_continuous_action_space and (time_step % action_std_decay_freq == 0):
                    ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

                # ------- 写日志（平均reward） -------
                # 注意：你记录的是log_freq窗口内的平均episode reward
                if time_step % log_freq == 0:
                    if log_running_episodes > 0:
                        log_avg_reward = log_running_reward / log_running_episodes
                        log_avg_reward = round(log_avg_reward, 4)
                        log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
                        log_f.flush()
                    log_running_reward = 0.0
                    log_running_episodes = 0

                # ------- 打印到控制台（平均reward） -------
                if time_step % print_freq == 0:
                    if print_running_episodes > 0:
                        print_avg_reward = print_running_reward / print_running_episodes
                        print_avg_reward = round(print_avg_reward, 2)
                        print(f"Episode : {i_episode}\t Timestep : {time_step}\t Average Reward : {print_avg_reward}")
                    print_running_reward = 0.0
                    print_running_episodes = 0

                # ------- 保存模型 -------
                if time_step % save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")

                if done:
                    break

            # 更新窗口统计（用于print/log）
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1

    env.close()

    # 总训练时间
    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print("Finished training at : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    # 单次运行入口：默认跑extended版本（方便你本地快速测试）
    train(
        random_seed=0,
        run_name="CartPole-v1_GAE",
        use_gae=True,
        gae_lambda=0.95,
        normalize_adv=True
    )
