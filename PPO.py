import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical

################################## set device ##################################
print("============================================================================================")
# 设备选择：默认CPU；如果有GPU则使用cuda:0
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        # 轨迹缓存：用于存储一段rollout中每一步的数据（用于PPO更新）
        self.actions = []        # 动作 a_t
        self.states = []         # 状态 s_t
        self.logprobs = []       # 旧策略下 log pi_old(a_t|s_t)
        self.rewards = []        # 奖励 r_t
        self.state_values = []   # 旧策略critic预测的 V_old(s_t)
        self.is_terminals = []   # 终止标记 done_t（terminated or truncated）

    def clear(self):
        # 清空缓存：每次做完一次PPO update后清空，开始下一段rollout
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super().__init__()

        # 标记动作空间类型：连续 or 离散
        self.has_continuous_action_space = has_continuous_action_space

        # 连续动作：需要维护动作方差（用于构造高斯分布）
        if has_continuous_action_space:
            self.action_dim = action_dim
            # action_var: 每个动作维度的方差（std^2）
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # ---------------- Actor 网络（策略网络） ----------------
        # 连续动作：输出动作均值 action_mean，并用高斯分布采样
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            # 离散动作：输出每个动作的概率（Softmax）
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # ---------------- Critic 网络（价值网络） ----------------
        # 输出状态价值 V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        # 连续动作：调整高斯策略的标准差（探索程度）
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """
        采样动作（与环境交互时用）：
        state: Tensor shape [state_dim]
        returns: action, logprob, value
        """
        if self.has_continuous_action_space:
            # 连续动作：多元正态分布 N(mean, diag(var))
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # 离散动作：Categorical分布（由softmax概率决定）
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # 从分布中采样动作 a_t
        action = dist.sample()
        # 记录 log pi(a_t|s_t) （用于重要性采样比率）
        action_logprob = dist.log_prob(action)
        # critic预测的状态价值 V(s_t)
        state_val = self.critic(state)

        # 注意：都detach，避免在采样阶段建立梯度图
        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        评估阶段（PPO更新时用）：
        给定状态序列和动作序列，计算：
        - logprobs: 当前策略下 log pi_theta(a_t|s_t)
        - state_values: 当前critic V_theta(s_t)
        - entropy: 策略熵（用于鼓励探索）
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # 连续动作维度为1时，确保shape一致
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std_init=0.6,
        use_gae=False,
        gae_lambda=0.95,
        normalize_adv=True,
    ):
        # 动作空间类型
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        # PPO关键超参
        self.gamma = gamma            # 折扣因子
        self.eps_clip = eps_clip      # clip范围 epsilon
        self.K_epochs = K_epochs      # 每次update的优化epoch数

        # ============ 你做的扩展（重要修改点） ============
        # use_gae：是否启用GAE-λ 优势估计（替换MC return-to-go）
        # gae_lambda：GAE的λ参数（控制偏差-方差折中）
        # normalize_adv：是否对优势 A_t 做标准化（提升训练稳定性）
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        # ===============================================

        # rollout缓存
        self.buffer = RolloutBuffer()

        # 当前策略网络（actor+critic）
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        # 一个优化器同时更新actor与critic（不同学习率）
        self.optimizer = torch.optim.Adam([
            {"params": self.policy.actor.parameters(), "lr": lr_actor},
            {"params": self.policy.critic.parameters(), "lr": lr_critic},
        ])

        # old策略：用于采样数据（PPO要求固定旧策略计算比率）
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # value loss：MSE
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        # 连续动作下可调探索强度
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # 连续动作：逐步减小动作std（减少探索、增加利用）
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = round(self.action_std - action_std_decay_rate, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        """
        与环境交互时调用：
        - 用旧策略 policy_old 采样动作
        - 并将 state/action/logprob/value 存入 buffer，供后续PPO update使用
        """
        with torch.no_grad():
            # 将numpy状态转为Tensor（并放到device上）
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)  # [state_dim]
            action, action_logprob, state_val = self.policy_old.act(state_t)

        # 存入rollout buffer
        self.buffer.states.append(state_t)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        # 返回env.step()所需格式
        if self.has_continuous_action_space:
            return action.detach().cpu().numpy().flatten()
        else:
            return action.item()

    def _compute_returns_and_advantages(self, rewards, dones, values):
        """
        你做的关键修改点之一：统一“baseline vs GAE”优势计算逻辑
        输入：
          rewards: Tensor [T]
          dones:   Tensor [T] (done=1 else 0)
          values:  Tensor [T] (旧critic预测 V(s_t))
        输出：
          returns: Tensor [T]   训练critic用的目标（R_t 或 GAE returns）
          advantages: Tensor [T] 训练actor用的优势 A_t
        """
        T = rewards.size(0)

        # ---------------- Baseline：Monte Carlo Return-to-Go ----------------
        # 从后往前算discounted return：遇到done则清零，保证episode边界正确
        if not self.use_gae:
            returns = torch.zeros(T, dtype=torch.float32, device=device)
            discounted = 0.0
            for t in reversed(range(T)):
                if dones[t] > 0.5:
                    discounted = 0.0
                discounted = rewards[t] + self.gamma * discounted
                returns[t] = discounted
            # 优势：A_t = R_t - V(s_t)
            advantages = returns - values
            return returns, advantages

        # ---------------- Extended：GAE(lambda) ----------------
        # GAE核心思想：
        #   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        #   A_t = delta_t + gamma*lambda*delta_{t+1} + ...
        # 通过lambda降低方差，提升训练稳定性
        advantages = torch.zeros(T, dtype=torch.float32, device=device)
        gae = 0.0

        # next_values[t] = V(s_{t+1})，用于bootstrap
        next_values = torch.zeros(T, dtype=torch.float32, device=device)
        if T > 1:
            next_values[:-1] = values[1:]

        # 最后一步：如果不是terminal，使用bootstrap（这里用最后的value近似）；
        # 如果terminal则为0（episode结束不bootstrap）
        next_values[-1] = values[-1] if dones[-1] < 0.5 else 0.0

        # 从后往前累计GAE（遇到done则mask=0，自动断开episode）
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]  # done=1 => mask=0
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        # GAE returns：R_t^GAE = A_t + V(s_t)
        returns = advantages + values
        return returns, advantages

    def update(self):
        """
        PPO更新：
        1) 从buffer取出旧数据（s, a, logp_old, V_old, r, done）
        2) 计算 returns 与 advantages（baseline或GAE）
        3) 可选：advantage normalization（你的第二个扩展）
        4) 进行K个epoch的PPO clipped优化
        """
        # 将buffer中list拼成Tensor（注意不随意squeeze，保持shape稳定）
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)         # [T, state_dim]
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().to(device)       # [T] or [T, action_dim]
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().to(device)     # [T]
        old_state_values = torch.stack(self.buffer.state_values, dim=0).detach().to(device)  # [T, 1] maybe

        # 将 V(s_t) 拉平成 [T]
        old_state_values = old_state_values.view(-1)

        rewards = torch.as_tensor(self.buffer.rewards, dtype=torch.float32, device=device)  # [T]
        dones = torch.as_tensor(self.buffer.is_terminals, dtype=torch.float32, device=device)  # [T]

        # 计算returns与advantages（baseline or GAE）
        returns, advantages = self._compute_returns_and_advantages(rewards, dones, old_state_values)

        # ============ 你做的扩展：Advantage Normalization ============
        # 对优势做标准化可以减少梯度尺度波动，常见于PPO实现，有助于稳定训练
        if self.normalize_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # =========================================================

        # 对策略进行K个epoch的优化（同一批rollout数据重复利用K次）
        for _ in range(self.K_epochs):
            # 用当前policy重新计算 logp_theta(a|s), V_theta(s), entropy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.view(-1)

            # 重要性采样比率 r_t(theta) = exp(logp - logp_old)
            ratios = torch.exp(logprobs - old_logprobs)

            # PPO clipped surrogate objective
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # 总loss：
            # -actor: 取 -min(surr1, surr2)（梯度上升转为loss最小化）
            # +critic: 0.5 * MSE(V(s), returns)
            # -entropy: 0.01 * entropy（鼓励探索）
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新old策略参数，使下一次采样使用新策略（PPO要求更新后同步）
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空buffer，开始下一段rollout
        self.buffer.clear()

    def save(self, checkpoint_path):
        # 保存旧策略（用于测试/继续训练）
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        # 加载checkpoint到policy_old与policy，保证一致性
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
