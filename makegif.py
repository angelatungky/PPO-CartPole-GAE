import os
import imageio
import torch
import gym
import numpy as np
from PPO import PPO

def make_gif():
    env_name = "CartPole-v1"
    has_continuous_action_space = False
    max_ep_len = 400

    # PPO hyperparameters (must match training)
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001

    # Load environment
    env = gym.make(env_name, render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space
    )

    checkpoint_path = "PPO_preTrained/CartPole-v1_GAE/PPO_CartPole-v1_GAE_seed_0.pth"
    print("Loading model from:", checkpoint_path)
    ppo_agent.load(checkpoint_path)

    frames = []

    state, _ = env.reset()
    for t in range(max_ep_len):
        frame = env.render()
        frames.append(frame)

        action = ppo_agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()

    os.makedirs("gifs", exist_ok=True)
    gif_path = "gifs/cartpole_ppo_gae.gif"
    imageio.mimsave(gif_path, frames, fps=30)

    print("GIF saved at:", gif_path)

if __name__ == "__main__":
    make_gif()
