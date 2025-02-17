import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import torch.nn as nn
import time
import random
import json
from utils.random_nn import RandomMLP
import argparse

class MappedPendulumEnv(gym.Wrapper):
    def __init__(self, 
                state_dim=64,   
                action_dim=8,   
                seed=None):
        base_env = gym.make('Pendulum-v1', max_episode_steps=200)
        super().__init__(base_env)
        
        self.obs_mapper = self._create_obs_mapper(
            input_dim=3,        # [cosθ, sinθ, angular_velocity]
            output_dim=state_dim,
            hidden_layers=[64, 32],
            activation=lambda x: np.clip(np.tanh(x), -5, 5),
            seed=seed
        )
        
        self.action_mapper = self._create_action_mapper(
            input_dim=action_dim,
            output_dim=1,       # 原始动作维度 (torque)
            hidden_layers=[32],
            activation=lambda x: np.clip(x, -1, 1) * np.exp(-np.abs(x)),
            seed=seed
        )

        # 重定义空间维度
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(action_dim,), dtype=np.float32
        )

    def _create_obs_mapper(self, input_dim, output_dim, 
                         hidden_layers, activation, seed):
        return RandomMLP(
            n_inputs=input_dim,
            n_outputs=output_dim,
            n_hidden_layers=hidden_layers,
            activation=activation,
            seed=seed
        )

    def _create_action_mapper(self, input_dim, output_dim,
                            hidden_layers, activation, seed):
        return RandomMLP(
            n_inputs=input_dim,
            n_outputs=output_dim,
            n_hidden_layers=hidden_layers,
            activation=activation,
            seed=seed
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.obs_mapper(obs.astype(np.float32)), info

    def step(self, action):
        mapped_action = self.action_mapper(action)
        
        mapped_action = np.clip(mapped_action * 2.0, -2.0, 2.0)
        
        next_obs, reward, terminated, truncated, info = super().step(mapped_action)
        return self.obs_mapper(next_obs.astype(np.float32)), reward, terminated, truncated, info
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    # 如果没有指定seed，则生成一个随机种子
    if args.seed is None:
        current_time = int(time.time() * 1000)
        args.seed = current_time % 100000  
        
    GLOBAL_SEED = args.seed
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    print(f"Using seed: {GLOBAL_SEED}")

    max_steps = 100000
    prt_freq = 1000
    train_steps = 80000
    eval_steps = 20000

    def make_env():
        return MappedPendulumEnv(
            state_dim=64, 
            action_dim=8,
            seed=GLOBAL_SEED
        )
    
    env_random = make_env()
    env_ppo_mlp = make_env()
    env_ppo_hybrid = make_env()

    # 记录任务信息（适配标准环境版本）
    task_info = {
        'seed': GLOBAL_SEED,
        'state_dim': 64,
        'action_dim': 8,
        'max_steps': 200  
    }
    
    with open('task_info_gym.json', 'w') as f:
        json.dump(task_info, f, indent=4)

    print("\n=== Testing Random Policy ===")
    state, info = env_random.reset()
    acc_reward = 0
    epoch_reward = 0
    steps = 0
    
    while steps < max_steps:
        action = env_random.action_space.sample()
        state, reward, terminated, truncated, info = env_random.step(action)
        done = terminated or truncated
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        
        if steps % prt_freq == 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            epoch_reward = 0
            
        if done:
            state, info = env_random.reset()
            
    print(f"Random Policy Summary: {acc_reward}")

    print("\n=== Testing PPO-MLP Policy ===")
    model_mlp = PPO('MlpPolicy', env_ppo_mlp, 
                   verbose=1, 
                   seed=GLOBAL_SEED,
                   learning_rate=1e-4,
                   n_steps=1024,
                   batch_size=32,
                   n_epochs=5,
                   gamma=0.99,
                   gae_lambda=0.95,
                   clip_range=0.2,
                   clip_range_vf=0.2,
                   ent_coef=0.01,
                   max_grad_norm=0.5)
    
    model_mlp.learn(total_timesteps=train_steps)

    state, info = env_ppo_mlp.reset()
    acc_reward_mlp = 0
    epoch_reward = 0
    steps = 0

    while steps < eval_steps:
        action, _ = model_mlp.predict(state)
        state, reward, terminated, truncated, info = env_ppo_mlp.step(action)
        done = terminated or truncated
        acc_reward_mlp += reward
        epoch_reward += reward
        steps += 1
        
        if steps % prt_freq == 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            epoch_reward = 0
            
        if done:
            state, info = env_ppo_mlp.reset()
        
    print(f"PPO-MLP Policy Summary: {acc_reward_mlp}")

    print("\n=== Testing PPO-HYBRID Policy ===")
    policy_kwargs = dict(
        lstm_hidden_size=64,
        n_lstm_layers=1,
        net_arch=dict(pi=[128,64], vf=[128,64])
    )

    model_hybrid = RecurrentPPO(
        "MlpLstmPolicy",
        env_ppo_hybrid,
        verbose=1,
        seed=GLOBAL_SEED,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            lstm_hidden_size=32,
            n_lstm_layers=1,
            net_arch=dict(
                pi=[64, 32],  
                vf=[64, 32]   
            ),
            activation_fn=nn.ReLU
        )
    )

    model_hybrid.learn(total_timesteps=train_steps)

    state, info = env_ppo_hybrid.reset()
    acc_reward_hybrid = 0
    epoch_reward = 0
    steps = 0
    lstm_states = None
    done = True
    episode_starts = np.ones(1, dtype=bool) 

    while steps < eval_steps:
        action, lstm_states = model_hybrid.predict(
            state, 
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True
        )
        state, reward, terminated, truncated, info = env_ppo_hybrid.step(action)
        done = terminated or truncated
        acc_reward_hybrid += reward
        epoch_reward += reward
        steps += 1

        episode_starts = np.array([terminated or truncated])
        
        if steps % prt_freq == 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            epoch_reward = 0
            
        if done:
            state, info = env_ppo_hybrid.reset()
            lstm_states = None
            
    print(f"PPO-HYBRID Policy Summary: {acc_reward_hybrid}")
    # env_ppo_hybrid.unwrapped.visualize_and_save("ppo_hybrid_policy_result.pdf")

    # 实验总结
    print("\n=== Experiment Summary ===")
    print(f"Random seed used: {GLOBAL_SEED}")
    print(f"Random Policy total reward: {acc_reward}")
    print(f"PPO-MLP total reward: {acc_reward_mlp}")
    print(f"PPO-HYBRID total reward: {acc_reward_hybrid}")
    print("\nTest Passed")

    