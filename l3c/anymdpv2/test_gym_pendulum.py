import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import random
from restools.plotting.plot_2D import savitzky_golay

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class PendulumVisualizer:
    def __init__(self, env):
        self.env = env
        self.observation_records = []
        self.action_records = []
        self.reward_records = []
        
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.observation_records.append(np.copy(obs))
        self.action_records.append(np.zeros((1,)))
        self.reward_records.append(0.0)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.observation_records.append(np.copy(obs))
        self.action_records.append(np.copy(action))
        self.reward_records.append(reward)
        return obs, reward, terminated, truncated, info
    
    def visualize_and_save(self, filename):
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=10, max_iter=500, learning_rate=100)
            
        obs_arr = np.array(self.observation_records, dtype="float32")
        act_arr = np.array(self.action_records, dtype="float32")
        
        # 处理数据维度
        if len(act_arr.shape) == 1:
            act_arr = act_arr.reshape(-1, 1)

        obs_tsne = tsne.fit_transform(obs_arr)
        act_tsne = tsne.fit_transform(act_arr)
        
        plt.figure(figsize=(10, 8))
        # Show Observation T-SNE
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(obs_tsne[:, 0], obs_tsne[:, 1], 
                            c='black', s=10, alpha=0.2)
        plt.title("Observation", fontsize=12, fontweight='bold', 
                 color='blue', pad=10)

        # Show Action T-SNE
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(act_tsne[:, 0], act_tsne[:, 1], 
                            c='black', s=10, alpha=0.2)
        plt.title("Action", fontsize=12, fontweight='bold', 
                 color='blue', pad=10)

        # Show State-Reward
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(obs_tsne[:, 0], obs_tsne[:, 1], 
                            c=self.reward_records, cmap='viridis', 
                            s=10, alpha=0.2)
        plt.colorbar()
        plt.title("States", fontsize=12, fontweight='bold', 
                 color='blue', pad=10)

        # Show Reward Curve
        plt.subplot(2, 2, 4)
        rewards_smooth = savitzky_golay(self.reward_records, 
                                      window_size=99, order=3)
        plt.plot(np.arange(len(self.reward_records)), 
                self.reward_records, c='red', alpha=0.2)
        plt.plot(np.arange(len(rewards_smooth)), 
                rewards_smooth, c='red')
        plt.title("Reward", fontsize=12, fontweight='bold', 
                 color='blue', pad=10)
        plt.savefig(filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed')
    args = parser.parse_args()

    if args.seed is None:
        current_time = int(time.time() * 1000)
        args.seed = current_time % 100000
        
    set_global_seed(args.seed)
    print(f"Using seed: {args.seed}")

    max_steps = 200000
    prt_freq = 1000
    train_steps = 160000
    eval_steps = 40000
    
    # 测试随机策略
    print("\n=== Testing Random Policy ===")
    env_random = gym.make('Pendulum-v1')
    env_random = PendulumVisualizer(env_random)
    
    state, info = env_random.reset(seed=args.seed)
    acc_reward = 0
    epoch_reward = 0
    steps = 0
    
    while steps < max_steps:
        action = env_random.env.action_space.sample()
        state, reward, terminated, truncated, info = env_random.step(action)
        done = terminated or truncated
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        
        if steps % prt_freq == 0 and steps > 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            epoch_reward = 0
            
        if done:
            state, info = env_random.reset()
            
    print(f"Random Policy Summary: {acc_reward}")
    env_random.visualize_and_save("pendulum_random_policy_result.pdf")
    
    # 测试PPO-MLP策略
    print("\n=== Testing PPO-MLP Policy ===")
    env_ppo = gym.make('Pendulum-v1')
    
    model = PPO('MlpPolicy', env_ppo, 
                verbose=1, 
                seed=args.seed,
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=32,
                n_epochs=5,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01)
    
    model.learn(total_timesteps=train_steps)
    
    # 评估阶段
    env_ppo = PendulumVisualizer(env_ppo)
    state, info = env_ppo.reset(seed=args.seed)
    acc_reward_mlp = 0
    epoch_reward = 0
    steps = 0

    while steps < eval_steps:
        action, _ = model.predict(state)
        state, reward, terminated, truncated, info = env_ppo.step(action)
        done = terminated or truncated
        acc_reward_mlp += reward
        epoch_reward += reward
        steps += 1
        
        if steps % prt_freq == 0 and steps > 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            epoch_reward = 0
            
        if done:
            state, info = env_ppo.reset()
            
    print(f"PPO-MLP Policy Summary: {acc_reward_mlp}")
    env_ppo.visualize_and_save("pendulum_ppo_mlp_policy_result.pdf")

    # 实验总结
    print("\n=== Experiment Summary ===")
    print(f"Used seed: {args.seed}")
    print(f"RANDOM POLICY total reward: {acc_reward}")
    print(f"PPO-MLP total reward: {acc_reward_mlp}")