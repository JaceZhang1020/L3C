import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, A2C, TD3, PPO
from typing import Dict, Any, List, Tuple
import random
import time
# from l3c.utils import pseudo_random_seed, RandomMLP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from copy import deepcopy
from restools.plotting.plot_2D import savitzky_golay
from sb3_contrib import RecurrentPPO
import re
import torch.nn as nn

GLOBAL_SEED = None  # 默认为None表示随机

def set_global_seed(seed):
    global GLOBAL_SEED
    GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)

def pseudo_random_seed(hyperseed=0):
    '''
    Generate a pseudo random seed based on current time and system random number
    '''
    timestamp = time.time_ns()
    system_random = int(random.random() * 100000000)
    pseudo_random = timestamp + system_random + hyperseed
    
    return pseudo_random % (4294967296)

def weights_and_biases(n_in, n_out):
    avg_in = np.random.normal(loc=0.0, scale=1.0, size=[n_in])
    avg_out = np.random.normal(loc=0.0, scale=1.0, size=[n_out])
    weights = np.outer(avg_out, avg_in) + np.random.normal(size=[n_out, n_in])
    weights = weights * np.sqrt(6.0 / (n_in + n_out))
    bias = 0.1 * np.random.normal(size=[n_out]) * avg_out
    return weights, bias

def actfunc(name):
    name = name.lower()
    if(name=='sigmoid'):
        return lambda x: 1/(1+np.exp(-x))
    elif(name.find('leakyrelu') >= 0):
        return lambda x: np.maximum(0.01*x, x)
    elif(name.find('bounded') >= 0):
        pattern = r"bounded\(([-+]?\d*\.\d+|[-+]?\d+),\s*([-+]?\d*\.\d+|[-+]?\d+)\)"
        match = re.match(pattern, name)
        if match:
            B = float(match.group(1).strip())
            T = float(match.group(2).strip())
        else:
            raise ValueError("Bounded support only BOUNDED(min,max) type")
        k = (T - B) / 2
        return lambda x: k*np.tanh(x/k) + k + B
    elif(name == 'sin'):
        return lambda x: np.concat([np.sin(x[:len(x)//2]), np.cos(x[len(x)//2:])], axis=-1)
    elif(name == 'none'):
        return lambda x: x

class RandomMLP(object):
    '''
    A class for generating random MLPs with given parameters
    '''
    def __init__(self, n_inputs, n_outputs, 
                 n_hidden_layers=None, 
                 hidden_activation=None, 
                 output_activation=None,
                 seed=None):
        # Set the seed for the random number generator
        if seed is None:
            seed = pseudo_random_seed()
        random.seed(seed)

        # Set the number of hidden units and activation function
        self.hidden_units = [n_inputs]
        if n_hidden_layers is not None:
            if(isinstance(n_hidden_layers, list)):
                self.hidden_units += n_hidden_layers
            elif(isinstance(n_hidden_layers, np.ndarray)):
                self.hidden_units += n_hidden_layers.tolist()
            elif(isinstance(n_hidden_layers, tuple)):
                self.hidden_units += list(n_hidden_layers)
            elif(isinstance(n_hidden_layers, int)):
                self.hidden_units.append(n_hidden_layers)
            else:
                raise TypeError(f"Invalid input type of n_hidden_layers: {type(n_hidden_layers)}")
        self.hidden_units.append(n_outputs)
        
        self.activation = []

        if hidden_activation is None:
            for _ in range(len(self.hidden_units)-2):
                self.activation.append(actfunc('sigmoid'))
        else:
            for hidden_act in hidden_activation:
                self.activation.append(actfunc(hidden_act))
        if output_activation is None:
            self.activation = actfunc('none')
        else:
            self.activation = actfunc(output_activation)
        
        # Initialize weights and biases to random values
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_units)-1):
            w, b = weights_and_biases(self.hidden_units[i], self.hidden_units[i+1])
            self.weights.append(w)
            if(i == len(self.hidden_units)-2):
                b *= 0
            self.biases.append(b)
            
    def forward(self, inputs):
        outputs = inputs
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            outputs = self.activation(weight @ outputs + bias)
        return outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

class AnyMDPEnv(gym.Env):
    def __init__(self, max_steps: int):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.max_steps = max_steps
        self.task_set = False

    def set_task(self, task: Dict[str, Any]):
        for key in task:
            setattr(self, key, task[key])
        # 定义无界的 observation_space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                          shape=(self.state_dim,), dtype=np.float32)
        # 定义 action_space
        self.action_space = spaces.Box(low=-1, high=1, 
                                     shape=(self.action_dim,), dtype=np.float32)
        self.task_set = True
        self.need_reset = True

    def reset(self, seed=None, options=None):
        if not self.task_set:
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        
        loc, noise = random.choice(self.born_loc)
        self._inner_state = loc + noise * np.random.normal(size=self.ndim)
        self._state = self.observation_map(self._inner_state)
        if self.mode == 'multi':
            self.available_goal = deepcopy(self.goal_loc)
        return self._state, {"steps": self.steps}
    
    def near_born_loc(self):
        for loc, noise in self.born_loc:
            dist = np.linalg.norm(self._inner_state - loc)
            if dist < noise * 3:
                return True
        return False

    def calculate_loc(self, loc, steps):
        g_loc = np.zeros(self.ndim)
        for n, k in loc:
            g_loc += k[:, 0] * np.cos(0.01 * n * self.steps) + k[:, 1] * np.sin(0.01 * n * self.steps)
        return g_loc / len(loc)
    
    def goal_reward_static(self, ns):
        min_dist = np.inf
        reward = 0
        done = False
        for gs, d, gr in self.goal_loc:
            dist = np.linalg.norm(ns - gs)
            if dist < d:
                reward += gr
                done = True
                break
        return reward, done
    
    def goal_reward_multi(self, ns):
        min_dist = np.inf
        reward = 0
        for gs, d, gr in self.available_goal:
            dist = np.linalg.norm(ns - gs)
            if dist < d:
                reward += gr
                self.available_goal.remove((gs, d, gr))
                break
        return reward, False
    
    def goal_reward_dynamic(self, ns):
        goal_loc = self.calculate_loc(self.goal_loc, self.steps)
        goal_dist = np.linalg.norm(ns - goal_loc)
        reward = 0
        if goal_dist < self.goal_potential[0]:
            reward = self.goal_potential[1] * (1 - goal_dist / self.goal_potential[0])
        return reward, False
    
    def goal_reward_consist(self, ns):
        reward = self.goal_reward(np.concatenate([self.inner_state, ns]))[0]
        return reward, False

    def step(self, action):
        if self.need_reset or not self.task_set:
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert np.shape(action) == (self.action_dim,)

        # update inner state
        inner_deta = self.action_map(np.concatenate([self._inner_state, np.array(action)], axis=-1))
        next_inner_state = (self._inner_state + 
                           inner_deta * self.action_weight + 
                           self.transition_noise * np.random.normal(size=(self.ndim,)))

        # basic reward
        reward = self.average_cost + self.reward_noise * np.random.normal()
        done = False
        
        if self.mode == 'static':
            reward, done = self.goal_reward_static(next_inner_state)
        elif self.mode == 'multi':
            reward, done = self.goal_reward_multi(next_inner_state)
            if len(self.available_goal) == 0:
                done = True
        elif self.mode == 'dynamic':
            reward, done = self.goal_reward_dynamic(next_inner_state)
        elif self.mode == 'consis':
            reward, done = self.goal_reward_consist(next_inner_state)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Calculate Pitfalls
        if not done:
            pitfall_penalty = 0
            switch = self.pitfalls_switch(next_inner_state)
            risk = np.sum((switch < 0.0).astype('float32')) / np.size(switch)
            if risk > self.risk_limit and not self.near_born_loc():
                reward += self.pitfalls_penalty
                done = True

        # Calculate Potential Energy
        if self.use_potential:
            reward += (self.potential_energy(next_inner_state)[0] - 
                      self.potential_energy(self._inner_state)[0])

        self.steps += 1
        info = {"steps": self.steps}
        
        self._inner_state = next_inner_state
        self._state = self.observation_map(self._inner_state)

        truncated = self.steps >= self.max_steps
        if done or truncated:
            self.need_reset = True
        
        return self._state, reward, done, truncated, info

    @property
    def state(self):
        return np.copy(self._state)
    
    @property
    def inner_state(self):
        return np.copy(self._inner_state)
    
class AnyMDPVisualizer(AnyMDPEnv):
    def set_task(self, task):
        # Set task will automatically reset all records
        super().set_task(task)
        self.observation_records = []
        self.inner_state_records = []
        self.action_records = []
        self.reward_records = []
    
    def color_spec(self, i):
        return [self.color_spec_type[i][idx] for idx in self.colors]

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.observation_records.append(np.copy(obs))
        self.action_records.append(np.zeros((self.action_dim,)))
        self.inner_state_records.append(np.copy(self.inner_state))
        self.reward_records.append(0.0)
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.observation_records.append(np.copy(obs))
        self.inner_state_records.append(np.copy(self.inner_state))
        self.action_records.append(np.copy(action))
        self.reward_records.append(reward)
        return obs, reward, done, truncated, info

    def visualize_and_save(self, filename=None):
        tsne = TSNE(n_components=2, random_state=pseudo_random_seed(),
                    perplexity=10, max_iter=500, learning_rate=100)
        if filename is not None:
            file_name = filename
        else:
            file_name = "./anymdp_visualizer_output.pdf"
            
        obs_arr = np.array(self.observation_records, dtype="float32")
        act_arr = np.array(self.action_records, dtype="float32")
        s_arr = np.array(self.inner_state_records)
        max_steps = len(self.inner_state_records)

        obs_tsne = tsne.fit_transform(np.array(obs_arr))
        act_tsne = tsne.fit_transform(np.array(act_arr))
        s_tsne = tsne.fit_transform(np.array(s_arr))
        
        plt.figure(figsize=(10, 8))
        # Show Observation T-SNE
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(obs_tsne[:, 0], obs_tsne[:, 1], c='black', s=10, alpha=0.2)
        plt.title("Observation", fontsize=12, fontweight='bold', color='blue', pad=10)

        # Show Action T-SNE
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(act_tsne[:, 0], act_tsne[:, 1], c='black', s=10, alpha=0.2)
        plt.title("Action", fontsize=12, fontweight='bold', color='blue', pad=10)

        # Show State T-SNE
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(s_tsne[:, 0], s_tsne[:, 1], c=self.reward_records, 
                            cmap='viridis', s=10, alpha=0.2, marker='o')
        plt.colorbar()
        plt.title("States", fontsize=12, fontweight='bold', color='blue', pad=10)

        # Show Reward Curve
        plt.subplot(2, 2, 4)
        rewards_smooth = savitzky_golay(self.reward_records, window_size=99, order=3)
        scatter = plt.plot(np.arange(np.size(self.reward_records)), 
                         self.reward_records, c='red', alpha=0.2)
        scatter = plt.plot(np.arange(np.size(rewards_smooth)), 
                         rewards_smooth, c='red')
        plt.title("Reward", fontsize=12, fontweight='bold', color='blue', pad=10)
        plt.savefig(file_name)

def create_task(state_dim: int = 256, action_dim: int = 256) -> Dict[str, Any]:
    if GLOBAL_SEED is not None:
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)  
    else:
        random.seed(pseudo_random_seed())
        np.random.seed(pseudo_random_seed())
    
    task = dict()
    task["mode"] = random.choice(["static", "dynamic", "multi", "consis"])
    task["state_dim"] = state_dim
    task["action_dim"] = action_dim
    task["ndim"] = random.randint(3, 33)  # At most 32-dimensional space
    task["max_steps"] = random.randint(100, 1000)
    # task["action_weight"] = random.uniform(5.0e-3, 0.10, size=(task['ndim'],))
    task["action_weight"] = np.random.uniform(5.0e-3, 0.10, size=(task['ndim'],))
    task["average_cost"] = np.random.exponential(0.01) * random.choice([-2, -1, 0, 1])
    # task["average_cost"] = random.exponential(0.01) * random.choice([-2, -1, 0, 1])
    task["transition_noise"] = max(0, np.random.normal(scale=5.0e-3))
    task["reward_noise"] = max(0, np.random.normal(scale=5.0e-3))
    task["use_potential"] = random.randint(0, 2)
    task["risk_limit"] = random.uniform(0.30, 0.55)

    # Observation mapping
    ndim = task['ndim']
    observation_map = RandomMLP(
        n_inputs=ndim, 
        n_outputs=state_dim,
        n_hidden_layers=random.randint(1, ndim // 2 + 1),
        output_activation='bounded(-1,1)'
    )
    task["observation_map"] = observation_map

    # Action mapping
    action_map = RandomMLP(
        n_inputs=action_dim + ndim,
        n_outputs=ndim,
        n_hidden_layers=random.randint(ndim * 2, ndim * 4),
        output_activation='bounded(-1,1)'
    )
    task["action_map"] = action_map

    # Born locations
    born_loc_num = random.randint(1, 10)
    task["born_loc"] = [(np.random.uniform(-1, 1, size=(task['ndim'],)), 
                         np.random.exponential(0.10,)) 
                        for _ in range(born_loc_num)]

    # Goals based on mode
    existing_locs = [loc for loc, _ in task["born_loc"]]
    if task["mode"] == "static":
        sgoal_num = 1
        sgoal_loc = []
        for _ in range(sgoal_num):
            min_dist = 0.0
            while min_dist < 0.5:
                sloc = np.random.uniform(-1, 1, size=(task["ndim"],))
                min_dist = 10000   
                for loc in existing_locs:
                    dist = np.linalg.norm(sloc-loc[0])
                    if dist < min_dist:
                        min_dist = dist
            sloc_copy = np.copy(sloc)
            sgoal_loc.append((sloc_copy, 
                            random.uniform(0.02, 0.2),
                            np.random.exponential(10.0)))
            existing_locs.append(np.copy(sloc))
        task["goal_loc"] = sgoal_loc
    elif task["mode"] == "multi":
        sgoal_num = random.randint(0, 10)
        sgoal_loc = []
        for _ in range(sgoal_num):
            min_dist = 0.0
            while min_dist < 0.5:
                sloc = np.random.uniform(-1, 1, size=(task["ndim"],))
                min_dist = 10000   
                for loc in existing_locs:
                    dist = np.linalg.norm(sloc-loc[0])
                    if dist < min_dist:
                        min_dist = dist
            sloc_copy = np.copy(sloc)
            sgoal_loc.append((sloc_copy, 
                            random.uniform(0.02, 0.2),
                            np.random.exponential(10.0)))
            existing_locs.append(np.copy(sloc))
        task["goal_loc"] = sgoal_loc
    elif task["mode"] == "dynamic":
        max_order, max_item = 16, 3
        item_num = random.randint(0, max_item + 1)
        dgoal_loc = [(0, np.random.normal(size=(task["ndim"], 2)))]
        for _ in range(item_num):
            order = random.randint(1, max_order + 1)
            factor = random.normal(size=(task["ndim"], 2))
            dgoal_loc.append((order, factor))
        task["goal_loc"] = dgoal_loc
        task["goal_potential"] = (random.uniform(0.10, 0.50),
                                random.exponential(2.0))
    elif task["mode"] == "consis":
        goal_reward = RandomMLP(2 * ndim, 1, 
                              n_hidden_layers=random.randint(ndim * 2, ndim * 4), 
                              output_activation='bounded(-1,1)')
        task["goal_reward"] = goal_reward
    else:
        raise ValueError(f"Unknown task type {task['mode']}")

    # Pitfalls
    pitfalls_switch = RandomMLP(
    n_inputs=ndim,
    n_outputs=random.randint(ndim * 2, ndim * 4),
    output_activation='sin'
    )
    task["pitfalls_switch"] = pitfalls_switch
    task["pitfalls_penalty"] = min(0, np.random.normal() - 1.0)

    # Potential Energy
    potential_energy = RandomMLP(
    n_inputs=ndim,
    n_outputs=1,
    n_hidden_layers=random.randint(ndim * 2, ndim * 4),
    output_activation='bounded(-1,1)'
    )
    task["potential_energy"] = potential_energy
    
    return task

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    # 如果没有指定seed，则生成一个随机种子
    if args.seed is None:
        # 使用时间戳生成种子
        current_time = int(time.time() * 1000)
        args.seed = current_time % 100000  # 取模保持在合理范围内
        
    set_global_seed(args.seed)
    print(f"Using seed: {args.seed}")

    max_steps = 200000
    prt_freq = 1000
    train_steps = 160000
    eval_steps = 40000
    
    # 创建任务
    task = create_task(state_dim=128, action_dim=16)

    task_info = {
    'seed': GLOBAL_SEED,
    'mode': task['mode'],
    'ndim': task['ndim'],
    'max_steps': task['max_steps'],
    'transition_noise': task['transition_noise'],
    'reward_noise': task['reward_noise'],
    'use_potential': task['use_potential'],  
    'risk_limit': task['risk_limit']         
    }
    
    # 保存任务信息
    import json
    with open('task_info.json', 'w') as f:
        json.dump(task_info, f, indent=4)
    
    # 测试随机策略
    print("\n=== Testing Random Policy ===")
    env_random = AnyMDPVisualizer(max_steps=max_steps)  
    env_random = gym.wrappers.TimeLimit(env_random, max_episode_steps=max_steps)
    env_random.unwrapped.set_task(task)  
    state, info = env_random.reset(seed=GLOBAL_SEED)  # 设置环境种子
    
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
        
        if steps % prt_freq == 0 and steps > 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            epoch_reward = 0
            
        if done:
            state, info = env_random.reset()
            
    print(f"Random Policy Summary: {acc_reward}")
    env_random.unwrapped.visualize_and_save("random_policy_result.pdf")
    
    print("\n=== Testing PPO-MLP Policy ===")
    env_ppo_mlp = AnyMDPVisualizer(max_steps=train_steps+eval_steps)  
    env_ppo_mlp = gym.wrappers.TimeLimit(env_ppo_mlp, max_episode_steps=train_steps+eval_steps)
    env_ppo_mlp.unwrapped.set_task(task)

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
        
        if steps % prt_freq == 0 and steps > 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            epoch_reward = 0
            
        if done:
            state, info = env_ppo_mlp.reset()
        
    print(f"PPO-MLP Policy Summary: {acc_reward_mlp}")
    env_ppo_mlp.unwrapped.visualize_and_save("ppo_mlp_policy_result.pdf")

    # 测试PPO-HYBRID策略
    print("\n=== Testing PPO-HYBRID Policy ===")
    env_ppo_hybrid = AnyMDPVisualizer(max_steps=train_steps+eval_steps)  
    env_ppo_hybrid = gym.wrappers.TimeLimit(env_ppo_hybrid, max_episode_steps=train_steps+eval_steps)
    env_ppo_hybrid.unwrapped.set_task(task)

    # RecurrentPPO的配置
    # policy_kwargs = dict(
    #     lstm_hidden_size=64,  # LSTM隐藏层大小
    #     n_lstm_layers=1,      # LSTM层数
    #     net_arch=dict(pi=[64], vf=[64])  # 网络架构
    # )

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

    while steps < eval_steps:
        action, lstm_states = model_hybrid.predict(
            state, 
            state=lstm_states,
            episode_start=done,
            deterministic=True
        )
        state, reward, terminated, truncated, info = env_ppo_hybrid.step(action)
        done = terminated or truncated
        acc_reward_hybrid += reward
        epoch_reward += reward
        steps += 1
        
        if steps % prt_freq == 0 and steps > 0:
            print(f"Step:{steps}\tEpoch Reward: {epoch_reward}")
            
            epoch_reward = 0
            
        if done:
            state, info = env_ppo_hybrid.reset()
            lstm_states = None
            episode_starts = True
        else:
            episode_starts = False
            
    print(f"PPO-HYBRID Policy Summary: {acc_reward_hybrid}")
    env_ppo_hybrid.unwrapped.visualize_and_save("ppo_hybrid_policy_result.pdf")

    # 在实验总结部分添加对比
    print("\n=== Experiment Summary ===")
    if args.seed is not None:
        print(f"Used specified seed: {args.seed}")
    else:
        current_seed = GLOBAL_SEED if GLOBAL_SEED is not None else "Random (No seed specified)"
        print(f"Random seed used: {current_seed}")
    print(f"RANDOM POLICY total reward: {acc_reward}")
    print(f"PPO-MLP total reward: {acc_reward_mlp}")
    print(f"PPO-HYBRID total reward: {acc_reward_hybrid}")
    print("\nTest Passed")