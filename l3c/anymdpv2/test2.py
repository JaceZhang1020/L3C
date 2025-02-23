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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import json

GLOBAL_SEED = None  # 默认为None表示随机

def validate_task_params(task):
    required_fields = [
        'mode', 'box_size', 'state_dim', 'action_dim', 
        'ndim', 'max_steps', 'action_weight'
    ]
    
    for field in required_fields:
        if field not in task:
            raise ValueError(f"Missing required field: {field}")
            
    if task['mode'] not in ['static', 'dynamic', 'universal']:
        raise ValueError(f"Invalid mode: {task['mode']}")
        
    if task['ndim'] < 1:
        raise ValueError("ndim must be positive")

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

def gen_uniform_matrix(n_in, n_out):
    w = np.random.normal(size=[n_out, n_in])
    u, s, vt = np.linalg.svd(w)
    s = np.diag(np.ones_like(s) * np.random.uniform(low=0.5, high=3))

    sm = np.zeros((n_out, n_in))
    np.fill_diagonal(sm, s)
    return u @ sm @ vt

def weights_and_biases(n_in, n_out, need_bias=False):
    weights = gen_uniform_matrix(n_in, n_out)
    if(need_bias):
        bias = 0.1 * np.random.normal(size=[n_out])
    else:
        bias = np.zeros(shape=[n_out])
    return weights, bias

def actfunc(raw_name):
    if(raw_name is None):
        name = 'none'
    else:
        name = raw_name.lower()
    if(name=='sigmoid'):
        return lambda x: 1/(1+np.exp(-x))
    elif(name=='tanh'):
        return np.tanh
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
        return lambda x:x
    else:
        raise ValueError(f"Invalid activation function name: {name}")

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_rewards = []
        self.episode_reward = 0
        
    def _on_step(self):
        # 记录每一步的即时奖励，而不是等到episode结束
        self.training_rewards.append(self.locals["rewards"][0])
        return True

def downsample_rewards(rewards, window_size):
    if len(rewards) == 0:
        return []
    
    # 确保数组长度能够被window_size整除
    pad_size = window_size - (len(rewards) % window_size) if len(rewards) % window_size != 0 else 0
    if pad_size:
        rewards = np.pad(rewards, (0, pad_size), 'edge')
    
    # 重塑数组并计算平均值
    return np.mean(rewards.reshape(-1, window_size), axis=1)

def sample_action_mapping(task):
    if not all(k in task for k in ['ndim', 'action_dim']):
        raise ValueError("Task must contain 'ndim' and 'action_dim'")
    
    ndim = task['ndim']
    action_dim = task['action_dim']
    s_map = RandomMLP(ndim, ndim, activation='tanh', biases=True)
    a_map = RandomMLP(action_dim, ndim, activation='tanh', biases=False)
    func = lambda s,a: s_map(s) + a_map(a)
    
    return {"action_map": func}

def sample_observation_mapping(task):
    ndim = task['ndim']
    observation_dim = task['state_dim']
    hidden_dim = random.randint(max(ndim - 3, 3), ndim + 1)
    
    observation_map = RandomMLP(
        ndim, 
        observation_dim,
        n_hidden_layers=hidden_dim,
        activation=['none','tanh'],
        biases=[True, False]
    )
    return {"observation_map": observation_map}

def sample_born_loc(task):
    if 'box_size' not in task:
        raise ValueError("Task must contain 'box_size'")
        
    born_loc_num = random.randint(1, 10)
    box_size = task['box_size']
    
    born_loc = [
        (
            np.random.uniform(
                -0.9 * box_size, 
                0.9 * box_size, 
                size=(task['ndim'],)
            ),
            np.random.exponential(0.10,)
        ) for i in range(born_loc_num)
    ]
    
    return {"born_loc": born_loc}

def sample_universal_reward(task):
    ndim = task['ndim']
    random_reward_fields = RandomMLP(
        ndim, 
        1,
        n_hidden_layers=random.randint(ndim * 2, ndim * 4),
        activation=['sin', 'tanh'],
        biases=[True, False]
    )
    
    factor = np.random.exponential(1.0)
    func = lambda x: factor * max(random_reward_fields(x) - 0.5, 0.0)
    
    return {"random_reward_fields": func}

def sample_goal_statictrigger(task, num=None):
    """采样静态触发目标"""
    if num is None:
        goal_num = random.randint(1, 10)
    else:
        goal_num = num
        
    # 收集已有位置以避免重叠
    repetitive_loc = [loc for loc, _ in task['born_loc']]
    for goal in task["goals"]:
        if 't' in goal.reward_type:  # 如果是触发型奖励
            repetitive_loc.append(goal.position(0))
            
    # 创建新目标
    for _ in range(goal_num):
        goal = RandomGoal(
            task['ndim'],
            repetitive_position=repetitive_loc,
            type='static',
            reward_type='t',
            box_size=task['box_size']
        )
        repetitive_loc.append(goal.position(0))
        task["goals"].append(goal)

def sample_goal_dynamic(task):
    """采样动态目标"""
    # 动态目标使用傅里叶函数生成轨迹
    task["goals"].append(
        RandomGoal(
            task['ndim'],
            type='fourier',  # 使用傅里叶类型
            reward_type='f',  # 使用场景奖励
            box_size=task['box_size']
        )
    )

def sample_goal_pitfalls(task, num=None):
    """采样陷阱目标"""
    if num is None:
        # 有50%概率不生成陷阱,上限是150个陷阱
        goal_num = max(0, random.randint(-50, 150))
    else:
        goal_num = num
        
    # 收集已有位置以避免重叠
    repetitive_loc = [loc for loc, _ in task['born_loc']]
    for goal in task["goals"]:
        if 't' in goal.reward_type:
            repetitive_loc.append(goal.position(0))
            
    # 创建陷阱
    for _ in range(goal_num):
        goal = RandomGoal(
            task['ndim'],
            repetitive_position=repetitive_loc,
            is_pitfall=True,  # 设置为陷阱
            type='static',
            reward_type='t',
            box_size=task['box_size']
        )
        repetitive_loc.append(goal.position(0))
        task["goals"].append(goal)

def sample_goal_potential_energy(task, num=None):
    """采样势能型目标"""
    if num is None:
        # 有50%概率不生成势能目标,上限是5个
        goal_num = max(0, random.randint(-5, 5))
    else:
        goal_num = num
        
    # 创建势能目标
    for _ in range(goal_num):
        task["goals"].append(
            RandomGoal(
                task['ndim'],
                is_pitfall=random.choice([True, False]),  # 随机决定是正向还是负向势能
                type='static',
                reward_type='p',  # 使用势能型奖励
                box_size=task['box_size']
            )
        )

class RandomMLP(object):
    '''
    A class for generating random MLPs with given parameters
    '''
    def __init__(self, n_inputs, n_outputs, 
                 n_hidden_layers=None, 
                 activation=None, 
                 biases=False,
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

        if activation is None:
            for _ in range(len(self.hidden_units)-1):
                self.activation.append(actfunc(None))
        elif isinstance(activation, list):
            assert len(activation) == len(self.hidden_units) - 1
            for hidden_act in activation:
                self.activation.append(actfunc(hidden_act))
        elif isinstance(activation, str):
            for _ in range(len(self.hidden_units)-1):
                self.activation.append(actfunc(activation))
        
        # Initialize weights and biases to random values
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_units)-1):
            if(isinstance(biases, list)):
                assert len(biases) == len(self.hidden_units) - 1
                w, b = weights_and_biases(self.hidden_units[i], self.hidden_units[i+1], need_bias=biases[i])
            else:
                w, b = weights_and_biases(self.hidden_units[i], self.hidden_units[i+1], need_bias=biases)
            self.weights.append(w)
            self.biases.append(b)
            
    def forward(self, inputs):
        outputs = inputs
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            outputs = self.activation[i](weight @ outputs + bias)
        if(np.size(outputs) > 1):
            return outputs
        else:
            return outputs[0]
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

class RandomFourier(object):
    def __init__(self,
                 ndim,
                 max_order=16,
                 max_item=5,
                 max_steps=1000,
                 box_size=2):
        n_items = random.randint(0, max_item + 1)
        self.coeffs = [(0, np.random.normal(size=(ndim, 2)) * np.random.exponential(scale=box_size / np.sqrt(n_items), size=(ndim, 2)))]
        self.max_steps = max_steps
        for j in range(n_items):
            # Sample a cos nx + b cos ny
            order = random.randint(1, max_order + 1) + np.random.normal(scale=1.0)
            factor = np.random.normal(size=(ndim, 2)) * np.random.exponential(scale=1.0, size=(ndim, 2))
            self.coeffs.append((order, factor))

    def __call__(self, t):
        # calculate a cos nx + b cos ny with elements of [t, [a, b]]
        x = t / self.max_steps
        y = 0
        for order, coeff in self.coeffs:
            y += coeff[:,0] * np.sin(order * x) + coeff[:,1] * np.cos(order * x)
        return y

class RandomGoal(object):
    def __init__(self,
                 ndim,
                 type='static',
                 reward_type='p',
                 repetitive_position=None,
                 repetitive_distance=0.2,
                 is_pitfall=False,
                 max_try=10000,
                 box_size=2):
        # Type: static, fourier
        # Reward type: field (f), trigger (t), potential (p) or combination (e.g., `ft`, `pt`)
        # Pitfall: if True, the goal is a pitfall, otherwise it is a goal
        eff_factor = np.sqrt(ndim)
        eff_rd = repetitive_distance * eff_factor
        self.reward_type = reward_type
        self.is_pitfall = is_pitfall
        if(type == 'static'):
            overlapped = True
            ntry = 0
            while overlapped and ntry < max_try:
                position = np.random.uniform(low=-box_size, high=box_size, size=(ndim, ))

                overlapped = False
                
                if(repetitive_position is None):
                    break

                for pos in repetitive_position:
                    dist = np.linalg.norm(pos - position)
                    if(dist < eff_rd):
                        overlapped = True
                        break
                ntry += 1
            if(ntry >= max_try):
                raise RuntimeError(f"Failed to generate goal position after {max_try} tries.")
            self.position = lambda t:position
        elif(type == 'fourier'):
            self.position = RandomFourier(ndim)
        else:
            raise ValueError(f"Invalid goal type: {type}")
        self.activate()

        self.has_field_reward=False
        self.has_trigger_reward=False
        self.has_potential_reward=False

        if('f' in self.reward_type): 
            self.field_reward = random.uniform(0.2, 0.8)
            self.field_threshold = np.random.exponential(box_size / 2) * eff_factor
            self.has_field_reward = True
        if('t' in self.reward_type): 
            self.trigger_reward = max(np.random.exponential(5.0), 1.0)
            self.trigger_threshold = random.uniform(0.20, 0.50) * eff_factor
            if(is_pitfall):
                self.trigger_threshold += box_size / 4
            self.trigger_rs_terminal = self.trigger_reward
            self.trigger_rs_threshold = 3 * box_size * eff_factor
            self.trigger_rs_potential = self.trigger_reward * self.trigger_rs_threshold / box_size
            self.has_trigger_reward = True
        if('p' in self.reward_type): 
            self.potential_reward = max(np.random.exponential(2.0), 0.5)
            self.potential_threshold = random.uniform(box_size/2, box_size) * eff_factor
            self.has_potential_reward = True

    def activate(self):
        self.is_activated = True

    def deactivate(self):
        self.is_activated = False

    def __call__(self, sp, sn, t=0, need_reward_shaping=False):
        # input previous state, next state        
        # output reward, done
        if(not self.is_activated):
            return 0.0, False, {}
        reward = 0
        shaped_reward = 0
        done = False
        cur_pos = self.position(t)
        dist = np.linalg.norm(sn - cur_pos)
        distp = np.linalg.norm(sp - cur_pos)
        if(self.has_field_reward):
            if(dist <= 3.0 * self.field_threshold):
                k = dist / self.field_threshold
                reward += self.field_reward * np.exp(- k ** 2)
        if(self.has_trigger_reward):
            if(dist <= self.trigger_threshold):
                reward += self.trigger_reward
                if(need_reward_shaping):
                    shaped_reward += self.trigger_rs_terminal - self.trigger_reward
                done = True
            if(need_reward_shaping):
                if(dist <= self.trigger_rs_threshold):
                    shaped_reward += self.trigger_rs_potential * (min(distp, self.trigger_rs_threshold) - dist) / self.trigger_rs_threshold
            #print(f"dist: {dist}, distp: {distp}, reward: {shaped_reward}, \
            #      trigger_rs_threshold: {self.trigger_rs_threshold}")
        if(self.has_potential_reward):
            if(dist <= self.potential_threshold):
                reward += self.potential_reward * (min(distp, self.potential_threshold) - dist) / self.potential_threshold
        shaped_reward += reward
        if(self.is_pitfall):
            reward *= -1
            shaped_reward = 0
        return reward, done, {'shaped_reward':shaped_reward}

class AnyMDPEnv(gym.Env):
    def __init__(self, max_steps):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.max_steps = max_steps
        self.task_set = False
        self.success_count = 0

    def set_task(self, task, verbose=False, reward_shaping=False):
        for key in task:
            setattr(self, key, task[key])
        # 定义无界的 observation_space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                          shape=(self.state_dim,), dtype=float)
        # 定义 action_space
        self.action_space = spaces.Box(low=-1, high=1, 
                                     shape=(self.action_dim,), dtype=float)
        self.task_set = True
        self.need_reset = True
        self.reward_shaping = reward_shaping
        if verbose:
            print('Task Mode:', self.mode)
            print('ndim:', self.ndim)

    def reset(self, seed=None, options=None):
        if not self.task_set:
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        loc, noise = random.choice(self.born_loc)
        self._inner_state = loc + noise * np.random.normal(size=self.ndim)
        self._state = self.observation_map(self._inner_state)
        
        return self._state, {"steps": self.steps}

    def step(self, action):
        if self.need_reset or not self.task_set:
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert np.shape(action) == (self.action_dim,)

        # update inner state
        inner_deta = self.action_map(self._inner_state, action)
        next_inner_state = (self._inner_state + 
                           inner_deta * self.action_weight + 
                           self.transition_noise * np.random.normal(size=(self.ndim,)))

        # Essential Rewards
        reward = self.average_cost
        done = False
        self._last_success = False

        for goal in self.goals:
            r, d, info = goal(self._inner_state, next_inner_state, 
                            t=self.steps, need_reward_shaping=self.reward_shaping)
            if self.reward_shaping:
                r = info["shaped_reward"]
            reward += r
            done = done or d
            if d:
                self._last_success = True

        if self._last_success:
            self.success_count += 1

        # Random reward fields if exists
        if hasattr(self, 'random_reward_fields'):
            reward += self.random_reward_fields(self._inner_state)

        # Add noise to reward
        if abs(reward) > 0.5:
            reward *= 1.0 + self.reward_noise * np.random.normal()

        self.steps += 1
        info = {"steps": self.steps}
        
        self._inner_state = next_inner_state
        self._state = self.observation_map(self._inner_state)

        # Check if out of bounds
        oob = (np.abs(self._inner_state) > self.box_size)
        truncated = (self.steps >= self.max_steps or oob.any())
        done = done or truncated
        
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
    def set_task(self, task, verbose=False, reward_shaping=False):
        # Set task will automatically reset all records
        super().set_task(task)
        self.observation_records = []
        self.inner_state_records = []
        self.action_records = []
        self.reward_records = []
    
    def color_spec(self, i):
        return [self.color_spec_type[i][idx] for idx in self.colors]
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)  # 正确获取父类返回值
        self.observation_records.append(np.copy(obs))
        self.action_records.append(np.zeros((self.action_dim,)))
        self.inner_state_records.append(np.copy(self.inner_state))
        self.reward_records.append(0.0)

        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)  
        self.observation_records.append(np.copy(obs))
        self.inner_state_records.append(np.copy(self.inner_state))
        self.action_records.append(np.copy(action))
        self.reward_records.append(reward)
        return obs, reward, terminated, truncated, info  # 


    def visualize_and_save(self, filename=None, training_rewards=None):
        if training_rewards is not None:
            all_rewards = training_rewards[:train_steps] + self.reward_records[:eval_steps]
        else:
            all_rewards = self.reward_records

        window_size = 100
        
        # 降采样
        downsampled_rewards = downsample_rewards(np.array(all_rewards), window_size)
        x_coords = np.arange(len(downsampled_rewards)) * window_size

        # plt.figure(figsize=(15, 12))
        
        # tsne = TSNE(n_components=2, random_state=pseudo_random_seed(),
        #             perplexity=10, max_iter=500, learning_rate=100)
        # obs_tsne = tsne.fit_transform(np.array(self.observation_records))
        
        # plt.subplot(2, 2, 1)
        # plt.scatter(obs_tsne[:, 0], obs_tsne[:, 1], c='black', s=10, alpha=0.2)
        # plt.title("Observation", fontsize=12, fontweight='bold', color='blue', pad=10)

        # act_tsne = tsne.fit_transform(np.array(self.action_records))
        # plt.subplot(2, 2, 2)
        # plt.scatter(act_tsne[:, 0], act_tsne[:, 1], c='black', s=10, alpha=0.2)
        # plt.title("Action", fontsize=12, fontweight='bold', color='blue', pad=10)

        # s_tsne = tsne.fit_transform(np.array(self.inner_state_records))
        # plt.subplot(2, 2, 3)
        # scatter = plt.scatter(s_tsne[:, 0], s_tsne[:, 1], c=self.reward_records, 
        #                     cmap='viridis', s=10, alpha=0.2, marker='o')
        # plt.colorbar(scatter)
        # plt.title("States", fontsize=12, fontweight='bold', color='blue', pad=10)

        # plt.subplot(2, 2, 4) 
        rewards_smooth = savitzky_golay(downsampled_rewards, 
                                    window_size=min(99, len(downsampled_rewards)-1), 
                                    order=3)
        
        # 绘制到第四个subplot
        plt.plot(x_coords, downsampled_rewards, c='red', alpha=0.2, label='Raw')
        plt.plot(x_coords, rewards_smooth, c='red', label='Smoothed')
        plt.title("Reward", fontsize=12, fontweight='bold', color='blue', pad=10)
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend()

        # 统一调整布局和保存
        plt.tight_layout()
        plt.savefig(filename or "./anymdp_visualizer_output.pdf", 
                bbox_inches='tight', dpi=300)
        plt.close()

        # rewards_smooth = savitzky_golay(self.reward_records, window_size=99, order=3)
        # scatter = plt.plot(np.arange(np.size(self.reward_records)), 
        #                  self.reward_records, c='red', alpha=0.2)
        # scatter = plt.plot(np.arange(np.size(rewards_smooth)), 
        #                  rewards_smooth, c='red')
        # plt.title("Reward", fontsize=12, fontweight='bold', color='blue', pad=10)
        # plt.savefig(file_name)

def create_task(
    state_dim: int = 256, 
    action_dim: int = 256, 
    mode=None, 
    ndim=None,
) -> Dict[str, Any]:
    if GLOBAL_SEED is not None:
        random.seed(GLOBAL_SEED)
        np.random.seed(GLOBAL_SEED)  
    else:
        random.seed(pseudo_random_seed())
        np.random.seed(pseudo_random_seed())
    
    # 基础任务配置
    task = dict()
    task["mode"] = mode if mode is not None else random.choice(["static", "dynamic", "universal"])
    task["box_size"] = 2
    task["state_dim"] = state_dim
    task["action_dim"] = action_dim
    task["ndim"] = ndim if ndim is not None else random.randint(4, 16)
    task["max_steps"] = random.randint(100, 1000)
    task["action_weight"] = np.random.uniform(0.01, 0.05, size=(task['ndim'],))
    task["average_cost"] = np.random.exponential(0.01) * random.choice([-2, -1, 0, 1])
    task["transition_noise"] = max(0, np.random.normal(scale=1.0e-4))
    task["reward_noise"] = max(0, np.random.normal(scale=1.0e-4))
    
    # 初始化目标列表
    task["goals"] = []
    
    # 参数验证
    validate_task_params(task)
    
    # 观测映射
    task.update(sample_observation_mapping(task))
    
    # 动作映射
    task.update(sample_action_mapping(task))
    
    # 出生点设置
    task.update(sample_born_loc(task))
    
    # 根据模式设置目标和奖励
    if task["mode"] == "static":
        sample_goal_statictrigger(task, num=1)
    elif task["mode"] == "dynamic":
        sample_goal_dynamic(task)
    elif task["mode"] == "universal":
        task.update(sample_universal_reward(task))
        
    # 额外目标设置
    if random.random() < 0.7:
        sample_goal_pitfalls(task)
    if random.random() < 0.5:
        sample_goal_potential_energy(task)
        
    return task

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--mode', type=str, choices=['static','dynamic','multi','consis'], 
                       default=None, help='Force task mode')
    parser.add_argument('--ndim', type=int, default=None, 
                       help='Force state dimension (3-32)')
    args = parser.parse_args()

    # 如果没有指定seed，则生成一个随机种子
    if args.seed is None:
        current_time = int(time.time() * 1000)
        args.seed = current_time % 100000
        
    set_global_seed(args.seed)
    print(f"Using seed: {args.seed}")

    # 设置参数
    max_steps = 80000      
    prt_freq = 10000       
    train_steps = 80000  
    eval_steps = 80000
    
    # 创建任务
    task = create_task(
        state_dim=64,     
        action_dim=16,    
        mode=args.mode, 
        ndim=args.ndim
    )

    # 保存任务信息
    task_info = {
        'seed': GLOBAL_SEED,
        'mode': task['mode'],
        'ndim': task['ndim'],
        'max_steps': task['max_steps'],
        'transition_noise': task['transition_noise'],
        'reward_noise': task['reward_noise']        
    }
    
    with open('task_info.json', 'w') as f:
        json.dump(task_info, f, indent=4)

    print("\n=== Testing Random Policy ===")
    env_random = AnyMDPVisualizer(max_steps=max_steps)
    env_random.set_task(task)

    state, info = env_random.reset()
    acc_reward = 0
    success_count_random = 0
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
            if env_random.unwrapped._last_success:  
                success_count_random += 1
            state, info = env_random.reset()
            
    avg_reward_random = acc_reward / steps if steps > 0 else 0
    print(f"Random Policy Summary: Cumulative Reward: {acc_reward}, Average Reward per Step: {avg_reward_random}")
    env_random.unwrapped.visualize_and_save("random_policy_result.pdf", training_rewards=None)

    print("\n=== Testing PPO-MLP Policy ===")
    env_ppo_mlp = AnyMDPVisualizer(max_steps=train_steps)
    env_ppo_mlp.set_task(task, verbose=True, reward_shaping=True)

    model_mlp = PPO(
        "MlpPolicy",
        env_ppo_mlp, 
        verbose=1,
        seed=GLOBAL_SEED,
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,
    )

    # 评估初始性能
    state, info = env_ppo_mlp.reset()
    acc_reward_mlp_pre = 0
    steps = 0
    success_count_mlp_pre = 0
    while steps < eval_steps:
        action, _ = model_mlp.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env_ppo_mlp.step(action)
        acc_reward_mlp_pre += reward
        steps += 1
        if terminated or truncated:
            if env_ppo_mlp.unwrapped._last_success:
                success_count_mlp_pre += 1
            state, info = env_ppo_mlp.reset()
    avg_reward_mlp_pre = acc_reward_mlp_pre / steps if steps > 0 else 0
    print(f"PPO-MLP Pre-training Performance: Cumulative Reward: {acc_reward_mlp_pre}, Average Reward per Step: {avg_reward_mlp_pre}, Successes: {success_count_mlp_pre}")

    # 训练
    model_mlp.learn(total_timesteps=train_steps)

    # 评估训练后性能  
    state, info = env_ppo_mlp.reset()
    acc_reward_mlp_post = 0
    steps = 0
    success_count_mlp_post = 0
    while steps < eval_steps:
        action, _ = model_mlp.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env_ppo_mlp.step(action)
        acc_reward_mlp_post += reward
        steps += 1
        if terminated or truncated:
            if env_ppo_mlp.unwrapped._last_success:
                success_count_mlp_post += 1
            state, info = env_ppo_mlp.reset()
    avg_reward_mlp_post = acc_reward_mlp_post / steps if steps > 0 else 0
    print(f"PPO-MLP Post-training Performance: Cumulative Reward: {acc_reward_mlp_post}, Average Reward per Step: {avg_reward_mlp_post}, Successes: {success_count_mlp_post}")
    env_ppo_mlp.unwrapped.visualize_and_save("ppo_mlp_policy_result.pdf")

    print("\n=== Testing PPO-HYBRID Policy ===")
    env_ppo_hybrid = AnyMDPVisualizer(max_steps=train_steps)
    env_ppo_hybrid.set_task(task, verbose=True, reward_shaping=True)

    model_hybrid = RecurrentPPO(
        "MlpLstmPolicy",
        env_ppo_hybrid,
        verbose=1,
        seed=GLOBAL_SEED,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        # clip_range=0.2,
        # clip_range_vf=0.2,
        # max_grad_norm=0.5
        # ent_coef=0.01
        # gae_lambda=0.95,
        policy_kwargs=dict(
            lstm_hidden_size=32,
            n_lstm_layers=2,
            enable_critic_lstm=True,
            net_arch=dict(
                pi=[64, 32],
                vf=[64, 32]
            ),
            activation_fn=nn.ReLU
        ),
    )

    # 评估初始性能
    state, info = env_ppo_hybrid.reset()
    acc_reward_hyb_pre = 0
    steps = 0
    success_count_hyb_pre = 0
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
        acc_reward_hyb_pre += reward
        steps += 1
        if terminated or truncated:
            if env_ppo_hybrid.unwrapped._last_success:
                success_count_hyb_pre += 1
            state, info = env_ppo_hybrid.reset()
            lstm_states = None
            done = True
        else:
            done = False
    avg_reward_hyb_pre = acc_reward_hyb_pre / steps if steps > 0 else 0
    print(f"PPO-HYBRID Pre-training Performance: Cumulative Reward: {acc_reward_hyb_pre}, Average Reward per Step: {avg_reward_hyb_pre}, Successes: {success_count_hyb_pre}")

    # 训练
    model_hybrid.learn(total_timesteps=train_steps)

    # 评估训练后性能
    state, info = env_ppo_hybrid.reset()
    acc_reward_hyb_post = 0
    steps = 0
    success_count_hyb_post = 0
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
        acc_reward_hyb_post += reward
        steps += 1
        if terminated or truncated:
            if env_ppo_hybrid.unwrapped._last_success:
                success_count_hyb_post += 1
            state, info = env_ppo_hybrid.reset()
            lstm_states = None
            done = True
        else:
            done = False
    avg_reward_hyb_post = acc_reward_hyb_post / steps if steps > 0 else 0
    print(f"PPO-HYBRID Post-training Performance: Cumulative Reward: {acc_reward_hyb_post}, Average Reward per Step: {avg_reward_hyb_post}, Successes: {success_count_hyb_post}")
    env_ppo_hybrid.unwrapped.visualize_and_save("ppo_hybrid_policy_result.pdf")
    # # 评估初始性能
    # mean_reward, std_reward = evaluate_policy(model_hybrid, env_ppo_hybrid, n_eval_episodes=10)
    # print(f"Before Training: Mean reward: {mean_reward}, Std reward: {std_reward}")

    # # 训练
    # model_hybrid.learn(total_timesteps=train_steps)

    # # 评估训练后性能
    # mean_reward, std_reward = evaluate_policy(model_hybrid, env_ppo_hybrid, n_eval_episodes=10)
    # env_ppo_hybrid.visualize_and_save("ppo_hyb_result.pdf")
    # print(f"After Training: Mean reward: {mean_reward}, Std reward: {std_reward}")

    # 实验总结
    print("\n=== Experiment Summary ===")
    if args.seed is not None:
        print(f"Used specified seed: {args.seed}")
    else:
        current_seed = GLOBAL_SEED if GLOBAL_SEED is not None else "Random (No seed specified)"
        print(f"Random seed used: {current_seed}")
    print(f"RANDOM POLICY  | Cumulative Reward: {acc_reward}, Average: {avg_reward_random}, Successes: {success_count_random}")
    print(f"PPO-MLP      Pre-training | Cumulative Reward: {acc_reward_mlp_pre}, Average: {avg_reward_mlp_pre}, Successes: {success_count_mlp_pre}", )
    print(f"PPO-MLP      Post-training| Cumulative Reward: {acc_reward_mlp_post}, Average: {avg_reward_mlp_post}, Successes: {success_count_mlp_post}")
    print(f"PPO-HYBRID   Pre-training | Cumulative Reward: {acc_reward_hyb_pre}, Average {avg_reward_hyb_pre}, Successes: {success_count_hyb_pre}")
    print(f"PPO-HYBRID   Post-training| Cumulative Reward: {acc_reward_hyb_post}, Average: {avg_reward_hyb_post}, Successes: {success_count_hyb_post}")
    print("\nTest Passed")