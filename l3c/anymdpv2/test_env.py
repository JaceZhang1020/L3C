def test_env_rewards(env, model, max_steps, n_tests=5):
    """
    测试环境的reward增长情况
    返回每次测试的reward序列和增长率
    """
    all_stats = []
    
    for test_i in range(n_tests):
        # 每次测试前重新创建和设置任务
        task = AnyMDPv2TaskSampler(state_dim=args.state_dim, 
                                  action_dim=args.action_dim)
        env.set_task(task, verbose=True, reward_shaping=True)
        
        episode_returns = []
        total_steps = 0
        episodes = 0
        
        while total_steps < max_steps:
            # 确保在开始新的episode时重置环境
            try:
                state_info = env.reset()
                if isinstance(state_info, tuple):
                    state = state_info[0]
                else:
                    state = state_info
            except ValueError:
                state = env.reset()
            
            done = False
            episode_reward = 0
            
            while not done and total_steps < max_steps:
                action, _ = model.predict(state, deterministic=False)
                try:
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                except ValueError:
                    next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                total_steps += 1
                state = next_state
            
            episode_returns.append(episode_reward)
            episodes += 1
        
        if len(episode_returns) >= 2:
            mid_point = len(episode_returns) // 2
            first_half_avg = numpy.mean(episode_returns[:mid_point])
            second_half_avg = numpy.mean(episode_returns[mid_point:])
            
            if abs(first_half_avg) < 1e-6:
                growth_rate = second_half_avg - first_half_avg
            else:
                growth_rate = (second_half_avg - first_half_avg) / abs(first_half_avg)
        else:
            growth_rate = 0
            first_half_avg = 0
            second_half_avg = 0
            
        stats = {
            "episode_returns": episode_returns,
            "growth_rate": growth_rate,
            "first_half_avg": first_half_avg,
            "second_half_avg": second_half_avg,
            "episodes": episodes
        }
        all_stats.append(stats)
        
    return all_stats

if __name__=="__main__":
    import gym
    import numpy
    import argparse
    from l3c.anymdpv2 import AnyMDPv2TaskSampler
    from stable_baselines3 import PPO, SAC
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    import torch.nn as nn

    args = argparse.ArgumentParser()
    args.add_argument("--max_step", type=int, default=80000)
    args.add_argument("--lr", type=float, default=3e-4)
    args.add_argument("--run", choices=["mlp", "lstm", "sac","all"], default="all")
    args.add_argument("--n_env_tests", type=int, default=5, help="Number of environments to test")
    args.add_argument("--state_dim", type=int, default=64)
    args.add_argument("--action_dim", type=int, default=16)
    args = args.parse_args()

    max_step = args.max_step
    lr = args.lr

    # 存储所有测试结果
    all_env_results = []

    # 测试多个环境
    for env_test in range(args.n_env_tests):
        print(f"\nTesting environment {env_test + 1}/{args.n_env_tests}")
        
        # 创建环境
        env = gym.make("anymdp-v2-visualizer")
        
        # 创建任务并设置
        task = AnyMDPv2TaskSampler(state_dim=args.state_dim, 
                                  action_dim=args.action_dim)
        env.set_task(task, verbose=True, reward_shaping=True)
        
        # 重置环境
        env.reset()

        # 创建模型
        model_mlp = PPO("MlpPolicy", env, verbose=1, learning_rate=lr, batch_size=64, gamma=0.99)
        model_lstm = RecurrentPPO(
            "MlpLstmPolicy", env, verbose=1, learning_rate=lr,
            n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95,
            policy_kwargs={
                "lstm_hidden_size": 32,
                "n_lstm_layers": 2,
                "enable_critic_lstm": True
            },
            clip_range=0.2,
        )
        model_sac = SAC(
            "MlpPolicy", env, verbose=0, learning_rate=3e-4,
            batch_size=256, buffer_size=1000000,
            learning_starts=100, train_freq=1, gradient_steps=1,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
                activation_fn=nn.ReLU
            ),
        )

        env_results = {
            "mlp": None,
            "lstm": None,
            "sac": None
        }

        if args.run == "mlp" or args.run == "all":
            print("\nTesting MLP Policy")
            env_results["mlp"] = test_env_rewards(env, model_mlp, max_step // 10)

        if args.run == "lstm" or args.run == "all":
            print("\nTesting LSTM Policy")
            env_results["lstm"] = test_env_rewards(env, model_lstm, max_step // 10)

        if args.run == "sac" or args.run == "all":
            print("\nTesting SAC Policy")
            env_results["sac"] = test_env_rewards(env, model_sac, max_step // 10)

        all_env_results.append(env_results)

    # 分析结果
    print("\nAnalysis of reward growth rates across all environments:")
    for model_type in ["mlp", "lstm", "sac"]:
        if all_env_results[0][model_type] is not None:
            growth_rates = []
            for env_result in all_env_results:
                for test in env_result[model_type]:
                    growth_rates.append(test["growth_rate"])
            
            print(f"\n{model_type.upper()} Statistics:")
            print(f"Mean growth rate: {numpy.mean(growth_rates):.4f}")
            print(f"Std growth rate: {numpy.std(growth_rates):.4f}")
            print(f"Min growth rate: {numpy.min(growth_rates):.4f}")
            print(f"Max growth rate: {numpy.max(growth_rates):.4f}")
            print(f"25th percentile: {numpy.percentile(growth_rates, 25):.4f}")
            print(f"50th percentile: {numpy.percentile(growth_rates, 50):.4f}")
            print(f"75th percentile: {numpy.percentile(growth_rates, 75):.4f}")