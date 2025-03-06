if __name__=="__main__":
    import gym
    import numpy
    import argparse
    from l3c.anymdpv2 import AnyMDPv2TaskSampler

    from stable_baselines3 import PPO, SAC
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3 import SAC
    import torch.nn as nn

    task = AnyMDPv2TaskSampler(state_dim=64, 
                             action_dim=16)

    env = gym.make("anymdp-v2-visualizer") 
    env.set_task(task, verbose=True, reward_shaping=True)

    args = argparse.ArgumentParser()
    args.add_argument("--max_step", type=int, default=80000)
    args.add_argument("--lr", type=float, default=3e-4)
    args.add_argument("--run", choices=["mlp", "lstm", "sac", "all"], default="all")
    args = args.parse_args()

    max_step = args.max_step
    lr = args.lr

    model_mlp = PPO(
        "MlpPolicy",      # 使用 MLP 策略网络
        env,                  # 环境对象
        verbose=1,            # 打印训练日志
        learning_rate=lr,   # 学习率
        batch_size=64,        # 批量大小
        gamma=0.99,           # 折扣因子
        # tensorboard_log="./ppo_tensorboard/"  # TensorBoard 日志目录
    )

    model_lstm = RecurrentPPO(
        "MlpLstmPolicy",      # 使用 MLP 策略网络
        env,                  # 环境对象
        verbose=1,            # 打印训练日志
        learning_rate=lr,   # 学习率
        n_steps=2048,         # 每个环境每次更新的步数
        batch_size=64,        # 批量大小
        n_epochs=10,          # 每次更新的迭代次数
        gamma=0.99,           # 折扣因子
        gae_lambda=0.95,      # GAE 参数
        policy_kwargs={
            "lstm_hidden_size": 32,    # LSTM 隐藏层大小
            "n_lstm_layers": 2,        # LSTM 层数
            "enable_critic_lstm": True # Critic 网络也使用 LSTM
        },
        clip_range=0.2,       # PPO 的 clip 范围
        # tensorboard_log="./ppo_tensorboard/"  # TensorBoard 日志目录
    )

    model_sac = SAC(  
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=3e-4,
                batch_size=256,
                buffer_size=1000000,
                learning_starts=100,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=dict(
                    net_arch=dict(
                        pi=[256, 256],
                        qf=[256, 256]
                    ),
                    activation_fn=nn.ReLU
                ),
    )


    if(args.run == "mlp" or args.run == "all"):

        print(f"Training MLP Policy for {max_step} steps")

        mean_reward_mlp_pre, std_reward_mlp_pre = evaluate_policy(model_mlp, env, n_eval_episodes=10)
        print(f"Before Training: Mean reward: {mean_reward_mlp_pre}, Std reward: {std_reward_mlp_pre}")

        model_mlp.learn(total_timesteps=max_step)

        mean_reward_mlp_post, std_reward_mlp_post = evaluate_policy(model_mlp, env, n_eval_episodes=10)
        print(f"After Training: Mean reward: {mean_reward_mlp_post}, Std reward: {std_reward_mlp_post}")

    if(args.run == "lstm" or args.run == "all"):

        print(f"Training LSTM Policy for {max_step} steps")

        mean_reward_lstm_pre, std_reward_lstm_pre = evaluate_policy(model_lstm, env, n_eval_episodes=10)
        print(f"Before Training: Mean reward: {mean_reward_lstm_pre}, Std reward: {std_reward_lstm_pre}")

        model_lstm.learn(total_timesteps=max_step)

        mean_reward_lstm_post, std_reward_lstm_post = evaluate_policy(model_lstm, env, n_eval_episodes=10)
        print(f"After Training: Mean reward: {mean_reward_lstm_post}, Std reward: {std_reward_lstm_post}")

    if(args.run == "sac" or args.run == "all"):

        print(f"Training SAC Policy for {max_step} steps")

        mean_reward_sac_pre, std_reward_sac_pre = evaluate_policy(model_sac, env, n_eval_episodes=10)
        print(f"Before Training: Mean reward: {mean_reward_sac_pre}, Std reward: {std_reward_sac_pre}")

        model_sac.learn(total_timesteps=max_step)

        mean_reward_sac_post, std_reward_sac_post = evaluate_policy(model_lstm, env, n_eval_episodes=10)
        print(f"After Training: Mean reward: {mean_reward_sac_post}, Std reward: {std_reward_sac_post}")

        print(f"result summary")
        print(f"Before PPO-MLPTraining: Mean reward: {mean_reward_mlp_pre}, Std reward: {std_reward_mlp_pre}")
        print(f"After PPO-MLP Training: Mean reward: {mean_reward_mlp_post}, Std reward: {std_reward_mlp_post}")
        print(f"Before PPO-LSTM Training: Mean reward: {mean_reward_lstm_pre}, Std reward: {std_reward_lstm_pre}")
        print(f"After PPO-LSTM Training: Mean reward: {mean_reward_lstm_post}, Std reward: {std_reward_lstm_post}")
        print(f"Before SAC Training: Mean reward: {mean_reward_sac_pre}, Std reward: {std_reward_sac_pre}")
        print(f"After SAC Training: Mean reward: {mean_reward_sac_post}, Std reward: {std_reward_sac_post}")