"""
How to run
    python -m fip_env.agents.fip_solver --train

"""
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID: str = 'fip_env/FlywheelInvertedPendulum-v0'


def _get_env(**kwargs: Any):
    return gym.make(ENV_ID, **kwargs)


def _train(env: Env, model: BaseAlgorithm):
    # Stop training when the mean reward reaches 500
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        eval_freq=1000,  # Evaluate every 1000 steps
        verbose=1,
        log_path=f'logs/{ENV_ID}'
    )

    # Train the model
    model.learn(total_timesteps=500000, callback=eval_callback)


def render_logs():
    filename = f"logs/{ENV_ID}/evaluations.npz"
    data = np.load(filename)
    timesteps = data['timesteps']
    results = data['results']  # Shape: (n_evaluations, n_eval_episodes)
    ep_lengths = data['ep_lengths']  # Shape: (n_evaluations, n_eval_episodes)

    # Compute mean reward and mean episode length for each evaluation
    mean_rewards = results.mean(axis=1)
    mean_ep_lengths = ep_lengths.mean(axis=1)

    # Print the results
    print("Timesteps:", timesteps)
    print("Mean Rewards:", mean_rewards)

    print("Mean Episode Lengths:", mean_ep_lengths)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label='Mean Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Evaluation Mean Reward Over Time')
    plt.legend()
    plt.grid()
    plt.show()


def main(should_train: bool = True, render: bool = False):
    if render:
        render_logs()
        return

    filename = "fip_solver.pth"
    if should_train:
        env = _get_env()
        model = PPO(
            "MlpPolicy",  # Policy network (Multi-layer Perceptron)
            env,
            verbose=1,  # Print training logs
            learning_rate=3e-4,
            n_steps=2048,  # Number of steps per update
            batch_size=64,  # Batch size for training
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # Generalized Advantage Estimation lambda
            ent_coef=0.035,  # Entropy coefficient for exploration
            max_grad_norm=0.5,  # Gradient clipping
        )

        _train(env, model)

        eval_env = _get_env(theta_threshold=np.pi)
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

        model.save(filename)
    else:
        model = PPO.load(filename)
        env = _get_env(kick_probability=0.3, theta_threshold=np.pi, verbose_termination=True, render_mode="human")
        obs, _ = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    main(should_train=args.train, render=args.render)
