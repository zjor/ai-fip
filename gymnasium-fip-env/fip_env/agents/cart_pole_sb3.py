"""
How to run
    python -m fip_env.agents.cart_pole_sb3
"""
import gymnasium as gym
from gymnasium import Env
from pydantic.v1.utils import truncate
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from sympy import trunc

ENV_ID = "CartPole-v1"


def _train(env: Env, model: BaseAlgorithm):
    # Stop training when the mean reward reaches 500
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        eval_freq=1000,  # Evaluate every 1000 steps
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=200000, callback=eval_callback)


def main(should_train: bool = True):
    filename = "ppo_cartpole.pth"
    if should_train:
        env = gym.make(ENV_ID)
        model = PPO(
            "MlpPolicy",  # Policy network (Multi-layer Perceptron)
            env,
            verbose=1,  # Print training logs
            learning_rate=3e-4,
            n_steps=2048,  # Number of steps per update
            batch_size=64,  # Batch size for training
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # Generalized Advantage Estimation lambda
            ent_coef=0.01,  # Entropy coefficient for exploration
            max_grad_norm=0.5,  # Gradient clipping
        )

        _train(env, model)

        eval_env = gym.make(ENV_ID)
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

        model.save(filename)
    else:
        model = PPO.load(filename)
        env = gym.make(ENV_ID, render_mode="human")
        obs, _ = env.reset()
        for _ in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs = env.reset()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    main(should_train=args.train)
