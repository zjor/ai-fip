import os.path
import sys
from enum import Enum
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import Env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

ENV_ID: str = 'fip_env/FlywheelInvertedPendulum-v0'


class RLModel(Enum):
    PPO = 'ppo'
    SAC = 'sac'


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def _get_env(**kwargs: Any):
    return gym.make(ENV_ID, **kwargs)


def _train(env: Env, model: BaseAlgorithm):
    # Stop training when the mean reward reaches 500
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=1)
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        eval_freq=1500,  # Evaluate every 1000 steps
        verbose=1,
        log_path=f'logs/{ENV_ID}'
    )

    # Train the model
    model.learn(total_timesteps=1000_000, callback=eval_callback)


def render_logs():
    filename = f"logs/{ENV_ID}/evaluations.npz"
    data = np.load(filename)
    timesteps = data['timesteps']
    results = data['results']  # Shape: (n_evaluations, n_eval_episodes)
    ep_lengths = data['ep_lengths']  # Shape: (n_evaluations, n_eval_episodes)

    # Compute mean reward and mean episode length for each evaluation
    mean_rewards = results.mean(axis=1)
    mean_ep_lengths = ep_lengths.mean(axis=1)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label='Mean Reward')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.title('Evaluation Mean Reward Over Time')
    plt.legend()
    plt.grid()
    plt.show()


def train(model_type: RLModel, model_save_filename: str):
    env = _get_env(
        max_steps=500,
        dtheta_threshold=10,
        dphi_threshold=20.0,
        max_torque=4
    )
    if model_type == RLModel.PPO:
        model = PPO(
            "MlpPolicy",  # Policy network (Multi-layer Perceptron)
            env,
            verbose=1,  # Print training logs
            learning_rate=3e-4,
            n_steps=2048,  # Number of steps per update, prev. 2048
            batch_size=64,  # Batch size for training, prev 64
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # Generalized Advantage Estimation lambda
            ent_coef=0.025,  # Entropy coefficient for exploration
            clip_range=0.1,
            max_grad_norm=0.5,  # Gradient clipping
        )
    elif model_type == RLModel.SAC:
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            tau=0.005,  # Target network update rate
            gamma=0.99,  # Discount factor
            action_noise=action_noise,
            train_freq=(1, "episode"),  # Update the model every episode
            gradient_steps=1,
            verbose=1,
        )

    if os.path.exists(model_save_filename):
        model.set_parameters(model_save_filename)
        print(f"Loaded model from {model_save_filename}")

    _train(env, model)

    eval_env = _get_env()
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    model.save(model_save_filename)


def run_with_sb3(trained_model_filename: str):
    model = PPO.load(trained_model_filename)
    env = _get_env(kick_probability=0.3,
                   max_steps=500,
                   dtheta_threshold=10,
                   dphi_threshold=20,
                   max_torque=4,
                   verbose_termination=True, render_mode="human")
    obs, _ = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()


def run_with_onnx(trained_model_filename: str):
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(f"{trained_model_filename}.onnx", providers=["CPUExecutionProvider"])
    env = _get_env(kick_probability=0.3,
                   max_steps=500,
                   dtheta_threshold=10,
                   dphi_threshold=20,
                   max_torque=4,
                   verbose_termination=True, render_mode="human")
    obs, _ = env.reset()
    for _ in range(1000):
        ort_inputs = {ort_session.get_inputs()[0].name: obs.reshape(1, -1)}
        actions = ort_session.run(None, ort_inputs)
        obs, reward, terminated, truncated, info = env.step(actions[0][0])
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()


def main(model: RLModel, should_train: bool = True, should_render_logs: bool = False, should_use_onnx: bool = False):
    if should_render_logs:
        render_logs()
        return

    filename = "fip_solver"
    if should_train:
        train(model, filename + ".pth")
    else:
        if should_use_onnx:
            run_with_onnx(filename)
        else:
            run_with_sb3(filename + ".pth")


if __name__ == '__main__':
    """
    How to run
        python -m fip_env.agents.fip_solver --train --model=[PPO,SAC]

    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=RLModel, default=RLModel.PPO,
                        help=f'Supported values: {[m.value for m in RLModel]}')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--onnx', action='store_true')
    args = parser.parse_args()

    main(model=args.model,
         should_train=args.train,
         should_render_logs=args.render,
         should_use_onnx=args.onnx)
