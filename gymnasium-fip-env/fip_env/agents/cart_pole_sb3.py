"""
CartPole-v1 source code:
    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py

How to run
    python -m fip_env.agents.cart_pole_sb3 --train
"""
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces, logger

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "CartPole-v1"


class CenteredCartPoleEnv(gym.Env):
    def __init__(self,
                 action_threshold=10.0,
                 kick_probability=0.0,
                 kick_strength=1.0,
                 max_steps=500,
                 theta_threshold_radians=None, **kwargs):
        self.env = gym.make(ENV_ID, **kwargs)  # Original CartPole environment
        self._super_env = self.env.unwrapped
        self.theta_threshold_radians = self._super_env.theta_threshold_radians if theta_threshold_radians is None else theta_threshold_radians

        # self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Box(low=-action_threshold, high=action_threshold, shape=(1,), dtype=np.float32)
        self.max_steps = max_steps
        self.current_step = 0

        self.kick_probability = kick_probability
        self.kick_strength = kick_strength

    def reset(self, **kwargs):
        self.current_step = 0
        return self.env.reset(**kwargs)

    def _super_step(self, action):
        _super_env = self.env.unwrapped
        x, x_dot, theta, theta_dot = _super_env.state
        # force = _super_env.force_mag if action == 1 else -_super_env.force_mag
        force = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])  # apply continuous force
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + _super_env.polemass_length * np.square(theta_dot) * sintheta
               ) / _super_env.total_mass
        thetaacc = (_super_env.gravity * sintheta - costheta * temp) / (
                _super_env.length
                * (4.0 / 3.0 - _super_env.masspole * np.square(costheta) / _super_env.total_mass)
        )
        xacc = temp - _super_env.polemass_length * thetaacc * costheta / _super_env.total_mass

        if _super_env.kinematics_integrator == "euler":
            x = x + _super_env.tau * x_dot
            x_dot = x_dot + _super_env.tau * xacc
            theta = theta + _super_env.tau * theta_dot
            theta_dot = theta_dot + _super_env.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + _super_env.tau * xacc
            x = x + _super_env.tau * x_dot
            theta_dot = theta_dot + _super_env.tau * thetaacc
            theta = theta + _super_env.tau * theta_dot

        _super_env.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -_super_env.x_threshold
            or x > _super_env.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if _super_env._sutton_barto_reward else 1.0
        elif _super_env.steps_beyond_terminated is None:
            # Pole just fell!
            _super_env.steps_beyond_terminated = 0

            reward = -1.0 if _super_env._sutton_barto_reward else 1.0
        else:
            if _super_env.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            _super_env.steps_beyond_terminated += 1

            reward = -1.0 if _super_env._sutton_barto_reward else 0.0

        if _super_env.render_mode == "human":
            _super_env.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(_super_env.state, dtype=np.float32), reward, terminated, False, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._super_step(action)

        if np.random.random() < self.kick_probability:
            # apply kick to the pole's angular velocity
            obs[3] += np.random.uniform(-self.kick_strength, self.kick_strength)

        cart_position = obs[0]
        position_penalty = -abs(cart_position)  # Negative reward for being away from center

        # Combine the original reward with the position penalty
        reward += position_penalty

        # Increment step counter
        self.current_step += 1

        # Check if the episode is done
        if self.current_step >= self.max_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()


def _get_env(**kwargs: Any):
    return CenteredCartPoleEnv(**kwargs)


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
    model.learn(total_timesteps=50000, callback=eval_callback)


def main(should_train: bool = True):
    filename = "ppo_cartpole.pth"
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
            ent_coef=0.01,  # Entropy coefficient for exploration
            max_grad_norm=0.5,  # Gradient clipping
        )

        _train(env, model)

        eval_env = _get_env()
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

        model.save(filename)
    else:
        model = PPO.load(filename)
        env = _get_env(render_mode="human", kick_probability=0.5, kick_strength=2.0, max_steps=1000, theta_threshold_radians=24 * 2 * np.pi / 360)
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
    args = parser.parse_args()

    main(should_train=args.train)