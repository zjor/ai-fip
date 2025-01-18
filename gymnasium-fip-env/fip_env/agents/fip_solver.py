import gymnasium as gym
import torch
from stable_baselines3 import PPO

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env_id = 'fip_env/FlywheelInvertedPendulum-v0'


def train():
    env = gym.make(env_id, render_mode=None)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50000)
    torch.save(model.state_dict(), 'fip.pth')


def play():
    env = gym.make(env_id, render_mode="human")
    model = PPO('MlpPolicy', env, verbose=1)
    model.load_state_dict(torch.load('fip.pth'))
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        # print(action)
        state, _, done, _, _ = env.step(action)


def main():
    env = gym.make(env_id, render_mode=None)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50000)

    env = gym.make(env_id, render_mode="human")
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _, _ = env.step(action)


if __name__ == "__main__":
    main()
