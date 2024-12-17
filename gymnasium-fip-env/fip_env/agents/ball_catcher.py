import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import numpy as np
import gymnasium
from gymnasium.wrappers import FlattenObservation

from fip_env.envs import BallCatcherEnv
from fip_env.envs.ball_catcher import Actions
from fip_env.rl import reinforce

params = {
    "h_size": 16,
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 20,
    "max_t": 2000,
    "gamma": 1.0,
    "lr": 1e-2,
    "state_space": 5,
    "action_space": 3,
}


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def train(device: torch.device, output_filename: str):
    env = gymnasium.make('fip_env/BallCatcher-v0', render_mode=None)
    env = FlattenObservation(env)
    reset_options = {
        'y0': env.size / 2,
        'vx': 30,
        'vy': 25
    }

    policy = Policy(
        params["state_space"],
        params["action_space"],
        params["h_size"],
    ).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=params["lr"])

    scores = reinforce(
        env,
        policy,
        optimizer,
        params["n_training_episodes"],
        params["max_t"],
        params["gamma"],
        100,
        env_reset_options=reset_options,
    )

    print("Training finished")
    torch.save(policy.state_dict(), output_filename)


def load_and_play(device: torch.device, policy_filename: str):
    policy = Policy(
        params["state_space"],
        params["action_space"],
        params["h_size"],
    ).to(device)
    policy.load_state_dict(torch.load(policy_filename))
    policy.eval()

    env = gymnasium.make('fip_env/BallCatcher-v0', render_mode="human")
    env = FlattenObservation(env)
    reset_options = {
        'y0': env.size / 2,
        'vx': 30,
        'vy': 25
    }

    state, _ = env.reset(options=reset_options)
    done = False
    while not done:
        action, _ = policy.act(np.array(state))
        state, reward, done, _, _ = env.step(action)


def main(training: bool = True):
    policy_filename = "policy.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if training:
        train(device, policy_filename)
    else:
        load_and_play(device, policy_filename)


def sandbox():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env: BallCatcherEnv = gymnasium.make('fip_env/BallCatcher-v0', render_mode="human")
    env.reset(options={
        'y0': env.size / 2,
        'vx': 30,
        'vy': 25
    })
    for i in range(400):
        obs, _, done, _, _ = env.step(Actions.up.value)
        if done:
            break


if __name__ == '__main__':
    # sandbox()
    main(training=False)
