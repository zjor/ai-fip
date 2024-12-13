import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
from torch import optim
from torch.distributions import Categorical

from fip_env.rl import reinforce

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env_id = 'CartPole-v1'
h_size = 16


class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def train():
    env = gym.make(env_id, render_mode=None)
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    params = {
        "n_training_episodes": 1000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 1.0,
        "lr": 1e-2,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    policy = Policy(params["state_space"], params["action_space"], h_size).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=params["lr"])
    reinforce(env, policy, optimizer, params['n_training_episodes'], params['max_t'], params['gamma'], 100)
    torch.save(policy.state_dict(), 'cartpole.pth')


def play():
    env = gym.make(env_id, render_mode="human")
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    policy = Policy(s_size, a_size, h_size).to(device)
    policy.load_state_dict(torch.load('cartpole.pth'))
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = policy.act(state)
        print(action)
        state, _, done, _, _ = env.step(action)


if __name__ == "__main__":
    train()
    play()
