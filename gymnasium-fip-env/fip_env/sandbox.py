from time import sleep

import gymnasium
import numpy as np
import torch
import torch.optim as optim
from gymnasium.wrappers import FlattenObservation

from fip_env.envs import FlywheelInvertedPendulumEnv
from fip_env.rl import Policy, reinforce

params = {
    "h_size": 16,
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 20,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 0.05,
    "state_space": 5,
    "action_space": 2,
}


def train(device: torch.device, output_filename: str):
    env = gymnasium.make('fip_env/BallCatcher-v0', render_mode=None)
    env = FlattenObservation(env)

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

    state, _ = env.reset()
    done = False
    while not done:
        action, _ = policy.act(np.array(state))
        state, reward, done, _, _ = env.step(action)


def main():
    policy_filename = "policy.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training = False
    if training:
        train(device, policy_filename)
    else:
        load_and_play(device, policy_filename)


def sandbox():
    env: FlywheelInvertedPendulumEnv = gymnasium.make('fip_env/FlywheelInvertedPendulum-v0', render_mode="human")
    env.reset()
    for i in range(500):
        # apply LQR regulator for tests
        # apply LQR regulator for tests
        action = - (195 * env.theta + 100 * env.theta_dot - 5 * env.phi_dot)
        env.step(np.array([action], dtype=np.float32))



if __name__ == '__main__':
    sandbox()
    # main()
