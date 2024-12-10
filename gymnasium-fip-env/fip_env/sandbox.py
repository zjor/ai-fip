import torch
import torch.optim as optim
import numpy as np
import gymnasium
from gymnasium.wrappers import FlattenObservation

from fip_env.rl import Policy, reinforce

if __name__ == '__main__':
    env = gymnasium.make('fip_env/BallCatcher-v0', render_mode=None)
    env = FlattenObservation(env)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = {
        "h_size": 16,
        "n_training_episodes": 5000,
        "n_evaluation_episodes": 10,
        "max_t": 1000,
        "gamma": 1.0,
        "lr": 1e-2,
        "state_space": 5,
        "action_space": 3,
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
    )

    print("Training finished")

    # for i in range(400):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, _, info = env.step(action)
    #     if terminated:
    #         break
    # pixels = env.render()
    # iio.imwrite(f"output-{i}.png", pixels)
