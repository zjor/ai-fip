# Gymnasium Examples
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Environments
This repository hosts the examples that are shown [on the environment creation documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- `GridWorldEnv`: Simplistic implementation of gridworld environment

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

### Contributing
If you would like to contribute, follow these steps:
- Fork this repository
- Clone your fork
- Set up pre-commit via `pre-commit install`

PRs may require accompanying PRs in [the documentation repo](https://github.com/Farama-Foundation/Gymnasium/tree/main/docs).


## Installation

To install your new environment, run the following commands:

```{shell}
cd fip_env
pip install -e .
```

## Development dependencies

- `pip install imageio`

## How to run locally

```{shell}
python -m fip_env.sandbox
```

## TODO
- [x] limit motion of the bat if goes beyond the boundaries
- [x] make the env smaller, reward is too delayed
- [x] make action steps bigger (10, 25, 50 pixels)
- [x] reduce FPS
- [x] try reward function without negative score for moves
- plot graph of the learning curve, avg score over the number of episodes
- try to add more hidden layers
- run on collab with GPU | Kaggle
- try huge number of episodes
- normalize input data between [0, 1]; do we even need it?
