# DRL and simulation learning track

This is the curriculum for mastering deep reinforcement learning and 3D
simulation through AI-FIP. Project progress and current learning actions are
tracked only in [project/tasks.md](../../project/tasks.md); the project phases
and gates live in [project/roadmap.md](../../project/roadmap.md).

Principle: learn each concept by building, explaining, and measuring it in the
project. Do not start new policy training until the honest-simulation gate is
passed with a classical controller.

## Curriculum

- **MuJoCo foundations:** understand MJCF body trees, frames, hinge joints,
  inertial parameters, actuators, sensors, timestep, integration and the Python
  stepping/viewer API.
- **Validated 3D digital twin:** construct the two-DOF pendulum and reaction
  wheel from primitive dynamic geometry; reproduce the analytical RK4 model in
  passive, constant-torque and closed-loop trajectories; add CAD meshes only as
  visual geometry.
- **Honest classical-control baseline:** model the measured motor envelope,
  current limits, friction/cogging, sensor noise, sampling and delay; demonstrate
  swing-up, LQR catch, stabilization and wheel despinning.
- **RL foundations:** be able to explain and calculate MDP transitions,
  returns, value and action-value functions, Bellman equations, policy gradients,
  actor-critic, advantage estimation/GAE and PPO clipping.
- **Gymnasium environment design:** define deployable observations and
  actions, termination versus truncation, reward components, normalization,
  deterministic seeding and vectorized evaluation around the validated model.
- **PPO baseline:** train a small 2x64 policy and compare it with LQR using
  physical metrics: success rate, angle RMS, wheel speed, saturation, current,
  energy and recovery envelope—not reward alone.
- **Robustness and sim-to-real:** train and test across measured parameter
  ranges, use held-out worst-case evaluation, compare pure PPO with residual PPO,
  export the actor and measure end-to-end latency on the target host.

## First practical exercise

1. Build a minimal MJCF model with one pendulum hinge and one wheel hinge using
   boxes and cylinders.
2. Step it from Python, apply wheel torque, render it, and log `qpos`, `qvel`
   and actuator torque.
3. Compare MuJoCo with the existing RK4 simulator for free fall, a fixed torque
   pulse and the same LQR controller.

The exercise is complete when the model structure and every physical parameter
can be explained, the comparison is reproducible, and discrepancies have explicit
error bounds or documented causes.

## Resources

- [MuJoCo modeling guide](https://mujoco.readthedocs.io/en/latest/modeling.html)
- [MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)
- [PPO explanation and equations](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Modern simulation and RL stack for this project](modern-simulation-and-rl-stack-2026.md)
- [DRL outline and glossary](outline.md)
