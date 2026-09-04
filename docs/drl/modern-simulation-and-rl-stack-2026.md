# Modern simulation and reinforcement-learning stack (2026)

Research date: 2026-09-04.

## Question

The original project uses a custom NumPy/RK4 model, a Gymnasium environment,
Stable-Baselines3 PPO/SAC, PyTorch, Pygame, and ONNX. This note evaluates a
modern replacement or extension with four goals:

- model the real reaction-wheel pendulum honestly, including the selected
  mj5208 motor and moteus r4.11 controller;
- build a useful 3D digital twin;
- train one policy from many parallel simulated attempts;
- deploy a small policy on the real device through ONNX or an equivalently
  simple runtime.

Training will be done on Apple Silicon. That constraint changes the practical
choice of acceleration backend and rules out relying on CUDA.

## Conclusion

Use a two-level simulation strategy:

1. Keep the repository's analytical/RK4 implementation as the transparent
   reference model and feasibility tool.
2. Build the honest 3D model in MuJoCo and validate it against the analytical
   model before using it for control or reinforcement learning.

The reliable Apple Silicon training baseline should be CPU MuJoCo wrapped as a
Gymnasium vector environment, initially with Stable-Baselines3 PPO. The plant
has only two mechanical degrees of freedom, so benchmark this before assuming
GPU acceleration is necessary.

Investigate MJX-JAX as a second track for JIT-compiled, batched training. MJX
documents Apple Silicon support, but JAX classifies the Apple Metal plug-in as
experimental and notes missing operations and data types. Therefore:

- MJX-JAX on CPU is a supported baseline;
- `jax-metal` is an experiment, not a project dependency or Phase 2 gate;
- every Metal result must be compared with CPU MuJoCo trajectories;
- pin the complete JAX, `jaxlib`, `jax-metal`, MuJoCo, and Playground version
  set after finding a working combination.

MuJoCo Playground is the preferred environment design reference and a candidate
training layer, but its current setup and default Warp backend are primarily
CUDA-oriented. Do not assume its published CUDA commands work unchanged on a
Mac.

## Why the current RL environment should not be extended directly

The existing environment is still a useful learning artifact, but it is not the
honest simulation required by Phase 2:

- rod length is 2 m and wheel radius is 0.75 m;
- the motor is an ideal clipped torque source;
- the integration/control rate is 24 Hz;
- the only random disturbance is an instantaneous angular-velocity kick;
- termination conditions are disabled;
- there is no motor speed/torque envelope, current limit, battery sag, cogging,
  thermal behavior, sensor model, estimator, quantization, or command delay.

There is also a training inconsistency: the reward is always non-positive, but
training is configured to stop only when mean reward reaches positive 500.

Updating library versions would not fix these modeling problems. Model fidelity
and validation must precede new training work, as required by the roadmap.

## Simulator options

### MuJoCo, MJX, and MuJoCo Playground

MuJoCo is the recommended primary simulator. It supports MJCF and URDF models,
rigid-body dynamics, native 3D visualization, interactive external forces,
custom forces and actuators, and simulated joint, IMU, force, and other sensor
data. Visual meshes can come from CAD while simple collision geometry and
measured inertias define the dynamics.

MJX supplies accelerator-oriented implementations of MuJoCo. Batched
`mjx.Model` values naturally express domain randomization, and batched
`mjx.Data` values represent many independent worlds. MJX-JAX is differentiable
and portable through JAX; MJX-Warp is the faster and more complete NVIDIA path,
but is not an Apple Silicon option.

MuJoCo Playground supplies reusable robot-learning environment structure,
parallel PPO training examples, domain-randomization patterns, checkpoint
visualization through `rscope`, and batch rendering. Its classic-control
cart-pole environment is a useful starting example for AI-FIP.

Relevant sources:

- [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html)
- [MuJoCo modeling guide](https://mujoco.readthedocs.io/en/latest/modeling.html)
- [MJX documentation and feature-parity table](https://mujoco.readthedocs.io/en/stable/mjx.html)
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- [MuJoCo Playground technical report](https://playground.mujoco.org/assets/playground_technical_report.pdf)

#### Motor-model relevance

Recent MuJoCo releases include a detailed DC-motor actuator model. Depending on
configuration it can represent:

- back-EMF and the torque-speed envelope;
- voltage, current, torque, and current-slew limits;
- winding inductance and current as a state;
- rotor inertia;
- Coulomb, viscous, nonlinear, and LuGre friction;
- cogging amplitude, periodicity, and phase;
- winding temperature and temperature-dependent resistance;
- PID/setpoint control, slew limiting, and anti-windup.

This closely matches the missing items in the honest-simulation gate. It is not
a complete moteus model, however. In particular, there is no direct current
command mode, and the documented thermal state does not model every form of
driver or magnet derating. Model the policy-facing moteus torque/current command
and its inner control loop explicitly, or verify that MuJoCo's torque
feed-forward mode reproduces measured moteus behavior.

Source: [MuJoCo DC motor model](https://mujoco.readthedocs.io/en/3.7.0/_static/dcmotor.pdf).

### Isaac Lab

Isaac Lab is the strongest alternative when photorealistic rendering, synthetic
camera data, ROS 2 integration, or experience with a widely used industrial
robotics stack is itself a goal. It provides GPU-vectorized physics, domain
randomization, actuator models, sensors, tiled rendering, and integrations with
RSL-RL, skrl, RL-Games, and Stable-Baselines3.

It is not the primary choice for this project because full Isaac Sim workflows
require Linux or Windows, an NVIDIA GPU, and a substantially heavier runtime.
The documented baseline is at least 32 GB RAM and 16 GB GPU VRAM. It cannot be
the local Apple Silicon training stack.

Sources:

- [Isaac Lab overview](https://developer.nvidia.com/isaac/lab)
- [Isaac Lab reinforcement-learning frameworks](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html)
- [Isaac Lab installation requirements](https://isaac-sim.github.io/IsaacLab/develop/source/setup/installation/index.html)

### Genesis World

Genesis World is an interesting experimental Apple Silicon alternative. It has
a Python API, a built-in viewer, IMU and camera sensors, differentiable physics,
parallel environments, and compiler targets including Apple Metal. Its viewer
can lay independent environments out on a grid, matching the desired visual of
many attempts progressing together.

It began as an academic project in late 2024 and its documentation acknowledges
that it is still early. Prototype the FIP in it and benchmark usability and
trajectory accuracy, but do not make it the only Phase 2 simulator until it has
been validated against the analytical model and CPU MuJoCo.

Sources:

- [Genesis World overview](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/what_is_genesis.html)
- [Genesis parallel-simulation tutorial](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/parallel_simulation.html)

### Custom JAX dynamics

Porting the current equations to pure JAX would be the shortest path to a fully
batched and differentiable mathematical simulator. It would retain complete
control over motor and sensor equations, but it would not provide the CAD-based
3D twin or an independent check on the equations. It is useful as a research
experiment and fast oracle, not as the sole simulator.

### Webots, Gazebo, Drake, and PyBullet

- Webots is accessible, cross-platform, visually friendly, and can export an
  interactive 3D web scene. It is attractive for teaching and presentation but
  not as strong for high-throughput training.
- Gazebo is valuable when a ROS-integrated system-level simulator is needed,
  but adds complexity without solving this project's motor-model problem.
- Drake is valuable for control theory, trajectory optimization, and independent
  mathematical verification, but is not the best primary visual parallel-RL
  environment.
- PyBullet offers no clear advantage over current MuJoCo for a new implementation.

Webots sources: [overview](https://cyberbotics.com/doc/guide/introduction-to-webots),
[web scene export](https://cyberbotics.com/doc/guide/web-scene).

## Apple Silicon execution strategy

Apple Silicon has three distinct paths; they should not be conflated.

### Reliable path: CPU MuJoCo

- Run native arm64 MuJoCo physics on CPU.
- Expose the task through Gymnasium.
- Start with synchronous vectorization, then benchmark subprocess workers.
- Use CPU policy training first; a tiny MLP may not benefit from GPU transfer.
- Try PyTorch MPS only after profiling shows optimizer work is material.
- Render only evaluation environments, never every training step.

For a two-DOF mechanism, CPU simulation may already generate experience faster
than the learner consumes it. Measure steps/second at 1, 8, 32, 128, and 512
environments before choosing a more complex backend.

### Research path: MJX-JAX

- Begin with the normal JAX CPU package on macOS.
- JIT the entire environment step and PPO rollout.
- Batch environments with `vmap`/array batch dimensions.
- Verify numerical agreement with CPU MuJoCo over open-loop and controlled
  trajectories.
- Then test Apple's `jax-metal` plug-in on an isolated branch/environment.

JAX's own installation documentation lists macOS arm64 CPU support as supported,
but Apple GPU support as experimental. Apple's plug-in documentation also says
that it does not pass all JAX tests and lacks `float64`, `complex64`, and
`complex128`. This project should use `float32` for policy training but retain
high-precision CPU validation where useful.

Sources:

- [JAX installation and platform support](https://docs.jax.dev/en/latest/installation.html)
- [Apple: accelerated JAX on Mac](https://developer.apple.com/metal/jax/)

### Experimental alternative: Genesis Metal

Implement the same minimal mechanism in Genesis only after the MuJoCo model is
validated. Compare setup effort, simulation throughput, batch reset behavior,
rendering, and trajectory error. This makes Genesis a valuable research subject
without putting the project's feasibility gate at risk.

## Parallel training and visualization

The common arrangement is one shared policy controlling many independent
environment instances, not many unrelated agents:

```text
                    randomized parameters
                             |
one shared policy -> 1,024 independent pendulums -> one policy update
```

Physics should normally run headless. A useful visualization and reporting
scheme is:

- save 8-32 representative evaluation trajectories at regular checkpoints;
- show those trajectories in a tiled 3D view;
- keep a fixed evaluation seed panel so progress is visually comparable;
- add a separate stress-test panel sampled from held-out parameter ranges;
- record videos only at selected checkpoints;
- use TensorBoard locally and optionally Weights & Biases for experiment
  comparison;
- plot median and 10th/90th percentiles, not just mean reward.

Track physical metrics in addition to reward:

- swing-up success and time;
- upright time and angle RMS;
- maximum recoverable push/initial angle;
- wheel-speed peak and RMS;
- torque/current saturation time;
- peak and RMS current;
- estimated winding temperature;
- energy per episode;
- success rate across randomized and worst-case model corners.

## Policy and network design

### Do not begin with a large architecture

The physical state and action are tiny. A suitable first actor is:

```text
normalized deployable observation
    -> Linear(64), Tanh
    -> Linear(64), Tanh
    -> Linear(1), Tanh
    -> scaled torque/current command
```

This is only a few thousand parameters and is easy to run deterministically at
high frequency. A Transformer, CNN, or large learned world model is not justified
for low-dimensional sensor input and known dynamics.

The actor should see only values available on the device, for example:

```text
sin(theta), cos(theta), estimated theta velocity, wheel velocity,
previous command, bus voltage
```

Temperature and rotor/electrical phase may be added only if the real controller
provides them and tests show they matter. Normalize every input from physical
bounds. The action should be dimensionless in `[-1, 1]` and mapped through the
real actuator constraints, rather than pretending torque is unlimited.

### Memory is more valuable than width

Sensor filtering, delay, battery sag, temperature, and uncertain friction make
the deployed problem partially observable. Test architectures in this order:

1. memoryless 2x64 MLP baseline;
2. the same MLP with 4-8 frames of observation/action history;
3. a 32-64-unit GRU only if frame stacking is insufficient.

Frame stacking is easier to train, export, inspect, and deploy. A recurrent
policy requires explicit hidden-state initialization and reset semantics on the
device. Theory and sim-to-real results nevertheless support memory when a policy
must infer randomized dynamics from recent transitions.

Sources:

- [Understanding domain randomization](https://arxiv.org/abs/2110.03239)
- [SB3 Recurrent PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
- [Rapid Motor Adaptation](https://www.roboticsproceedings.org/rss17/p011.pdf)

### Asymmetric actor-critic

Use simulator-only privileged information in the critic, not in the actor:

```text
actor:  deployable sensor history -> 64 -> 64 -> action
critic: true state + current + temperature + randomized parameters
        -> 128 -> 128 -> value
```

This can make training easier without creating an impossible deployment
dependency. The actor still learns solely from information the real device can
provide.

Source: [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542).

## Algorithms to compare

### Primary baseline: PPO

PPO remains a good robotics baseline. It is robust, simple, and well suited to
large batches of parallel environments. Modern robot-learning stacks still use
it as their default; replacing it solely because it is not new would not improve
the project.

On Apple Silicon, begin with Stable-Baselines3 PPO for the reliable CPU MuJoCo
path. For an end-to-end JAX/MJX experiment, use the PPO implementation exercised
by MuJoCo Playground/Brax. RSL-RL is attractive for NVIDIA GPU robotics and now
exports ONNX directly, but it is not the default local Mac path unless its MPS
behavior is explicitly validated.

### Secondary comparisons

- SAC: an established sample-efficient continuous-control baseline.
- CrossQ: a modern SAC-derived method designed for improved sample efficiency
  at a low update-to-data ratio.
- TQC: another useful off-policy continuous-control baseline available through
  SB3 Contrib.

Current SB3 guidance recommends SAC, TD3, CrossQ, or TQC for single-process
continuous control and PPO-family methods when parallel wall-clock throughput is
the priority.

Sources:

- [Stable-Baselines3 algorithm guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [CrossQ, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f381114cf5aba4e45552869863deaaa7-Abstract-Conference.html)

### Why not Dreamer or a large world model

DreamerV3 is a significant general-purpose result, but it learns a recurrent
world model in addition to actor and critic networks. AI-FIP already has known,
low-dimensional dynamics and can generate simulated experience cheaply. Dreamer
would be an interesting later study, not a justified first implementation.

Source: [DreamerV3](https://doi.org/10.1038/s41586-025-08744-2).

## Controller experiments

Evaluate at least three controllers against exactly the same deterministic and
randomized test suites:

1. energy shaping plus LQR;
2. pure PPO;
3. residual PPO, `u = u_classical + delta_u_nn`.

The hybrid is likely the safest first sim-to-real candidate: the classical
controller supplies understood swing-up and local stabilization, while the
network learns imperfect catch timing, delay, friction, and other residual
effects. The pure neural policy remains the important comparison for the
project's research goal.

Prefer one policy trained by curriculum over two independently deployed neural
policies:

1. balance from a narrow upright distribution;
2. widen initial angle and angular velocity;
3. introduce pushes;
4. extend to full swing-up;
5. progressively widen physically justified domain randomization.

Also retain the explicit energy-shaping/LQR state machine as a classical
baseline. A learned high-level selector can be investigated later if one global
policy cannot learn both behaviors reliably.

## Sim-to-real procedure

Fidelity does not come from 3D graphics. It comes from identified dynamics,
correct interfaces, and validation. Use this sequence:

1. Derive mass, center of mass, and inertia from the component-level CAD/BOM.
2. Measure pivot free decay and motor coast-down to estimate friction.
3. Measure commanded versus delivered torque/current and step response.
4. Measure battery voltage under load and controller timing/jitter.
5. Measure IMU/encoder noise, bias, quantization, estimator delay, and rate.
6. Fit the nominal simulator from these experiments.
7. Define randomization distributions from measurement uncertainty and observed
   variation, not arbitrary wide guesses.
8. Reserve held-out parameter corners and disturbance sequences for evaluation.
9. Pass energy-shaping and LQR gates before training a neural controller.
10. Progress through software-in-the-loop, recorded-data replay,
    hardware-in-the-loop, restrained low-torque tests, and only then free runs.

Randomize at least masses/CoM/inertias, motor constant and resistance, voltage
sag, current-loop response, torque-speed envelope, friction, cogging, sensor
noise/bias/quantization, estimator filtering, command delay/jitter, and external
disturbances. Domain randomization should surround a measured nominal model, not
replace system identification.

Sources:

- [Dynamics randomization for sim-to-real](https://openai.com/index/sim-to-real-transfer-of-robotic-control-with-dynamics-randomization/)
- [MuJoCo Playground domain-randomization approach](https://playground.mujoco.org/assets/playground_technical_report.pdf)

## Proposed experimental sequence

### Stage A: authoritative dynamics

- Create a minimal MJCF model using primitive bodies.
- Compare free fall, small-angle oscillation, energy, and commanded-torque
  trajectories with the analytical simulator.
- Add the motor and sensor models one feature at a time.
- Re-run the Phase 2 LQR and swing-up gate after each material change.

### Stage B: visualization

- Replace primitives visually with CAD meshes without changing inertial or
  collision definitions.
- Add a native MuJoCo single-environment viewer with push interaction.
- Add fixed-camera recording for reproducible comparisons.

### Stage C: Apple Silicon benchmarks

- Benchmark CPU MuJoCo plus Gymnasium at several environment counts.
- Benchmark MJX-JAX on CPU with the same task and step size.
- Experiment with `jax-metal` in a separate pinned environment.
- Prototype Genesis Metal only after the reference trajectories are established.
- Record compile time, steady-state steps/second, memory, numerical error, and
  training time to a fixed success threshold.

### Stage D: learning baselines

- Train PPO with the 2x64 memoryless actor.
- Add frame stacking, then an asymmetric critic.
- Compare SAC and CrossQ/TQC.
- Compare pure, classical, and residual policies over several fixed seeds.
- Export only actors that pass the full held-out stress suite.

## Decision boundary

The present decision is to use Apple Silicon for training and to investigate
MuJoCo as the primary honest simulator. The exact accelerated training backend
is deliberately not decided until the Stage C benchmark. A successful research
result may be that simple CPU MuJoCo is faster, more reproducible, and easier to
validate than either experimental Metal path for this small mechanism.
