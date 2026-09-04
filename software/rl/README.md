# Reinforcement learning

Gymnasium environment for the flywheel inverted pendulum, training with Stable-Baselines3, export of the trained policy to ONNX.

```
fip_env/    Gymnasium environment (fip_env/FlywheelInvertedPendulum-v0) and shared physics/graphics
train/      training and evaluation scripts (SB3 PPO/SAC)
export/     PyTorch → ONNX export of the actor network
models/     trained artifacts; *.onnx is tracked and consumed by software/web and the device
logs/       SB3 evaluation logs (ignored)
```

## Setup

```shell
poetry install
pre-commit install
```

## Run

```shell
poetry run python -m train.fip_solver --train --model=PPO   # train, saves models/fip_solver.pth
poetry run python -m train.fip_solver                        # play with the SB3 model
poetry run python -m train.fip_solver --onnx                 # play with the ONNX model
poetry run python -m train.fip_solver --render               # plot evaluation logs
poetry run python -m export.export_model_to_onnx             # models/fip_solver.pth → models/fip_solver.onnx
```

## Status / Next

- introduce episode termination condition by angle and wheel rotation speed
- limit number of steps when rendering
- two-phase training: swing-up, then stabilize and stop the wheel (different reward per phase)
