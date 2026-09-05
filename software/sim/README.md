# Simulation and classical control

Python model of the flywheel inverted pendulum: RK4 integration of the equations of motion (see [docs/physics](../../docs/physics/equations-of-motion.md)) with LQR, PID and neural-network control laws for comparison.

## Run

```shell
poetry install
poetry run app
poetry run phase0
poetry run phase0-swingup --sweep [--torque-limit 0.5]
poetry run phase0-control --pessimistic --sweep
poetry run python -m unittest discover -s tests
```

`phase0` runs the analytical hardware feasibility pre-check from
[`phase0.toml`](phase0.toml) using the moteus firmware conventions (Kt = 8.27/Kv,
voltage-circle torque-speed envelope). `phase0-swingup` simulates energy-pumping
swing-up plus LQR catch with that envelope and reports the required flywheel
momentum. `phase0-control` sweeps sample rate, delay and noise for the sampled LQR.
Mechanical low/nominal/high values are planning ranges, not measurements; the
report stays `NOT READY` until a CAD mass model exists. Results and decisions:
[docs/physics/phase-0-feasibility.md](../../docs/physics/phase-0-feasibility.md).

## Project tracking

Simulation work is tracked centrally in
[project/tasks.md](../../project/tasks.md), currently T-001 and T-005–T-012.
The associated curriculum and mastery criteria are in
[docs/drl/README.md](../../docs/drl/README.md).
