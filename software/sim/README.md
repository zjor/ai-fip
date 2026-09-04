# Simulation and classical control

Python model of the flywheel inverted pendulum: RK4 integration of the equations of motion (see [docs/physics](../../docs/physics/equations-of-motion.md)) with LQR, PID and neural-network control laws for comparison.

## Run

```shell
poetry install
poetry run app
poetry run phase0
poetry run phase0-control --pessimistic --sweep
```

`phase0` runs the analytical hardware feasibility pre-check from
[`phase0.toml`](phase0.toml). Its low/nominal/high mechanical values are planning
ranges, not measurements. The report intentionally remains `NOT READY` while
required physical inputs are unknown.

## Status / Next

- Phase 0 pre-check: replace the planning ranges in `phase0.toml` with a
  component-level CAD/BOM mass model; establish motor thermal torque and cogging
- Phase 0 pre-check: obtain required swing-up momentum and sensor noise/latency
  limits from the constrained model
- LQR stabilizes the upright position but the wheel does not stop; add wheel-speed weight to the cost function
- Roadmap Phase 2 (honest simulation): add motor torque/speed curve, current limit, sensor noise, loop delay, friction
