# Simulation and classical control

Python model of the flywheel inverted pendulum: RK4 integration of the equations of motion (see [docs/physics](../../docs/physics/equations-of-motion.md)) with LQR, PID and neural-network control laws for comparison.

## Run

```shell
poetry install
poetry run app
```

## Status / Next

- LQR stabilizes the upright position but the wheel does not stop; add wheel-speed weight to the cost function
- Roadmap Phase 2 (honest simulation): add motor torque/speed curve, current limit, sensor noise, loop delay, friction
