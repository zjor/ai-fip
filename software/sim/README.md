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

## Status / Next

- Phase 0 pre-check: `phase0.toml` `[design]` now comes from `hardware/cad` (`make masses`);
  replace the RPi/pi3hat and axle hardware estimates with weighed parts after printing
- Phase 0 pre-check: confirm `max_voltage_ratio` (moteus `min_pwm` depends on PWM rate)
  and the motor thermal model on the bench; add cogging and friction to the swing-up
- Swing-up sim: the pendulum model treats the wheel as a point mass in I_p; add the
  wheel's own inertia to the pendulum equation and add sensor noise to the pumping phase
- LQR stabilizes the upright position but the wheel does not stop; add wheel-speed weight to the cost function
- Roadmap Phase 2 (honest simulation): add motor torque/speed curve, current limit, sensor noise, loop delay, friction
