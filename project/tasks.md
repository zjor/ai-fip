# Tasks

Single source of truth for all unfinished AI-FIP work.

Current objective: close Phase 0 inputs and pass the Phase 2 honest-simulation
gate. Task order is global across hardware, software and learning.

## Now

- [ ] **T-001 [SIM, LEARN] Build and understand the minimal MuJoCo model.**
  Create a primitive-geometry MJCF model with a pendulum hinge and reaction-wheel
  hinge; step, actuate, render and log it from Python. Compare free fall, a fixed
  torque pulse and LQR trajectories with the analytical RK4 model.
  **Done when:** the comparison is reproducible, every dynamic parameter can be
  explained, and discrepancies have explicit bounds or documented causes.
- [ ] **T-002 [HW, CAD] Verify the mj5208 stator mounting pattern.**
  Read the official 2D drawing and replace the assumed square `stator_pitch`
  geometry in `hardware/cad/params.scad` if necessary.
  **Done when:** the source and dimensions are recorded and `motor_flange` is
  ready for its first print.
- [ ] **T-003 [HW, SIM, CAD] Decide the battery offset below the pivot.**
  Evaluate the current centred placement and practical negative-Z offsets in the
  mass model, including gravity coefficient, pendulum inertia and clearances.
  **Done when:** one placement is selected and propagated to the CAD specification
  and Phase 0 parameters.

## Waiting

- [ ] **T-004 [HW] Receive mjbots order MJ5921.**
  Expected delivery: 2026-09-10–11. This blocks measured motor/controller inputs.
  **Done when:** contents are inventoried and any discrepancies are recorded.

## Next

- [ ] **T-005 [HW, SIM] Characterize mj5208 + moteus r4.11 on the bench.**
  Measure/confirm the torque-speed envelope, effective voltage limit, cogging,
  friction, continuous thermal limit and relevant current behavior.
  **Depends on:** T-004. **Done when:** measured ranges and provenance replace
  planning assumptions in the feasibility and simulation inputs.
- [ ] **T-006 [SENSOR, HW] Select the host and pendulum-angle sensor.**
  Check available Raspberry Pi hardware; evaluate RPi 4 + pi3hat versus another
  host/CAN-FD interface, and validate the pi3hat IMU versus an axle encoder for
  swing-up acceleration. Complete the sensor noise/rate/latency budget.
  **Done when:** host and sensor architecture are selected with measured or
  datasheet-backed bounds usable by the simulator.
- [ ] **T-007 [POWER, HW] Select and obtain the battery and charger.**
  Use the recorded requirement: ordinary 4S LiPo, 1000 mAh preferred
  (1000–1300 mAh), at least 40 A continuous / 45C at 1000 mAh, XT30 plus JST-XH
  5-pin, no larger than 75 × 35 × 25 mm, target 100–130 g; obtain a 4S balance
  charger with storage mode.
  **Done when:** exact parts, dimensions and measured masses are recorded.
- [ ] **T-008 [CAD, HW] Finish and validate the printable mechanics.**
  Add power/CAN cable routing, a switch pocket and an optional axle-encoder seat;
  print and weigh the parts; replace RPi/pi3hat and axle-hardware mass estimates;
  select the M6 bolt count to reach flywheel inertia near 0.003 kg·m².
  **Depends on:** T-002, T-003 and the physical parts needed for fit checks.
- [ ] **T-009 [SIM] Make the MuJoCo model honest.**
  Add the measured motor envelope and current limits, flywheel contribution to
  pendulum inertia, friction/cogging, sensor sampling/noise/quantization, estimator
  behavior, command delay and battery voltage range.
  **Depends on:** T-001, T-005 and T-006.
- [ ] **T-010 [CONTROL, SIM] Stabilize and despin with LQR.**
  Tune and verify sampled LQR in the honest model, including torque saturation
  and wheel-speed cost.
  **Done when:** it holds the required recovery envelope and brings wheel speed
  back toward zero across the declared parameter range.
- [ ] **T-011 [CONTROL, SIM] Demonstrate swing-up and catch.**
  Validate energy pumping, LQR handoff and the wheel-speed-at-catch constraint
  with real actuator limits.
  **Depends on:** T-009 and T-010.
- [ ] **T-012 [PHYSICS, SIM] Close the feasibility and honest-simulation gates.**
  Publish the final feasible parameter window and run reproducible worst-case
  checks for balance, disturbance recovery, swing-up and sensor timing.
  **Depends on:** T-005–T-011.

## Later

- [ ] **T-013 [BUILD] Manufacture and assemble the final mechanics and electronics.**
- [ ] **T-014 [CONTROL, HW] Run LQR stabilization and swing-up on real hardware.**
- [ ] **T-015 [LEARN, RL] Master the RL foundations used here.**
  Explain and calculate MDPs, returns, value functions, Bellman equations, policy
  gradients, actor-critic, GAE and PPO clipping; see `docs/drl/README.md`.
- [ ] **T-016 [RL] Build the validated Gymnasium environment.**
  Define deployable observations/actions, termination versus truncation, reward
  components, normalization, deterministic seeding and vectorized evaluation.
- [ ] **T-017 [RL] Train and evaluate a small PPO baseline.**
  Start with a 2×64 actor; bound rendering steps and compare against LQR using
  physical success, angle, wheel-speed, saturation, current and energy metrics.
- [ ] **T-018 [RL] Evaluate robustness and sim-to-real.**
  Use measured domain randomization and held-out corners; compare pure PPO with
  residual PPO; export the actor and measure target-host latency.
- [ ] **T-019 [CONTENT] Produce articles, demonstrations and project media.**
- [ ] **T-020 [WEB] Improve the browser demonstration.**
  Add useful state/control graphs and make the interface mobile-friendly after
  it consumes a policy produced by the validated RL pipeline.
