# Project tasks

Single source of truth for all unfinished AI-FIP work.

Current objective: finish the simulation foundations, turn the delivered
hardware into measured Phase 0/1 inputs, and pass the Phase 2 honest-simulation
gate. Task order is global across hardware, software and learning.

## Now

- [ ] **T-005.A [HW, SIM] Bound response timing on the current desk stand.**
  Stopped inspection, a physical bidirectional position-step run, and centered
  friction/cogging runs before and after anticogging are complete; results and
  run IDs are in [log.md](log.md). Analyze the recorded position-step timestamps,
  tracking and repeatability to bound command-to-motion delay and tail latency;
  repeat the same Tier-A run only if the existing data cannot support the bound.
  **Setup:** current desk stand and light arrow.
  **Done when:** timing and tracking bounds, uncertainty and run provenance are
  documented for the simulator.
- [ ] **T-005.C [HW, SIM] Measure the static torque constant.**
  Build and check the [specified ±100 mm lever and scale fixture](../hardware/motor-torque-fixture-spec.md),
  then implement and run guarded short bidirectional 0–8 A current plateaus.
  Fit physical torque against measured Q-axis current; moteus-reported torque is
  a comparison channel, not the reference.
  **Setup:** fixed motor, balanced lever, compression scale and hard stop.
  **Done when:** independent $K_t$ estimates in both directions, fit uncertainty
  and run provenance meet the fixture spec's acceptance criteria.
- [ ] **T-002 [HW, CAD] Verify the mj5208 stator mounting pattern.**
  Read the official 2D drawing and replace the assumed square `stator_pitch`
  geometry in `hardware/cad/params.scad` if necessary.
  **Done when:** the source and dimensions are recorded and `motor_flange` is
  ready for its first print.

## Next

- [ ] **T-005.B [HW, SIM] Measure rotating losses and acceleration.**
  Build a balanced, retained flywheel assembly with known inertia and a guard;
  no clock arrow. Run guarded coast-down and acceleration tests to estimate
  speed-dependent losses and check torque-to-acceleration behavior.
  **Done when:** loss-versus-speed and acceleration estimates have uncertainty,
  assembly inertia and run provenance.
  **Depends on:** a qualified retained rotating assembly and guard.
- [ ] **T-005.D [HW, SIM] Measure the torque-speed and voltage envelope.**
  Use a guarded rotating assembly with known inertia or external load to measure
  reachable torque and current versus speed across the 12.0–16.8 V design range.
  An unloaded speed sweep alone is not a torque-at-speed measurement.
  **Done when:** effective current/voltage limits and a torque-speed envelope
  with uncertainty and run provenance are usable by the simulator.
  **Depends on:** T-005.C and a qualified rotating load or known-inertia setup
  with a guarded supply. T-005.B can provide the known-inertia assembly.
- [ ] **T-005.E [HW, SIM] Measure the continuous thermal limit.**
  Add a winding thermistor or external temperature probe and a torque-rated
  restraint; run supervised thermal plateaus with temperature, rise-rate, board,
  voltage and time stop conditions. The initial ±8 A lever fixture is not
  qualified for this test.
  **Done when:** continuous torque/current versus temperature has explicit
  conditions, uncertainty and run provenance.
  **Depends on:** T-005.C and a qualified thermal restraint/instrument.
- [ ] **T-005.F [SIM, HW] Publish measured motor inputs.**
  Fit and review the bench results, then transfer parameter ranges, uncertainty
  and run IDs into `docs/physics/phase-0-feasibility.md` and
  `software/sim/phase0.toml`; keep raw runs uncommitted.
  **Done when:** all A–E results are transferred with measured values, declared
  uncertainty and provenance; any unmeasurable input is explicitly identified.
  **Depends on:** T-005.A–E.
- [ ] **T-003 [HW, SIM, CAD] Decide the battery offset below the pivot.**
  Evaluate the current centred placement and practical negative-Z offsets in the
  mass model, including gravity coefficient, pendulum inertia and clearances.
  **Done when:** one placement is selected and propagated to the CAD specification
  and Phase 0 parameters.
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
  **Depends on:** T-002, T-003, T-006, T-007 and the physical parts needed for
  fit checks.
- [ ] **T-009 [SIM] Make the MuJoCo model honest.**
  Add the measured motor envelope and current limits, flywheel contribution to
  pendulum inertia, friction/cogging, sensor sampling/noise/quantization, estimator
  behavior, command delay and battery voltage range.
  **Depends on:** T-005.F, T-006, T-007 and T-008.
- [ ] **T-010 [CONTROL, SIM] Stabilize and despin with LQR.**
  Tune and verify sampled LQR in the honest model, including torque saturation
  and wheel-speed cost.
  **Done when:** it holds the required recovery envelope and brings wheel speed
  back toward zero across the declared parameter range.
  **Depends on:** T-009.
- [ ] **T-011 [CONTROL, SIM] Demonstrate swing-up and catch.**
  Validate energy pumping, LQR handoff and the wheel-speed-at-catch constraint
  with real actuator limits.
  **Depends on:** T-009 and T-010.
- [ ] **T-012 [PHYSICS, SIM] Close the feasibility and honest-simulation gates.**
  Publish the final feasible parameter window and run reproducible worst-case
  checks for balance, disturbance recovery, swing-up and sensor timing.
  **Depends on:** T-005.F, T-009, T-010 and T-011.

## Later

- [ ] **T-013 [BUILD] Manufacture and assemble the final mechanics and electronics.**
  **Depends on:** T-012.
- [ ] **T-014 [CONTROL, HW] Run LQR stabilization and swing-up on real hardware.**
  Compare recorded telemetry with the honest simulator and bound the remaining
  model error. **Depends on:** T-013.
- [ ] **T-015 [LEARN, RL] Master the RL foundations used here.**
  Explain and calculate MDPs, returns, value functions, Bellman equations, policy
  gradients, actor-critic, GAE and PPO clipping; see `docs/drl/README.md`.
- [ ] **T-016 [RL] Build the validated Gymnasium environment.**
  Define deployable observations/actions, termination versus truncation, reward
  components, normalization, deterministic seeding and vectorized evaluation.
  **Depends on:** T-014.
- [ ] **T-017 [RL] Train and evaluate a small PPO baseline.**
  Start with a 2×64 actor; bound rendering steps and compare against LQR using
  physical success, angle, wheel-speed, saturation, current and energy metrics.
  **Depends on:** T-016.
- [ ] **T-018 [RL] Evaluate robustness and sim-to-real.**
  Use measured domain randomization and held-out corners; compare pure PPO with
  residual PPO; export the actor and measure target-host latency.
  **Depends on:** T-017.
- [ ] **T-019 [CONTENT] Produce articles, demonstrations and project media.**
- [ ] **T-020 [WEB] Improve the browser demonstration.**
  Add useful state/control graphs and make the interface mobile-friendly after
  it consumes a policy produced by the validated RL pipeline.
