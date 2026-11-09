# Coast-down bench experiment — design

Status: implemented 2026-10-28 (`bench coast-down`, `bench analyze`; plan
`.claude/plans/2026-10-28-coast-down-bench.md`). Session plumbing now lives in
`experiments/tier_b.py` (2026-11-01), behaviour unchanged. YouTrack: PNDLM-11 (T-005.B). Parent spec:
[motor-bench.md](motor-bench.md) §6 step 5. Inertia:
[test-disc.md](test-disc.md),
[measurements/2026-10-28-test-disc.json](../measurements/2026-10-28-test-disc.json).

## 1. Purpose

`poetry run bench coast-down` records free coast-downs of the test disc from a
ladder of speeds and, with `--accelerate`, torque-step spin-ups from rest.
Outputs after `bench analyze`: loss torque versus speed with uncertainty, a
steady-state current cross-check at each plateau, and an effective inertia from
torque-to-acceleration. All three feed the honest model.

## 2. Bench conditions (as agreed)

- Supply: 24 V / 5 A lab supply. It cannot sink current, so the disc is
  **never braked by the motor**. Every stop is a free coast. At 24 V the
  no-load speed is well above 3000 rpm; spin-up bus current stays under 2 A.
- Disc: Ø150 PLA, I ≈ 0.00037 kg·m² (rendered; replaced by the weighed value in
  the JSON record), mounted with 3× M3 × 8 and threadlocker, guard closed.
- Controller: moteus r4.11, ID 1, fdcanusb, 100 ms watchdog.

## 3. Sequence

Ladder run (default): for each speed in the ladder, in the configured
direction:

1. **Preflight** stopped: fault 0, voltage, temperature, finite position,
   mode 0 confirmed.
2. **Operator gate** per step: the tool prints the step and waits for `GO`.
   Any other input ends the run cleanly after the current coast.
3. **Spin-up**: velocity mode (`position = NaN`, `velocity = target`,
   `maximum_torque = cap`, `accel_limit`), refreshed every sample.
   Timeout if the target is not reached.
4. **Plateau**: hold the velocity for `plateau_s`; record q-current. The mean
   current here times K_t is the direct loss torque at that speed.
5. **Coast**: `set_stop()` (phases open) and query only, until
   `|velocity| < coast_stop_rev_s` continuously for `coast_settle_s`, or
   `coast_timeout_s`.
6. Next step: a `preflight` standstill wait with the same settle rule absorbs a
   late velocity blip before the prompt.

Acceleration run (`--accelerate`): for each torque in `accel_torques_nm`, in
both directions: quiet baseline at rest, feedforward torque step
(`kp_scale = kd_scale = 0`) until `|velocity| ≥ accel_target_rev_s`, then
`set_stop()` and record the coast as above.

Defaults (`bench.toml [coast_down]`, my calls):

| Key | Value | Note |
|---|---|---|
| `sample_rate_hz` | 200 | one query per period, no catch-up bursts |
| `ladder_rpm` | 500, 1000, 1500, 2000, 2500, 3000 | 8.3 … 50 rev/s; `--ladder 20,50,100` overrides for a dry run of the sequence |
| `direction` | +1 | run again with −1 for the other direction |
| `spin_up_torque_nm` | 0.10 | ≈ 4 A phase; 3000 rpm in ≈ 3 s |
| `accel_limit_rev_s2` | 20 | firmware ramp |
| `spin_up_timeout_s` | 15 | |
| `plateau_s` | 3.0 | |
| `coast_stop_rev_s` | 0.2 | below this the coast is over |
| `coast_settle_s` | 0.25 | ... once the speed has stayed below it this long; the encoder velocity jitters ±0.05 rev/s at low speed |
| `coast_timeout_s` | 120 | ≈ 12 s expected |
| `accel_torques_nm` | 0.05, 0.10, 0.15 | within the K_t linear range |
| `accel_target_rev_s` | 30 | 1800 rpm, then coast |
| `max_velocity_rev_s` | 55 | host stop threshold, Tier B |
| `max_torque_nm` | 0.20 | Tier B cap on any command |
| `inertia_record` | `docs/hardware/measurements/2026-10-28-test-disc.json` | `assembly.I_z_kg_m2` (0.000354, disc + screws, weighed) and its provenance copied into `run.json` |
| `kt_nm_per_a` | 0.0244 | PNDLM-9, for the plateau cross-check |

## 4. Safety

- **Tier B limits** are a new `[safety]` pair, `max_rotating_velocity_rev_s`
  and `max_rotating_torque_nm`; the Tier A excursion limit does not apply
  (unbounded rotation). Fault, bus voltage and temperature checks stay as they
  are on every sample.
- **Controller overrides**, volatile only, never `conf write`, restored in
  `finally` with readback: `servo.timeout_mode 0` (watchdog → torque off, so a
  host hang coasts instead of braking); `servopos.position_min/max NaN`
  (unbounded); `servo.max_velocity` = 55 rev/s, which derates torque above
  that speed on the controller side (not a hard limit; the host stop
  threshold is the limit); flux brake threshold 27 V so any regeneration
  above the supply voltage is dissipated in the windings instead of pushed
  into a supply that cannot sink it. The key depends on the firmware:
  `servo.flux_brake_min_voltage 27` on older builds, or
  `servo.flux_brake_margin_voltage` = `servo.max_voltage` − 27 on 2025
  builds (our r4.11: 30 − 27 = 3, which is already the default). No key at
  all stops the run before motion.
- **A safety stop does not stop the disc.** On any exception, a cancelled
  task (what `asyncio.run` does on Ctrl-C) or a lost reply the tool sends
  `set_stop()` (retried), and then keeps querying and printing the speed until
  the disc is below `coast_stop_rev_s`, tolerating lost replies, so the
  operator is never left with a blind spinning disc. A second Ctrl-C during
  that wait is swallowed with a reminder. Power disconnect stays the only
  emergency stop; it also coasts.
- **Watchdog trip is a safety stop, not data.** During spin-up, plateau and
  acceleration the controller must report position mode after the first two
  samples; mode 11 (timeout) ends the run, because the firmware ignores
  further commands and the rows would otherwise look like a plateau.
- Regeneration: velocity commands use `kp_scale 0`, a pure velocity loop, so
  the lag built up during the ramp cannot be corrected by a braking pulse at
  its end; the only motor-driven decelerations are the velocity loop's small
  corrections during the plateau, which the flux brake absorbs. Bus voltage
  is checked every sample against the 30 V limit.
- Ramp discipline from the test-disc spec: the ladder is the ramp. The
  operator stops at any step if vibration grows or the plateau current jumps.

## 5. Analysis (`bench analyze <run>`)

`analyze` dispatches on the `experiment` field in `run.json` (friction and
coast-down). Completed, safety-stopped and interrupted runs are analyzed;
only steps with a `step_complete` event count.

- Coast: per step, velocity is smoothed with a linear fit over
  non-overlapping 0.25 s windows (at least 5 samples each); α from the slope.
- Inertia: the rotating inertia is the disc plus the motor rotor, which the
  disc record does not know. With acceleration trials the total is recovered
  without a loss model from matched speeds,
  I = K_t·i_q / (α_accel + |α_coast(ω)|), per window; mean, sd and the ratio
  to the disc inertia (the motor-rotor share) go to `summary.json`. Without
  acceleration trials the disc value is used and the basis is marked
  `disc_only`.
- Loss torque τ = I·|α_coast| with that I. All steps pool into one
  speed-binned loss map (`loss-map.csv`, 1 rev/s bins, mean and sd), and a
  least-squares fit τ = a + b·ω + c·ω² (Coulomb, viscous, windage) with
  statistical parameter standard errors; the inertia's relative sd is
  reported beside it and is not folded in.
- Plateau: |mean signed q-current| × K_t per step, listed next to the coast
  fit at the same speed. The sign is kept until after the mean: the ripple
  crosses zero at low speed, and a per-sample |i_q| reads several percent high
  (fixed 2026-11-08, see §8).
- `summary.json` plus SVG plots: speed versus time per step, loss torque
  versus speed with the fit.

## 6. Out of scope

Torque-speed envelope (PNDLM-12), thermal (PNDLM-13), any active braking,
automatic balancing checks.

## 7. Implementation outline

1. `config.py`: `CoastDownConfig`, Tier B safety fields, validation.
2. `safety.py`: `validate_coast_down`; `check_state` gains an explicit
   velocity override (already supported) and no excursion.
3. `experiments/coast_down.py`: sequence above, modelled on
   `response_timing.py` (overrides, readback, stop path).
4. `analysis/coast_down.py` and dispatch in `summarize.analyze`.
5. `cli.py`: subcommand with `--dry-run`, `--accelerate`, `--direction`,
   `--ladder` (comma-separated rpm, each validated against the Tier B cap).
6. Tests in `tests/bench/`: config validation, fake-controller sequence,
   override restore on failure, wait-for-standstill on safety stop,
   analysis on synthetic data with known a, b, c and I.

## 8. Documented deviations and assumptions (2026-11-08)

Results: [measurements/2026-11-01-rotating-losses.json](../measurements/2026-11-01-rotating-losses.json).
These points replace further bench runs for the Phase 0 gate.

- **Plateau cross-check.** The first analysis averaged |i_q| per sample, so it
  read 4–14 % above the coast loss. Re-analysed from the same telemetry with the
  signed mean: −2.9 … +8.3 % (mean 3.8 %). 11 of 14 steps agree within
  0.75 mNm, which is about the coast-fit rmse (0.72 mNm). Three steps at
  2483–2978 rpm stay 1.2 mNm high. Not resolved. The candidate is K_t at speed,
  because K_t was measured on a held rotor. The −1.2 A reading in mode 0 is
  ruled out: it is 30× the gap in current, and it would flip sign with
  direction. Not a blocker, because the coast is the primary measurement.
- **Coulomb term on the final assembly (assumption, not measured).**
  a = 4.4 mNm nominal. For the final assembly the models use a range of
  3–7 mNm (−30 % / +60 %) to cover bearing preload and temperature. These runs
  covered only 19–28 °C on the controller. The RL environment should randomize a in
  this range, and LQR must be checked at both ends. No separate measurement is
  planned unless a check at 7 mNm fails.
- **Losses near zero speed (assumption).** Below ~1 rev/s the loss map is
  noisy (sd 2.2 mNm in the 0.5 rev/s bin), and breakaway torque was not
  measured. The models use a smooth Coulomb term a·tanh(ω/ω_s) with
  ω_s = 0.5 rad/s. One optional run measures breakaway torque with a slow
  torque ramp from rest. It is needed only if the simulation shows that
  stick-slip limits the wheel-stop behaviour.
- **Simulation check, 2026-11-08 (`software/sim`, `phase0-control --losses`).**
  Sampled LQR, 500 Hz, 5 ms delay, nominal and pessimistic design points,
  last 3 s of a 10 s hold. With sensor noise, the LQR torque is about
  120 mNm rms, 17× the high Coulomb term. The relative wheel speed changes sign
  75–105 times per second, so the wheel never rests in stiction. Losses from
  none up to 3× the high corner (21 mNm) change the angle rms by less than
  0.003° and the wheel-speed rms by less than 0.4 rpm. Without noise and with a
  sharp Coulomb step (ω_s = 0.05 rad/s), friction leaves a steady wheel speed of
  0.16 rpm at 7 mNm and 1.5 rpm at 21 mNm. There is no limit cycle. The LQR has
  no integrator, so the friction torque stays as a small speed offset. Result:
  stick-slip does not limit balance or wheel stop. The breakaway run is not
  needed for Phase 0. Breakaway above the Coulomb level (Stribeck) is not
  modelled.
