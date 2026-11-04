# Torque-speed envelope bench experiment — design

Status: measured 2026-11-04, record
[measurements/2026-11-04-torque-speed.json](../measurements/2026-11-04-torque-speed.json)
(248.3 ± 1.0 rpm/V, r_max = 1.00 with Kv 339 from the measured K_t); tool implemented
2026-11-01 (`bench torque-speed`, `bench analyze`, `bench envelope`; plan
`.claude/plans/2026-11-01-torque-speed-bench.md`). YouTrack: PNDLM-12 (T-005.D). Parent spec:
[motor-bench.md](motor-bench.md) §6 step 6. Builds on
[coast-down.md](coast-down.md) (sequence, overrides, Tier B safety) and the
loss and inertia record
[measurements/2026-11-01-rotating-losses.json](../measurements/2026-11-01-rotating-losses.json).
Model under test: the moteus voltage circle in
[physics/phase-0-feasibility.md](../../physics/phase-0-feasibility.md)
§"Torque-speed envelope" and `software/sim/phase0.toml [moteus]`.

## 1. Purpose

`poetry run bench torque-speed --bus 12.0` records, at one bus voltage, how
much torque the motor still delivers as the test disc speeds up. Torque is
not sensed; it is the known rotating inertia times the measured acceleration
plus the loss torque at that speed, both from the 2026-11-01 record. Outputs
after analysis, per bus voltage: delivered torque versus speed for each
commanded torque, the no-load speed and the knee speed with uncertainty. Across
voltages: a fit of the voltage-circle parameter `max_voltage_ratio`, which
`phase0.toml` marks UNSURE (0.85–0.95), so the simulator gets a measured
envelope for the 12.0–16.8 V battery range.

Why it matters: the swing-up and the catch spend wheel momentum, and the
simulator's gate keeps the wheel under 70 % of no-load speed at the catch. If
the real knee is lower than the model predicts, the wheel runs out of torque
earlier and the swing-up strategy in the feasibility doc is wrong.

## 2. Prerequisites and bench conditions

- **Adjustable bus.** The adjustable lab supply (up to 20 A, confirmed
  2026-11-01), set by hand to each voltage of the sweep with its current limit
  at 8 A or more (the 0.15 Nm top step draws 0.15 / 0.0244 = 6.1 A by the
  measured K_t, with current-noise peaks at 6.5 A). It cannot sink current, so the
  disc is never braked by the motor: every stop is `set_stop()` and a free
  coast, as in coast-down. The 24 V / 5 A supply of PNDLM-11 is not used.
- **Disc rated to 4000 rpm.** [test-disc.md](test-disc.md) §4 is extended
  to 4000 rpm (ω = 419 rad/s): hoop stress ≈ 1.2 MPa, spoke tension ≈ 3.7 MPa,
  tip speed 31 m/s, stored energy ≈ 33 J, 1 g imbalance at 70 mm ≈ 12 N.
  Reserve stays ≥ 10× on every stress figure. Approved 2026-11-01.
- **Stand anchored** (rail anchors fitted), guard closed, nobody in the disc
  plane. Same disc, screws and threadlocker as PNDLM-11; inertia and loss fit
  taken from the record, not re-measured.
- Controller: moteus r4.11, ID 1, fdcanusb, 100 ms watchdog. Firmware
  `servo.max_voltage` is 30 V; the sweep stays below 17 V.
- Expected no-load speeds (model, Kv 330): 2740 rpm at 12.0 V, 3380 at
  14.8 V, 3840 at 16.8 V. With Kv from the measured K_t (339 rpm/V) the
  16.8 V figure becomes about 3950 rpm, at the disc rating. See §4 for how
  the cap handles that.

## 3. Sequence

One run per bus voltage. `--bus <V>` is mandatory and names the expected
voltage; the sweep is run as six invocations, 11.0 V first.

1. **Preflight** stopped: fault 0, temperature, finite position, mode 0.
   Bus voltage must be within `bus_tolerance_v` of `--bus`, else the tool
   refuses to start and prints the measured value. This catches a supply left
   at the previous setting.
2. **Operator gate** per step: the tool prints the step (torque, direction,
   expected no-load speed at this bus) and waits for `GO`. Any other input
   ends the run cleanly after the current coast.
3. **Quiet baseline** at rest, as in `--accelerate`.
4. **Torque step**: feedforward torque (`kp_scale = kd_scale = 0`,
   `maximum_torque = cap`) from rest, refreshed every sample. The commanded
   value is nominal: the firmware converts it with its own `motor.Kv` (296,
   K_t ≈ 0.028 Nm/A), so a 0.05 Nm command draws ≈ 1.8 A and delivers ≈ 0.043
   Nm by the measured K_t (seen in run `20261101T105228Z`). The tool sends the
   nominal values of `torques_nm` unchanged (the Tier B cap applies to the
   firmware-reported torque, as `check_state` already does) and the analysis
   works from measured q-current and measured acceleration, never from the
   nominal command. The step runs until one of:
   - *saturated*: the windowed acceleration (0.25 s linear fit, same helper as
     coast-down) stays below `saturation_alpha_rev_s2` for `saturation_settle_s`
     while `|velocity| > min_saturation_rev_s` (so a stalled or barely moving
     rotor is not mistaken for no-load);
   - *velocity-capped*: `|velocity| ≥ max_velocity_rev_s − cap_margin_rev_s`;
   - *supply-limited*: the bus voltage falls below `--bus` − `sag_stop_v`
     on any sample; the tool sends `set_stop()` and the step ends cleanly
     (a supply in current fold-back is a result, not a safety event; the 10 V
     safety floor stays as the last line);
   - *timeout*: `step_timeout_s`.
   Saturation reached above `max_velocity_rev_s − servo.max_velocity_derate`
   counts as *velocity-capped*, not *saturated*: in the derate band the
   firmware itself removes the torque. The termination reason is written to
   `events.csv` and `run.json` per step.
5. **Hold**: keep the torque command for `hold_s` after a *saturated* end so
   the plateau current and bus voltage at no-load are on record. No hold after
   a *velocity-capped* or *supply-limited* end; the step goes straight to the
   coast.
6. **Coast**: `set_stop()` and query only until the coast-down stop rule
   (`coast_stop_rev_s` for `coast_settle_s`, or `coast_timeout_s`).
7. Next step after a standstill wait, as in coast-down.

Step order: torques ascending, each in +1 then −1, so the first spin-up to a
new no-load speed is the gentlest one. Across runs the operator goes 11.0 V
first and up; the tool prints a warning if `--bus` skips an entry of
`bus_voltages_v` that has no completed run yet in `logs/bench` for this
experiment (ramp discipline, not a hard stop). The 11.0 V run doubles as the shakedown of
the tool: its no-load speed is below the 3000 rpm the disc has already run.

Defaults (`bench.toml [torque_speed]`, my calls):

| Key | Value | Note |
|---|---|---|
| `sample_rate_hz` | 200 | as coast-down |
| `bus_voltages_v` | 11.0, 12.0, 13.2, 14.8, 16.0, 16.8 | the accepted values of `--bus`; any other value needs `--bus` plus `--allow-any-bus`. 11.0 V is the shakedown point: no-load ≈ 2500 rpm, inside the range already run; lower is pointless because of the 10 V safety floor and the 0.8 V sag stop |
| `bus_tolerance_v` | 0.2 | preflight check against `--bus` |
| `torques_nm` | 0.05, 0.10, 0.15 | `--torques` overrides; each ≤ Tier B cap. The reported torque carries about +0.03 Nm of current noise: a 0.18 Nm step tripped the 0.20 Nm Tier B cap on one sample (2026-11-04), so 0.15 Nm is the top step (peaks 0.18 Nm reported, 6.5 A) |
| `directions` | +1, −1 | `--direction` restricts to one |
| `saturation_alpha_rev_s2` | 1.0 | at 0.05 Nm the initial acceleration is ≈ 21 rev/s²; 1 rev/s² is 5 % of that |
| `saturation_settle_s` | 0.5 | |
| `min_saturation_rev_s` | 5.0 | 300 rpm; below this "no acceleration" means stuck, not no-load |
| `hold_s` | 1.0 | plateau current and bus voltage at no-load |
| `step_timeout_s` | 15 | 0.05 Nm reaches 2740 rpm in ≈ 2.5 s plus the taper |
| `cap_margin_rev_s` | 3.0 | step ends *velocity-capped* at 63.7 rev/s, below the firmware derate band that starts at `max_velocity − max_velocity_derate` = 64.7 rev/s |
| `sag_stop_v` | 0.8 | step ends *supply-limited* when the bus drops this far below `--bus`; checked every sample. At 11.0 V the stop is 10.2 V, one 0.1 V telemetry step above the 10.0 V `min_bus_voltage_v` floor, so a fold-back ends the step *supply-limited* before the safety check trips; `validate` requires `--bus` − `sag_stop_v` ≥ floor + 0.1 V for the listed voltages and for any `--bus` |
| `coast_stop_rev_s`, `coast_settle_s`, `coast_timeout_s` | 0.2, 0.25, 120 | as coast-down |
| `max_velocity_rev_s` | 66.7 | 4000 rpm, host stop and `servo.max_velocity` override |
| `max_torque_nm` | 0.20 | Tier B cap |
| `inertia_record`, `loss_record` | the 2026-11-01 record | `effective_total_kg_m2`, pooled a, b, c |
| `kt_nm_per_a` | 0.0244 | PNDLM-9 |

## 4. Safety

- **Tier B caps** rise with the disc rating: `max_rotating_velocity_rev_s`
  55 → 70 in `[safety]`; the experiment uses 66.7 (4000 rpm) and
  `servo.max_velocity` is overridden to the same value, so the firmware
  derates torque from 64.7 rev/s and the host stops at 66.7. Coast-down keeps
  its own 55 rev/s setting; nothing else changes for Tier A.
- At 16.8 V the no-load speed may sit above the cap. Then the step ends
  *velocity-capped*, the no-load speed for that voltage is reported as a lower
  bound, and the knee from the torque taper below the cap is still valid. No
  attempt is made to reach it.
- **Overrides**, volatile only, restored in `finally` with readback, as in
  coast-down: `servo.timeout_mode 0`, position bounds NaN,
  `servo.max_velocity` 66.7, flux brake margin per firmware key. A watchdog
  trip (mode 11) during a commanded phase ends the run as a safety stop.
- **Supply** cannot sink, so no braking anywhere; the hold phase keeps the
  torque command, it never commands a lower speed.
- Every sample: fault, bus voltage (`min_bus_voltage_v` 10 V is already
  below the sweep; `max_bus_voltage_v` 30 V), controller temperature, phase
  current, Tier B velocity and torque.
- Ctrl-C, lost reply and fault paths exactly as coast-down.

## 5. Analysis

`bench analyze <run>` on a torque-speed run (per bus voltage):

- **Delivered torque per window** of the torque phase: τ(ω) = I_eff·α(ω) +
  τ_loss(ω), with I_eff and the pooled a, b, c from the record, over
  non-overlapping speed windows: a window closes after 1.5 rev/s of speed and
  at least 10 rows, so it is short in time under hard acceleration (about
  3 rev/s at 0.15 Nm) and long on the plateau. Fixed 0.25 s windows spanned
  16 rev/s at 0.15 Nm and could not place the knee (2026-11-04). The
  uncertainty per window combines the inertia relative sd (2.6 %) and the
  loss fit rmse.
- **Supply-limited flag** per window: mean bus voltage below `--bus` −
  `sag_stop_v`, the host's own stop level. The lead drop at the step
  currents (about 0.1 Ω, 0.6 V at 6 A) stays above it and is not a sag; the
  measured bus travels with every knee and no-load point, so the drop does
  not bias the voltage-circle fit. Flagged windows are kept in the table and
  excluded from fits.
- **No-load speed** per direction: the mean over every saturated step in that
  direction of its hold-phase mean speed (the plateau does not depend on the
  command); hold rows whose bus is below `--bus` − `sag_stop_v` are
  dropped and counted; uncertainty = sd over the hold plus the
  loss-induced bias (the loss torque at that speed divided by the slope of the
  taper). Velocity-capped steps give a lower bound and are marked so.
- **Low-speed torque** per step: the delivered torque from one line fit of
  |v| against time over the raw torque-phase rows (bus not sagging) between
  `min_saturation_rev_s` and 50 % of the expected no-load speed at this bus,
  at their mean speed, and the mean q-current there (one fit over the whole
  band at any torque). This, not the nominal
  command, is the reference for the knee and the fit (the firmware's K_t
  differs from the measured one by about 14 %).
- **Knee speed** per step and direction: the speed at which the windowed
  delivered torque falls below 90 % of the step's low-speed torque for good:
  the last window at or above the threshold and the next one bracket the
  crossing, which is interpolated linearly in torque between them, with half
  their speed distance as uncertainty. A single noisy window below the
  threshold with full torque after it is not a knee (the voltage limit never
  lets the torque recover). The knee carries the mean q-current and the mean
  measured bus voltage of the two bracketing windows.
- **Plateau current**: mean q-current and bus voltage over the hold, for the
  record.
- Direction symmetry: no-load and knee reported per direction; the mean is
  used downstream and the difference is reported.
- Outputs: `summary.json`, `torque-vs-speed.csv` (per step, per window),
  `plots/torque-vs-speed.svg` (one curve per commanded torque, both
  directions, flagged windows hollow), `plots/speed-vs-time.svg`.

`bench envelope <run>...` across runs (new subcommand, my call), or
`bench analyze --envelope`:

- Collects no-load and knee points from each per-voltage summary.
- **Voltage-circle fit.** Model as in `phase0.toml`:
  V_eff = 0.5·V_bus·r_max·(1 − m), m = 0.15 fixed (firmware constant);
  (v_per_hz·f + R·i_q)² + (π·p·L·i_q·f)² ≤ V_eff², with R = 0.0467 Ω and
  L_q = 44.8 µH from the controller config dump, p = 14. v_per_hz is taken
  from the measured K_t (Kv = 8.27 / 0.0244 = 339 rpm/V, ±3 % from PNDLM-9),
  not fitted: with only no-load speeds, r_max and v_per_hz are degenerate.
  The single fitted parameter is r_max. Its samples are every saturated
  no-load point (i_q = the hold's mean q-current, the loss current; V_bus =
  the hold's measured bus) and every knee (i_q and V_bus = the measured
  q-current and bus of the bracketing windows) at all voltages; the nominal
  `--bus` is used only when no bus was measured, and the point is marked so.
  r_max is reported per Kv candidate as mean, sd and se, with the mean per
  group (no-load, knee). For the Kv from K_t, the K_t uncertainty (±3 %) is
  propagated: `kt_term` is half the change of the mean r_max between
  Kv·1.03 and Kv·0.97, and the reported se is √(se_scatter² + kt_term²).
- The fit is degenerate between r_max and v_per_hz, so the **primary
  deliverable** is the measured quantity: no-load speed versus V_bus, slope
  (rpm/V, "effective no-load Kv") and intercept with standard errors, plus the
  knee table. r_max is reported twice, once with Kv 339 from the measured K_t
  and once with the firmware's `motor.Kv` 296, each with its standard error,
  and a value above 1 is reported as such (it means that Kv is too high for
  the circle, not that the firmware exceeds its PWM range). PNDLM-14 writes
  into `phase0.toml` the measured slope and the (Kv, r_max) pair that
  reproduces it with r_max ≤ 1. A non-zero intercept or curvature in the
  no-load line is reported, not hidden.
- Output `logs/bench/<timestamp>_envelope/summary.json` and
  `plots/envelope.svg` (knee and no-load points over V_bus with the fitted
  circle), listing the source run IDs.

## 6. Record and acceptance

The measurement record `docs/hardware/measurements/<date>-torque-speed.json`
holds: bus voltages, per-voltage no-load and knee tables with uncertainty and
termination reasons, the r_max fit with uncertainty and the inputs it used
(K_t, R, L, m), the operating envelope of each run, the six run IDs, and the
velocity-cap note for 16.8 V. PNDLM-14 then transfers r_max, the measured
no-load speeds and the envelope table into `phase0.toml` and the feasibility
doc.

Done when (from the ticket): effective current/voltage limits and a
torque-speed envelope with uncertainty and run provenance are usable by the
simulator. Concretely: the five design-range voltages recorded (11.0 V is
extra), both directions, no fault or
safety stop (*supply-limited* and *velocity-capped* step ends are results,
not failures), the no-load slope and r_max with standard errors, every
step's termination reason on record.

## 7. Out of scope

External load or dynamometer, braking, thermal plateaus (PNDLM-13), measuring
R, L or Kv independently (controller calibration values are used as inputs),
battery tests (a 4S pack can confirm the 16.8 V point later).

## 8. Implementation outline

As built (2026-11-01), in `hardware/firmware/moteus-host/moteus_host/bench/`:

- `experiments/tier_b.py`: the shared Tier B session (connect, volatile
  overrides restored with readback, stop with retries, standstill wait, status
  classification). Coast-down uses it too.
- `experiments/torque_speed.py`: sequence of §3, `StepMonitor` (termination
  reasons), bus preflight, ramp warning, operator gate.
- `analysis/torque_speed.py`: §5 per run, dispatched from `bench analyze`;
  `analysis/envelope.py`: across runs, `bench envelope`.
- `config.py`, `safety.py`, `cli.py`, `bench.toml [torque_speed]`; the README
  section.
- Tests in `tests/bench/`: `test_torque_speed_config.py`,
  `test_torque_speed_monitor.py`, `test_torque_speed.py`,
  `test_torque_speed_analysis.py`, `test_envelope.py`, with `tier_b_fakes.py`
  as the shared fake controller.

## 9. Open points


- Whether the firmware's own base-velocity computation (visible in telemetry?)
  can be logged for comparison; builder's call, not required.
- Bus-current telemetry is not available from the controller; supply current
  is not recorded. Sag detection stands in for it.
