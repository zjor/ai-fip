# MJ5208 static torque fixture

## 1. Purpose

This fixture independently measures the mj5208 torque constant for T-005.C. A
rigid lever attached to the rotor presses vertically on a compression scale
while moteus commands short Q-axis current plateaus. The scale is the force
reference; moteus-reported torque is recorded only for comparison.

The first test is deliberately limited to ±8 A and approximately ±0.20 Nm. It
does not characterize peak torque, continuous thermal torque or torque at speed.

## 2. Measurement principle

```text
                         motor axis
                             O
                             │
    left contact ●───────────┼───────────● right contact
                 <--- 100.0 mm --->
                                                ↓ force
                                          ┌───────────┐
                                          │   scale   │
                                          └───────────┘
```

Use a balanced crossbar with a contact point on each side. Put the scale under
the contact loaded by the commanded torque direction; move it to the other side
for the opposite direction. The force must be perpendicular to the lever.

For effective radius $r$ and zero-corrected scale reading $m$:

$$
\tau = m g r
$$

With $r = 0.1000$ m and the reading expressed in grams:

$$
\tau[\mathrm{N\,m}] = m[\mathrm{g}] \times 0.000980665
$$

The fitted model is:

$$
\tau = K_t I_q + b
$$

where $I_q$ is measured Q-axis current, not merely its requested value. The
slope is the independently measured torque constant. The intercept contains
scale preload, zero error, residual cogging and fixture bias.

## 3. Mechanical design

### 3.1 Lever

- balanced metal crossbar, nominally 200 mm between contact centerlines;
- effective radius from the motor axis to each force line: **100.0 mm**;
- measure each radius after assembly to ±0.2 mm or better;
- aluminium flat bar is preferred; a printed lever is not the measurement
  reference because creep and bending add avoidable error;
- rigid hub using the mj5208 rotor M3 pattern, with no angular play;
- an adjustable rounded metal contact at each end, such as an M4 screw with a
  ball or acorn end;
- bar horizontal at the measurement point, so the vertical scale force is
  perpendicular to the radius.

The effective radius is the perpendicular distance from the motor axis to the
vertical force line. It is not automatically the physical bar length if the bar
or contact is angled.

### 3.2 Load path

- bolt the motor stator bracket and scale base to the same stiff baseplate;
- do not hold the motor, lever or scale by hand;
- constrain the scale so it cannot slide sideways;
- include a secondary hard stop just beyond normal scale deflection;
- add a loose guard or restraint that catches the lever if contact is lost;
- leave the unloaded end of the crossbar clear.

The scale is a measurement instrument, not the only structural restraint.

### 3.3 Scale

The available compression scale is rated to 3 kg with 0.1 g display resolution.
At 100 mm its nominal torque increment is 0.0000981 Nm. This is more than
sufficient; calibration accuracy, repeatability, force alignment and lever
radius dominate the uncertainty.

Check the scale before use with at least two known masses spanning roughly
0.2–1.0 kg. Record displayed resolution, repeatability, auto-zero behavior,
auto-off behavior and whether the reading is stable under a constant load.

## 4. Range

| Scale reading | Torque at 100.0 mm |
|---:|---:|
| 10 g | 0.00981 Nm |
| 100 g | 0.0981 Nm |
| 200 g | 0.196 Nm |
| 500 g | 0.490 Nm |
| 1000 g | 0.981 Nm |
| 1733 g | 1.70 Nm |

The scale can nominally contain the published 1.7 Nm motor peak at this radius,
but scale capacity alone does not qualify the fixture for a peak-torque test.
The hub, bracket, baseplate, contact, restraint, supply and thermal limits would
all require a separate Tier-C review.

## 5. Initial test protocol

### 5.1 Preconditions

- crossbar and hub inspected and fasteners marked against loosening;
- motor stator, scale and hard stop fixed to the common baseplate;
- contact radius measured and recorded for each side;
- scale checked with known masses, zeroed and prevented from auto-off;
- no hands or loose objects in the lever envelope;
- physical power disconnect immediately accessible;
- controller stopped, fault 0, bus voltage and temperature in limits;
- software watchdog and unconditional `set_stop()` path verified with a fake
  controller before the physical run.

### 5.2 Current plateaus

Use commanded Q-current plateaus of:

```text
0, 0.5, 1, 2, 4, 6, 8 A
```

The 0.5 A and 1 A points were added from the T-005.A pulse pilot (PNDLM-9
addendum): the pendulum works near zero torque, and the pilot saw 60–80 % of the
expected q-current at 0.02 Nm, so the fit must cover the low-current range.

Run positive and negative directions separately, placing the scale beneath the
appropriate compression contact. Use measured `q_current_a` in analysis. Start
with 2–3 second plateaus and stop between plateaus. Repeat every nonzero point
three times. Abort on motion, loss of contact, unstable load, unexpected noise,
fault, voltage violation, current violation or temperature violation.

At the planning value $K_t \approx 0.025$ Nm/A, 8 A should produce about 0.20
Nm or 204 g at 100 mm. Do not raise the current range merely because the scale
has unused capacity.

### 5.3 Manual scale readings

The first version uses manual readings. The host keeps resending the plateau
command while current is active, which refreshes the firmware watchdog; a
blocking terminal prompt is only ever shown while the controller is stopped.
The moteus firmware watchdog only covers position mode (the timeout check in
`fw/bldc_servo_control.h` runs for `kPosition` and `kStayWithinBounds`, never
for `kCurrent`; a 2026-10-25 probe on this controller confirmed that a
`set_current` command stays in mode 9 after 2.5 timeouts). Each plateau is
therefore sent as a position-mode command with zero gains and a feedforward
torque of amps × $K_{t,fw}$, where $K_{t,fw} = 8.27 / \texttt{motor.Kv}$ is the
firmware's linear torque constant (rotation cutoff 10000 A on this motor). The
fit uses the measured Q-current, so the command path does not enter the
result.
The available scale has no hold function, so the operator watches the display
during the plateau and types the stable value afterwards. The implemented
sequence of `poetry run bench torque-constant` is:

1. the controller is stopped, its configuration is captured, and the run
   refuses to start without a positive finite `motor.Kv`; then a 0 A (0 Nm)
   probe command with the 0.1 s per-command watchdog is sent and the host
   stays silent for 2.5 timeouts; the run refuses to continue unless the
   controller reports its timeout state (mode 11, or 12), which proves the
   firmware watchdog applies to the plateau command path on this controller;
2. the operator types the measured contact radius for the positive and the
   negative side in millimetres, then a free-text scale-check note (known
   masses, resolution, auto-off); these go to `events.csv`;
3. the operator types the confirmation word `LEVER`; anything else aborts;
4. the prompt shows the suggested next current from the standard sequence; the
   operator types a current in amps, optionally followed by a plateau time in
   seconds (`4 5` is 4 A for 5 s), or `q` to finish;
5. a short query-only baseline is recorded, then a terminal bell marks the
   start of the plateau; the current is ramped linearly from 0 to the target
   over `ramp_s` (0.25 s), because an instant 6 A step compressed the fixture
   about 0.3 mm in 6 ms and tripped the 0.05 rev/s stop on 2026-10-25; a
   countdown prints each second while telemetry is
   recorded; the firmware's enable and calibration states (modes 2–4) are
   tolerated for the first 50 ms of a plateau, after that only position mode
   (10) is accepted; note that the stopped-state baseline Q-current is
   uncalibrated (about 1.2 A on this controller) and is reported only, the
   0 A plateau is the real zero point;
6. the controller stops and is confirmed stopped;
7. the operator types the observed scale reading in grams (blank skips the
   point) and an optional note; the script records plateau number, current,
   side and the reading in `events.csv`;
8. the script waits the configured rest time before the next plateau.

A camera recording of the scale display is useful evidence when the scale lacks
a data interface.

**Zero-command torque (measured 2026-10-25).** A 0 Nm position-mode command
with zero gains is not torque-free on this controller. Right after the enable
the motor applied about −0.46 A (−0.013 Nm reported) for roughly 100 ms, and at
rest it keeps a reported +0.1 to +0.2 A (+0.003 to +0.005 Nm); the firmware
torque path (`ISR_DoPositionCommon`: PID × 0 + feedforward 0 + inertia term 0)
commands zero, so the mechanism is in the current loop or its sensing and is
not identified. On a balanced lever this pulse lifted the lever 7.7° off the
scale and tripped the velocity stop (data:
[2026-10-25-zero-torque-probe.csv](measurements/2026-10-25-zero-torque-probe.csv),
script `hardware/firmware/moteus-host/tools/zero_torque_probe.py`).
Consequences: the hard stop of §3 is mandatory on the side away from the scale,
or the lever needs a preload of at least 30 g at 105 mm (twice the 12.6 g
equivalent of the pulse) so it stays in contact; the first 20 % of each plateau
(0.6 s) is already excluded from the fit, which covers the transient; the
steady residual appears in the fit intercept and in the 0 A plateau reading.

## 6. Software

`poetry run bench torque-constant` (module
`hardware/firmware/moteus-host/moteus_host/bench/experiments/torque_constant.py`)
reuses the bench recorder and safety layer. It:

- requires the confirmation word `LEVER`;
- commands each plateau as position-mode feedforward torque amps × 8.27 /
  `motor.Kv` with `kp_scale`, `kd_scale` and `ilimit_scale` 0 and
  `maximum_torque` = 8 A × $K_{t,fw}$ (see §5.3: current mode has no firmware
  watchdog); the firmware commands D-axis current 0; the requested amps are
  recorded in `command_q_current_a` next to `feedforward_torque_nm`;
- refreshes the firmware watchdog by resending the command every sample
  (200 Hz requested) with a 0.1 s per-command timeout, and a 0 A probe at
  startup proves the timeout fires on this command path before any torque is
  commanded; no `servo.timeout_mode` override and no `conf write`;
- refuses commanded currents above `max_current_a` (8 A, bounded by the tracked
  `safety.max_phase_current_a`) and stops if the measured current exceeds
  `max_current_a + current_overshoot_a` (8.5 A), reported torque exceeds
  `max_reported_torque_nm` (0.30 Nm), speed exceeds 0.05 rev/s or the rotor
  moves more than 0.01 rev from its start;
- records requested and measured Q/D current, position, velocity, reported
  torque, bus voltage, controller temperature, fault, mode and host timing at
  F32 current resolution;
- stops between plateaus and on normal exit, fault, lost response, timeout or
  `Ctrl-C`;
- records the contact radii, the scale-check note, every scale reading and
  operator note in `events.csv`;
- enforces a rest time between plateaus (`rest_s`, 5 s) and a session I²t
  budget (`session_i2t_budget_a2s`, 2500 A²s; the standard profile uses about
  2180 A²s). The mj5208 has no thermistor and the controller's motor
  temperature input is not wired, so this budget plus the board temperature
  stop at 65 °C are the only thermal protection. An 8 A plateau for 3 s puts
  about 13 J into the copper (0.047 Ω phase resistance).

All values live in `[torque_constant]` of `bench.toml`; `--dry-run` prints the
resolved settings without connecting.

### 6.1 Rise-time mode

`poetry run bench torque-constant --rise-time` answers the PNDLM-9 addendum
(q-current rise time after a torque step). With the rotor held by the fixture
it sends feedforward torque steps through the same command path as the
pendulum controller (`set_position` with zero gains and `feedforward_torque`),
capped at `max_rise_torque_nm` (0.05 Nm) for `rise_step_s` (0.5 s), sampling at
`rise_sample_rate_hz` (400 Hz requested). No scale readings are asked. The
host sees only CAN replies, so a rise time equal to the sample spacing is an
upper bound, not a measurement; the analyzer reports `"none within one
sample"` when every step reaches 95 % of its steady current in the first
sample.

### 6.2 Offline analysis

`software/sim/app/analyze_torque_constant.py` converts every reading to torque
at that side's radius, fits $\tau = K_t I_q + b$ per direction against measured
q-current with standard errors and residual RMS, repeats the fit against
moteus-reported torque and against the *commanded* current (the first 0.5 A
plateau on 2026-10-25 measured only 0.24 A, so torque per commanded amp and
per measured amp differ), fits the low-current points (commanded 0.5 A and
1 A) separately, evaluates the acceptance flags of section 7, computes rise
times to 63 % and 95 % for step trials, and writes one JSON with SHA-256
hashes of the raw files. The `simulator` block carries the values the
simulator needs: $K_t$ per measured amp, $K_t$ per commanded amp
(`kt_per_commanded_a_nm_per_a`, the gain a torque request actually gets),
low-current $K_t$, firmware $K_t$ (8.27/Kv) for comparison, and
`torque_rate_limit`: the largest 95 % rise time in milliseconds, the string
`"none within one sample"` when every step reached 95 % in its first sample,
the string `"not reached"` when no step reached 95 % within the hold, or null
when the run has no steps.

The analyzer reports statistical uncertainty of the fit only. Scale
calibration, radius and alignment uncertainty (section 7) are propagated by
hand in the measurement report.

## 7. Uncertainty and acceptance

At minimum, propagate:

- scale calibration and repeatability;
- scale resolution;
- left/right effective-radius measurement;
- lever non-perpendicularity, using the measured sine correction if needed;
- contact-position repeatability;
- measured-current variation during the reading window;
- regression residuals and positive/negative slope disagreement.

Accept the initial result when:

- force is monotonic with measured current in both directions;
- repeated readings at each plateau agree within a declared tolerance;
- positive and negative fitted $|K_t|$ values agree within 5%, or the asymmetry
  is investigated;
- the fit includes at least four nonzero current magnitudes per direction;
- no fixture motion, controller fault or safety-limit violation occurred;
- raw telemetry, events, controller configuration, dimensions and scale checks
  are preserved with the run ID.

## 8. Safety boundary

The initial fixture is approved only for short plateaus up to 8 A. It is not yet
approved for the published 1.7 Nm peak, continuous stall/thermal testing or
rotating torque-speed work. Those tests require a reviewed restraint, motor
temperature instrumentation and, for rotating tests, a retained balanced rotor
and guard.
