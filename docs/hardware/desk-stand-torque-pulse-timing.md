# Desk-stand torque-pulse response timing

Supervised hardware pilot on 2026-10-25 for
[PNDLM-8](https://zjor.youtrack.cloud/issue/PNDLM-8). It replaces the
position-mode observation bound of 22–47 ms from the
[position-step analysis](desk-stand-response-timing.md) with a torque-mode
bound measured by the method in [specs/response-timing.md](specs/response-timing.md).

**Result: a host torque command becomes visible in moteus telemetry within
7–13 ms, confirmed within 18–19 ms, at three torque levels.** The bound is set
by the host sample period (5.5 ms), not by the actuator; the true actuator delay
is shorter and remains unidentified.

## Provenance and reproduction

- Runs (all completed, 4 pulses each, two per direction, 20 ms pulses):

  | Pulse torque | Run | Samples | Derived JSON |
  |---|---|---|---|
  | ±0.01 Nm | `20261025T113416Z_response-timing` | 409 | [0010](measurements/20261025-response-timing-0010.json) |
  | ±0.015 Nm | `20261025T114422Z_response-timing` | 433 | [0015](measurements/20261025-response-timing-0015.json) |
  | ±0.02 Nm | `20261025T115034Z_response-timing` | 330 | [0020](measurements/20261025-response-timing-0020.json) |

- Setup: current desk stand, light arrow, moteus r4.11 (hardware revision 8,
  firmware ABI 0x010100) on mj5208, 24.3 V bus, controller 23 °C, fault 0.
  Host: laptop, Python 3.12, moteus 1.1.1, fdcanusb over USB.
- Acquisition code: Git revision `7c3036658e7f4a4597e74b225e6415df8e6ace55`,
  dirty worktree (the pilot tool and its fixes were uncommitted at run time).
- Each run records the full controller configuration, `telemetry.csv`,
  `events.csv` and `run.json`. The derived JSON files carry SHA-256 hashes of
  those inputs. Raw logs stay in ignored
  `hardware/firmware/moteus-host/logs/bench/`.
- Controller configuration was changed only in volatile memory
  (`servo.timeout_mode` 12 → 0 → 12, restored and read back in every run).

Reproduce from `hardware/firmware/moteus-host` and `software/sim`:

```sh
echo PULSE | poetry run bench response-timing          # hardware, supervised
poetry run python -m app.analyze_response_timing \
  ../../hardware/firmware/moteus-host/logs/bench/<run-id> \
  --output ../../docs/hardware/measurements/<date>-response-timing-<torque>.json
```

Configuration used, differing from the spec as written before the pilot:
`quiet_velocity_rev_s` 0.03 (spec: 0.01), `max_velocity_rev_s` 0.8 (spec:
0.25), `pulse_torque_nm` 0.01 / 0.015 / 0.02. See "Pilot findings" for why.

## Host timing

| Quantity | 0.01 Nm | 0.015 Nm | 0.02 Nm |
|---|---|---|---|
| Achieved sample rate | 182 Hz | 183 Hz | 182 Hz |
| Median sample gap | 5.44 ms | 5.44 ms | 5.48 ms |
| Max sample gap in run | 19.6 ms | 17.1 ms | 15.2 ms |
| Max gap inside an observation window | 6.8 ms | 7.0 ms | 7.0 ms |
| CAN round trip median / p95 / p99 | 1.89 / 2.48 / 2.71 ms | 1.90 / 2.51 / 2.83 ms | 1.98 / 2.63 / 3.17 ms |
| CAN round trip max | 15.78 ms | 3.47 ms | 7.57 ms |

The transport is not the bottleneck: median round trip is under 2 ms. The host
(Python + USB) sets the resolution: 200 Hz requested, 182–183 Hz achieved, with
occasional 15–20 ms stalls outside observation windows. This matches the
position-step analysis (median 1.94 ms, max 30 ms).

## Detected response

Detection per spec: first sample whose deviation from the baseline median
exceeds twice the maximum baseline deviation, same sign on two consecutive
samples for confirmation. Times are measured from the host's pulse request.

| Trial | Torque | Current first / confirm | Position first / confirm | Velocity first / confirm | ΔI | Δθ | Δω |
|---|---|---|---|---|---|---|---|
| 0 | +0.01 | — / — | 12.4 / 17.7 | 12.4 / 17.7 | — | +0.0005 | +0.064 |
| 1 | −0.01 | — / — | 12.2 / 17.8 | 12.2 / 17.8 | — | −0.0005 | −0.061 |
| 2 | +0.01 | — / — | 12.5 / 18.8 | 12.5 / 18.8 | — | +0.0006 | +0.088 |
| 3 | −0.01 | — / — | 12.1 / 17.5 | 12.1 / 17.5 | — | −0.0005 | −0.064 |
| 0 | +0.015 | 12.2 / 18.1 | 12.2 / 18.1 | 12.2 / 18.1 | +0.54 | +0.0008 | +0.101 |
| 1 | −0.015 | 7.2 / 12.8 | 12.8 / 17.8 | 12.8 / 17.8 | −0.48 | −0.0006 | −0.078 |
| 2 | +0.015 | 6.7 / 12.3 | 12.3 / 17.6 | 12.3 / 17.6 | +0.32 | +0.0008 | +0.107 |
| 3 | −0.015 | 12.9 / 18.1 | 12.9 / 18.1 | 12.9 / 18.1 | −0.62 | −0.0008 | −0.115 |
| 0 | +0.02 | 7.1 / 12.3 | 12.3 / 18.0 | 7.1 / 12.3 | +0.45 | +0.0013 | +0.054 |
| 1 | −0.02 | 7.4 / 12.3 | 12.3 / 17.5 | 12.3 / 17.5 | −0.57 | −0.0007 | −0.086 |
| 2 | +0.02 | 12.7 / 18.1 | 7.4 / 12.7 | 7.4 / 12.7 | +0.56 | +0.0003 | +0.049 |
| 3 | −0.02 | 7.6 / 12.9 | 12.9 / 19.1 | 12.9 / 19.1 | −0.56 | −0.0008 | −0.104 |

Units: ms, A, rev, rev/s. Watchdog timeout (mode 11) was observed after every
pulse, so each pulse was a single non-refreshed command.

Reading the table:

- Every first detection falls on either the first (≈7 ms) or the second
  (≈12–13 ms) sample after the request. The detection time therefore quantizes
  to the host sample period; the actuator responds within the first sample
  whenever the signal is strong enough to clear the noise threshold there.
- At 0.01 Nm current never clears its noise threshold (0.22–0.31 A) and
  position/velocity clear theirs only at the second sample. At 0.015 and
  0.02 Nm current is visible at the first sample in 5 of 8 pulses.
- Commanded 0.02 Nm produced only 0.45–0.57 A of q-current, which the
  controller reports as 0.012–0.014 Nm. Either torque build-up takes longer
  than 20 ms or the torque-constant estimate differs. Not resolved here.

## Pilot findings (stand, encoder, controller)

- **Velocity noise floor.** Reported velocity at rest varies by up to
  0.025 rev/s while position varies by 0.0001 rev. The spec's 0.01 rev/s quiet
  gate never passed; it was raised to 0.03 rev/s. The angle-span gate stayed at
  0.0005 rev and passed in every accepted baseline.
- **Velocity filter lag.** After the watchdog cut torque at ≈21 ms the reported
  velocity kept rising for another ≈6 ms (0.53 → 0.66 rev/s in an aborted
  0.02 Nm run). The reported velocity lags the true velocity; peak velocity
  limits must allow for that.
- **Safety limit vs. pulse energy.** Peak reported velocity: 0.25 rev/s at
  0.01 Nm, 0.45 at 0.015 Nm, 0.62 at 0.02 Nm. Runs at 0.02 Nm aborted at limits
  of 0.25, 0.45 and 0.6 rev/s before completing at 0.8 rev/s. Net excursion per
  run stayed within 0.0105–0.0174 rev (4–6°).
- **Non-zero current at zero torque.** Baseline mean q-current in mode 10 with
  all gains scaled to zero was −0.07 to −0.09 A (std 0.12–0.16 A). Recorded,
  not explained.
- **Stand balance is marginal.** In one earlier run (`20261025T112603Z`) the
  arrow drifted at zero torque (−0.002 rev/s) and slipped once to 0.23 rev/s;
  later runs stood still. Tier A has reached its useful limit; a balanced
  rotor (Tier B) is needed for stronger excitation.
- **Tool defects fixed during the pilot** (all would have aborted any run):
  first-transaction transport open ≈80 ms exceeded the 50 ms bound; `conf get`
  replies without `OK`; a fresh `moteus.Stream` loses its first reply to the
  flow-control probe; `set_stop(query=True)` reports the pre-stop mode. Test
  fakes now reproduce these behaviors.

## Use in simulation and limits of inference

- **Actuator command delay for the pendulum model: ≤ 13 ms, likely ≤ 7 ms.**
  Use 10 ms as the pessimistic value and 5 ms as nominal. This supersedes the
  22–47 ms position-mode figure. It is a detection bound at 5.5 ms resolution,
  not a measured dead time; a tighter bound needs faster host sampling, not
  more pulses.
- **Loop delay budget.** `phase-0-feasibility.md` allocates ≤ 12 ms total
  loop delay. The `phase0-control --pessimistic --sweep` check (run 2026-10-25)
  passes at 15 ms and fails at 20 ms, so the measured bound leaves roughly
  7 ms of margin on this host and controller design.
- **Host timing model.** Keep the position-step recommendation: replay actual
  request/response times where possible. For a fixed model use 5.5 ms sample
  period, 2 ms round trip, and a 20 ms feedback gap as a robustness scenario.
  This describes the laptop + fdcanusb pipeline, not the planned RPi + pi3hat.
- **Torque build-up.** Do not assume the commanded torque is reached within
  20 ms; measured current reached 60–70 % of the expected value. A longer pulse
  experiment is needed before the simulator can use a torque rate limit.
- **Velocity signal.** The motor-encoder velocity is noisy (±0.025 rev/s) and
  lagged (≈6 ms); position is clean. Estimators in simulation should take the
  velocity from differentiated position or from the IMU, not from the reported
  value as-is.
- The results validate neither the heavier flywheel nor a different
  controller configuration. Remaining scope is tracked in PNDLM-8.
