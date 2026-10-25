# Desk-stand response timing

Offline analysis on 2026-10-25 for [PNDLM-8](https://zjor.youtrack.cloud/issue/PNDLM-8).
No hardware commands were issued. The existing encoder telemetry is sufficient
to measure observed tracking and host communication timing; it does not identify
a pure motor/torque-command delay.

## Provenance and reproduction

- Run: `20260920T172343Z_position-step`, completed 2026-09-20, 1700 samples,
  eight moves, two repetitions of +30°, center, −30°, center.
- Setup: current desk stand, light arrow, position mode, anticogging scale 0.
- Requested sampling: 200 Hz; 0.5 s initial hold, 1 s per move; torque cap
  0.10 Nm, trajectory velocity cap 0.5 rev/s, acceleration cap 4 rev/s².
- Acquisition: moteus 1.1.1, Python 3.12.6, recorded Git revision
  `d5cb463948cb06ea6c811be8302f6b4e897265c7`, dirty worktree. The dirty flag means
  that revision alone cannot reconstruct the exact acquisition code.
- Raw files remain in ignored
  `hardware/firmware/moteus-host/logs/bench/20260920T172343Z_position-step/`.
  The [derived JSON](measurements/20260920-position-step.json) records SHA-256
  hashes of telemetry, events, metadata and controller configuration, as well
  as all per-move results. It is an analysis artifact, not a copy of raw logs.

From `software/sim` (standard library only):

```sh
poetry run python -m app.analyze_position_step \
  ../../hardware/firmware/moteus-host/logs/bench/20260920T172343Z_position-step \
  --output ../../docs/hardware/measurements/20260920-position-step.json
poetry run python -m unittest discover -s tests -p test_position_step_analysis.py
```

## Host timing

All times below are milliseconds. Percentiles use linear interpolation over
this recording, not a confidence bound on future operation.

| Quantity | Median | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Request → response (RTT) | 1.943 | 2.396 | 2.746 | 30.083 |
| Request minus scheduled time | 0.411 | 0.528 | 9.408 | 25.816 |
| Response minus scheduled time | 2.365 | 2.915 | 13.619 | 30.591 |
| Consecutive response gap | 5.004 | 5.620 | 5.936 | 33.571 |

There are 2/1700 RTTs above 10 ms, and 5/1699 response gaps above 10 ms.
Largest RTTs: sample 477 (30.083 ms), 69 (22.860 ms), 1214 (9.812 ms).
The maximum is not at a target change. It still affects a continuously updated
control loop. Late cycles are followed by catch-up requests: the smallest
response gap is 1.281 ms. Consecutive late samples are therefore correlated.

The timestamp called `host_time_s` equals response time. `target_changed`
events occur before waiting for the next scheduled transmission; they are not
the command send time. This analysis uses the first **request timestamp with a
changed commanded position** as time zero. RTT is not one-way latency and must
not be divided by two without additional evidence. Request time includes host
software before actual bus transmission; no bus or controller sample timestamp
was recorded.

## Detectable movement

For each move, take the median angle over the preceding 100 ms of received
samples as the baseline. Define a directional displacement threshold as
`max(0.1°, 2 × maximum absolute baseline deviation from its median)`. Require
three consecutive samples above it; report the first sample and the third
(confirmation). This is an explicit analysis convention, not a physical
definition of rotor start. The baseline includes real hold oscillations as
well as sensor noise; it is not an independent noise calibration.

| Start sample | Move | Threshold, ° | Previous response, ms | First of three, ms | Confirmation, ms |
|---:|---|---:|---:|---:|---:|
| 100 | center → +30° | 0.100 | 16.97 | 21.71 | 32.58 |
| 300 | +30° → center | 0.100 | 16.65 | 22.08 | 32.25 |
| 500 | center → −30° | 0.242 | 22.21 | 26.78 | 36.84 |
| 700 | −30° → center | 0.401 | 41.85 | 46.55 | 57.26 |
| 900 | center → +30° | 0.203 | 16.69 | 21.86 | 31.54 |
| 1100 | +30° → center | 0.100 | 16.93 | 21.89 | 31.79 |
| 1300 | center → −30° | 0.313 | 30.28 | 35.51 | 45.51 |
| 1500 | −30° → center | 0.357 | 26.46 | 31.88 | 41.97 |

The first qualifying telemetry response arrives 21.71–46.55 ms after the host
request; online confirmation takes 31.54–57.26 ms. The preceding response column
brackets observations only: controller sampling, filtering and return transport
prevent interpreting it as a bracket on physical onset. In particular, earlier
isolated crossings may exist before the accepted three-sample sequence.

Threshold sensitivity matters. Across moves, a fixed 0.1° threshold gives
15.32–26.78 ms, 0.25° gives 26.46–35.51 ms, and 0.5° gives 36.72–51.83 ms.
The 0.1° threshold is smaller than the pre-command deviation on five moves;
those detections cannot reliably distinguish new motion from hold oscillation.
The adaptive thresholds exceed all observed baseline deviations, but do not
prove causality or remove friction/trajectory effects. Only two repetitions per
transition exist: movement-onset p95/p99 estimates would not be meaningful.

The configured source-0 encoder PLL bandwidth is 200 Hz. Its phase delay is
not measured here. Angle changes also require finite acceleration to become
detectable. A pure command-to-torque dead time cannot be separated from these
effects using the present position trajectory.

## Tracking and repeatability

The tracking band is |position error| ≤ 0.005 rev (1.8°) **and** |velocity| ≤
0.02 rev/s. First entry is distinct from remaining in that band. A final entry
at the last sample has zero measured dwell and is not evidence of settling.

| Start sample | First entry, ms | Final uninterrupted dwell before next command/end, ms | Median position error in final 100 ms, ° | Peak absolute velocity in final 100 ms, rev/s |
|---:|---:|---:|---:|---:|
| 100 | 302.00 | 124.82 | −0.623 | 0.018 |
| 300 | 296.61 | 0 | +0.016 | 0.056 |
| 500 | 267.18 | 0 | +0.370 | 0.102 |
| 700 | 331.49 | 0 | −0.297 | 0.056 |
| 900 | 341.88 | 85.90 | −0.562 | 0.023 |
| 1100 | 282.26 | none: ends outside band | +0.019 | 0.072 |
| 1300 | 275.79 | 0 | +0.334 | 0.102 |
| 1500 | 267.07 | 0 | −0.288 | 0.070 |

All eight moves enter the band in 267.07–341.88 ms, but only the first has a
final uninterrupted dwell of at least 100 ms. Position errors in all final
100 ms windows are within −0.650° to +0.562°; the velocity condition causes
the repeated exits. Pairwise differences between first-entry times for the
same transition are 39.87, 14.35, 8.61 and 64.42 ms. Median final position
errors repeat within 0.062° across each pair, despite speed oscillation.

Small overshoot exists on both +30° → center moves: 0.137°. The other moves
do not cross their targets. No faults were recorded.

These results refine the historical 2026-09-20 journal: its approximate
265–340 ms figures describe first entry, not sustained settling; its
“no overshoot” statement misses a small center-crossing overshoot. The historical
entry is preserved, with this analysis supplying the correction.

## Use in simulation and limits of inference

- The 5 ms nominal sample period and measured host timing distribution are
  supported for this acquisition setup. Replay actual request/response times
  where possible to preserve burst correlations; a fixed 2 ms loop delay omits
  observed outliers. This is not a measurement of a future RPi/pi3hat pipeline.
- For an explicitly assumed robustness scenario, a rounded 35 ms **feedback
  gap** covers the largest observed 33.571 ms gap. It is not a certified worst
  case, not a measured one-way command delay, and not evidence that larger gaps
  cannot occur. Do not add all table maxima: the quantities overlap.
- **Superseded 2026-10-25:** the torque-pulse pilot in
  [desk-stand-torque-pulse-timing.md](desk-stand-torque-pulse-timing.md)
  bounds command-to-observation at 7–13 ms; use that figure, not 22–47 ms.
- The 22–47 ms movement-detection range is a position-mode observation at the
  stated thresholds and load. Do not insert it as a pure torque-command delay
  in the pendulum model: that would conflate transport, trajectory generation,
  acceleration, friction and encoder processing.
- Tracking data support approximate angle attainment but do not support a
  blanket claim of settled motion by 340 ms. They also do not validate the
  heavier flywheel or the post-anticogging configuration.
- Existing data therefore establish empirical communication and tracking
  ranges, but leave pure actuator delay unidentified. A repeated Tier-A
  position test would need a verified quiet pre-step hold and enough repeats
  to improve that observation bound. Identifying torque-mode delay requires
  matching excitation and temporal resolution; a camera is not required.
  Remaining scope and decisions are maintained in PNDLM-8.
