# Encoder/current response timing pilot

Measurement design and implementation for
[PNDLM-8](https://zjor.youtrack.cloud/issue/PNDLM-8). The existing
[position-step analysis](desk-stand-response-timing.md) motivates a quiet
baseline and torque excitation instead of another position trajectory.

## Locations

| Component | Location |
|---|---|
| Hardware command | `hardware/firmware/moteus-host/moteus_host/bench/experiments/response_timing.py` |
| CLI | `poetry run bench response-timing` from `hardware/firmware/moteus-host` |
| Configuration | `[response_timing]` in `hardware/firmware/moteus-host/bench.toml` |
| Raw records | ignored `hardware/firmware/moteus-host/logs/bench/<UTC>_response-timing/` |
| Offline analyzer | `software/sim/app/analyze_response_timing.py` |
| Hardware tests | `hardware/firmware/moteus-host/tests/bench/test_response_timing.py` |
| Analysis tests | `software/sim/tests/test_response_timing_analysis.py` |

## Timing capability findings

The standard moteus register query returns position, velocity and measured
current, but no controller sample timestamp for those readings. The installed
Python SDK is 1.1.1. This pilot uses host monotonic request/response timestamps
and floating-point Q current (the older bench schema requested 0.1 A integer
resolution). Other safety telemetry remains in the response.

Upstream firmware revision `657b5c5` has `servo.emit_debug`, but
[`ISR_MaybeEmitDebug`](https://github.com/mjbots/moteus/blob/657b5c5/fw/bldc_servo.cc)
requires `debug_uart_` and writes each sample directly to that UART. The official
[`plot_highrate.py`](https://github.com/mjbots/moteus/blob/657b5c5/utils/plot_highrate.py)
reads an already captured binary file and constructs time from the sample
index and PWM rate. This is not an on-demand CAN buffer available through the
present adapter. No debug UART capture hardware is assumed or configured.
The saved firmware ABI does not uniquely identify the deployed source revision.

Consequently this pilot measures **host command-to-observation** intervals.
It cannot resolve a sub-millisecond current-loop delay or separate one-way CAN
latency, encoder filtering and mechanics. A nominal 200 Hz poll interval is
5 ms; measured gaps and RTTs, not this nominal number alone, determine the
useful resolution. No additional equipment is needed for these coarse bounds.

## Excitation and controller behavior

The pilot defaults to four alternating pulses (two per direction), ±0.01 Nm,
20 ms each. These are provisional pilot settings, not demonstrated physical
performance or an inertia-derived guarantee on speed. A single nonzero command
sets all position PID scales to zero and uses feedforward torque, following
[moteus torque-control guidance](https://mjbots.github.io/moteus/guides/control-modes/#torque-control).
It has a 20 ms controller watchdog. Subsequent samples are **queries only**;
they do not renew the command. The duration is bounded by the controller
watchdog rather than repeated host torque commands. The exact pulse application
and expiration timestamps are not known at the host.

The saved bench configuration uses `servo.timeout_mode=12` (active zero-velocity
control). Before enabling the pilot, the tool temporarily sets it to 0
(torque off), verifies readback and logs the override. It never executes
`conf write`. After a confirmed stop, it restores the original value and verifies
readback. If stop cannot be confirmed, restoration is skipped, the run is marked
`cleanup_failed`, and the operator is told to disconnect power and inspect the
setting. A killed process may leave the volatile timeout setting at 0 until
reconfigured or power-cycled.

The controller's position bounds must be finite and provide the entire planned
±0.05 rev clearance around the starting position. They are not widened.
Speed >0.8 rev/s or cumulative displacement >0.05 rev triggers a host stop
(amended 2026-10-25 from 0.25 rev/s: a 0.02 Nm × 20 ms pulse reaches 0.62 rev/s
reported, and the reported velocity lags the true one by ≈6 ms).
These are sampled abort thresholds, not hard mechanical limits. Current,
voltage, temperature, fault and non-finite telemetry checks also apply.
The pulse torque cannot exceed 0.02 Nm and duration cannot exceed 30 ms in this
pilot implementation. Response waits time out after 50 ms by default.

There is no automatic return to the original angle: between trials the program
requests zero torque and waits for rest. This avoids adding position-regulator
oscillation to the baseline. If gravity or friction makes the arrow drift, the
quiet gate may fail, which is a valid pilot finding, not a reason to silently
relax the criteria. A physical disconnect remains necessary.

## Quiet gate and acquisition

Before each pulse require a continuous 250 ms window with:

- reported position/torque mode 10;
- absolute velocity ≤0.03 rev/s (amended 2026-10-25 from 0.01: the encoder's
  reported velocity at rest varies by up to 0.025 rev/s);
- total angle span ≤0.0005 rev (0.18°);
- no gap between received samples above 2.5 nominal periods.

Give up after five seconds without a qualifying window. Zero-torque requests
use the same PID scales and watchdog as the pulse, so the quiet baseline is in
the same control mode. Current noise is recorded, not assumed to be zero.
Position quietness does not guarantee quiet current; the analyzer adapts its
current threshold to the measured baseline variation.

Each trial observes for at least 150 ms from the host pulse request and must
observe watchdog timeout mode 11. It then confirms stop. The next trial waits
for its own quiet baseline. Polling does not send catch-up bursts after late
responses. Every telemetry row includes `trial_id`, phase, request/response
times and commands actually sent; query rows do not pretend to resend torque.

`run.json`, `controller-config.txt`, `events.csv` and `telemetry.csv` retain
configuration, software/firmware identity, quiet-gate events, timeout overrides
and restoration, as well as completion or failure status. Original controller
configuration is captured before the volatile override. A lost response can
mean a command was applied without a telemetry row; failed runs must not be
treated as a complete command history.

## Operator pilot

From `hardware/firmware/moteus-host`, first validate without opening a transport:

```sh
poetry run bench response-timing --dry-run
```

For the supervised physical pilot, secure the stand and arrow, clear its travel,
and prepare the power disconnect. Then run:

```sh
poetry run bench response-timing
```

The tool first stops the controller and captures configuration. It prints the
resolved settings and requires typing `PULSE` before enabling torque or changing
the timeout setting. Do not move the arrow manually after this confirmation.
On an abort, inspect the reason and raw recording before repeating; the program
does not automatically increase torque or retry a failed trial.

From `software/sim`, analyze the resulting recording:

```sh
poetry run python -m app.analyze_response_timing \
  ../../hardware/firmware/moteus-host/logs/bench/<run-id> \
  --output ../../hardware/firmware/moteus-host/logs/bench/<run-id>/response-analysis.json
```

## Analysis and acceptance

Use the accepted pre-pulse window for each baseline. For current, angle and
velocity separately, detect two consecutive same-sign departures from the
baseline median exceeding twice its maximum absolute baseline deviation, with
floors of 0.05 A, 0.1° and 0.02 rev/s respectively. The confirming pair cannot
span a gap above 2.5 nominal periods. Current detection only uses samples in
active mode 10, excluding the watchdog torque-off transient. Mechanical motion
can be detected after the pulse ends.

Report the first and confirmation response times, thresholds, signed changes,
active sample count, maximum observation gap, timeout-state observation, run
status, RTT statistics and source SHA-256 hashes. `not_detected` means the
excitation and recording did not resolve a response; it never means zero delay.
Partial runs and large-gap trials require review and are not pooled into a
success statistic. Raw time series remain available for plotting or fitting.

The first four pulses establish whether the setup yields usable observations;
they do not establish a reliable latency distribution. A usable trial needs a
quiet baseline, no faults/limits, observed watchdog cutoff, sufficient active
samples and a repeatable signal above measured noise. Later repetitions must
retain direction, amplitude and starting angle so friction/position effects can
be distinguished. If response is faster than the poll resolution, report a
resolution-limited interval rather than a precise pure dead time. Further
measurement decisions and physical-run results belong in PNDLM-8.
