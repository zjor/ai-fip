# moteus host controller

Control loop for the 2026 build: Raspberry Pi + pi3hat (CAN-FD, IMU) commanding a moteus driver with an mjbots mj5208 motor. Python, LQR first, then the ONNX policy from `software/rl/models/`.

See [docs/hardware/hardware.md → Driver: moteus](../../../docs/hardware/hardware.md) and [docs/project/roadmap.md](../../../docs/project/roadmap.md) Phase 3. Current work is tracked in [docs/project/tasks.md](../../../docs/project/tasks.md).

## Minute clock

`minute-clock` is a small bench exercise for learning position control. An arrow
attached to the motor advances 30 degrees every 5 seconds and completes one
revolution per minute. It runs continuously until `Ctrl-C`.

Use Python 3.12+ and the official [`moteus` Python
package](https://pypi.org/project/moteus/). From this directory:

```shell
poetry install
poetry run minute-clock --calibrate
```

With `--calibrate`, the controller first enters stop mode. Rotate the arrow to
12 o'clock by hand and press Enter. The program stores that absolute encoder
angle, modulo one revolution, in `.minute-clock-zero.json`. The file is local
device state and is ignored by Git.

On later runs, omit `--calibrate`. The program loads the stored zero, moves to
the nearest equivalent 12 o'clock position, waits until the arrow settles, and
then starts the five-second ticks. Recalibrate only if the arrow moves relative
to the rotor or if you intentionally want to redefine 12 o'clock.

The program assumes controller ID 1 and lets the `moteus` package auto-detect
the connected `fdcanusb`. Run it first with the motor securely mounted, the
arrow clear of people and objects, and a way to disconnect power immediately.
The initial limits are intentionally modest: 0.10 Nm, 0.25 rev/s and
0.50 rev/s².

Positive position may look clockwise or counterclockwise depending on the side
from which the arrow is viewed and the configured motor direction. If the first
tick goes the wrong way, press `Ctrl-C` and recalibrate with the other sign:

```shell
poetry run minute-clock --calibrate --clockwise-sign -1
```

The chosen clockwise sign is stored alongside the zero position.

Useful options:

```shell
poetry run minute-clock --help
poetry run minute-clock --id 2 --max-torque 0.05
```

The command is refreshed every 20 ms and uses a 100 ms watchdog. Normal exit,
`Ctrl-C`, and Python exceptions all send `set_stop()` before the process exits.
Cut power if the host or CAN adapter itself fails; software stopping is not an
emergency stop.

## Bench characterization

The T-005.A–F bench suite stores raw runs under the ignored `logs/bench/` directory.
Tracked safety and experiment defaults are in `bench.toml`; the measurement
design and later experiments are documented in
[`docs/hardware/motor-bench-spec.md`](../../../docs/hardware/motor-bench-spec.md).

Start with the stopped-state inspection. It does not command motion:

```shell
poetry run bench inspect
```

It records five seconds of telemetry plus the complete controller configuration
and firmware identity. Review the reported voltage, temperature, home state and
fault before continuing.

The first motion experiment performs two cycles of +30°, centre, −30°, centre
at the conservative limits in `bench.toml`:

```shell
poetry run bench position-step
```

The program prints the resolved motion limits and requires typing `MOVE` before
it enables position control. Both commands always attempt `set_stop()` on exit.
Each run contains `run.json`, `controller-config.txt`, `telemetry.csv` and
`events.csv`.

After position steps pass, the low-speed friction/cogging experiment stages at
the nearest endpoint of `[-0.5, +0.5]` rev, sweeps one full revolution to the
opposite endpoint, then returns over the same positions:

```shell
poetry run bench friction
```

It takes about 100 seconds of motion at 0.02 rev/s, caps torque at 0.05 Nm, and
requires typing `SWEEP`. The arrow needs unobstructed 360° clearance. Progress is
reported every five seconds, and experiment-specific hard limits stop motion if
speed exceeds 0.35 rev/s or displacement exceeds 1.10 revolutions. The velocity
margin permits the short stick-slip releases that this experiment measures; it
does not change the commanded 0.02 rev/s trajectory. Setup samples are retained
but excluded from the forward/reverse measurement phases. Preflight reads the
moteus position bounds and verifies that both fixed sweep endpoints fit with the
configured 0.02 rev endpoint margin.

For this bench fixture, the moteus bounds are `[-1.1, +1.1]`:

```text
conf set servopos.position_min -1.1
conf set servopos.position_max 1.1
conf write
```

Restart tview after changing configuration, then close it before running
`bench` so both programs do not compete for the CAN adapter. These remain hard
controller limits; the friction command also independently stops at 1.10 rev
displacement from its recorded origin.

Analyze a completed friction run offline with:

```shell
poetry run bench analyze logs/bench/<run-id>
```

The command leaves raw files unchanged and writes `summary.json`, a
100-position-bin `friction-map.csv`, and standalone SVG plots under `plots/`.
It uses matched forward/reverse medians to estimate direction-dependent
friction and position-periodic torque, and reports the dominant spatial
harmonics. All torque outputs remain labeled as moteus model-derived estimates,
not independent torque measurements.
