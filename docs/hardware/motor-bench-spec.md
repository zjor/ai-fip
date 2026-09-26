# mj5208 + moteus r4.11 bench characterization

## 1. Purpose

This bench suite turns T-005.A–F into reproducible measurements for the Phase 0
feasibility model and the later honest MuJoCo actuator model. It is not a single
maximum-power test. Work progresses from low-energy checks to mechanically
contained tests, and every run preserves raw commands, responses, configuration
and operator notes.

The required outputs are:

- command-to-motion delay and position/velocity tracking;
- low-speed friction and cogging versus rotor position, before and after moteus
  anti-cogging compensation;
- an independently measured torque constant from a lever arm and scale/load
  cell;
- coast-down loss versus speed with a known attached inertia;
- reachable torque versus speed and bus voltage;
- continuous stall torque versus motor winding temperature;
- explicit uncertainty and provenance for every value transferred into the
  simulation.

## 2. Operator workflow

The suite will expose one `bench` command from
`hardware/firmware/moteus-host`:

```shell
poetry run bench inspect
poetry run bench position-step
poetry run bench friction
poetry run bench coast-down
poetry run bench torque-constant
poetry run bench torque-speed
poetry run bench thermal-hold
poetry run bench analyze logs/bench/<run-id>
```

Each motion command prints the resolved limits and setup requirements, performs
a stopped-state preflight, asks for explicit operator confirmation, records one
run directory, and sends `set_stop()` on normal exit, fault, timeout or
`Ctrl-C`. A physical power disconnect remains mandatory; software stop is not
an emergency stop.

The friction experiment stages at the nearer of −0.5 or +0.5 rev, then measures
one full mechanical revolution to the opposite endpoint and back. Thus both
directional datasets cover identical rotor positions within `[-0.5, +0.5]`;
setup data is retained under a separate phase. Preflight validates the captured
moteus `servopos.position_min/max`, rejects a starting position outside them,
and requires 0.02 rev clearance between each sweep endpoint and either finite
bound. The bench program never rewrites persistent controller configuration.

Tests are introduced in three safety tiers:

| Tier | Experiments | Mechanical requirement |
|---|---|---|
| A | `inspect`, `position-step`, very slow `friction` | Current desk stand; light arrow; clear workspace |
| B | `coast-down`, inertia/acceleration sweeps, moderate-speed tests | Balanced retained rotor/flywheel and a guard; no arrow |
| C | `torque-constant`, `torque-speed`, `thermal-hold` | Purpose-built restraint/load fixture, independent force or temperature instrumentation, supervised run |

The clock is not reused for Tier B or C: its arrow is unsuitable for meaningful
speed, torque or thermal characterization.

## 3. Telemetry record

The target acquisition rate is 200 Hz for motion tests and 20 Hz for slow
thermal tests. Every row in `telemetry.csv` contains values as received or sent;
derived quantities belong in analysis outputs.

### Host and run state

| Field | Unit | Purpose |
|---|---:|---|
| `sample` | count | Detect missing or duplicated rows |
| `host_time_s` | s | Monotonic time from the first command |
| `scheduled_time_s` | s | Intended sample time for jitter analysis |
| `request_time_s`, `response_time_s` | s | CAN round-trip and timing jitter |
| `experiment_phase` | text | Preflight, settle, sweep, coast, cooldown, etc. |

### Commanded values

| Field | Unit |
|---|---:|
| `command_mode` | text |
| `command_position_rev` | rev |
| `command_velocity_rev_s` | rev/s |
| `feedforward_torque_nm` | N·m |
| `maximum_torque_nm` | N·m |
| `velocity_limit_rev_s` | rev/s |
| `accel_limit_rev_s2` | rev/s² |
| `watchdog_timeout_s` | s |

Unused command fields are empty, not silently replaced with zero.

### moteus response

| Field | Unit | Notes |
|---|---:|---|
| `mode`, `fault` | enum | Always recorded; any fault ends motion |
| `position_rev` | rev | Output position |
| `velocity_rev_s` | rev/s | Output velocity |
| `torque_nm` | N·m | moteus estimate, not an independent torque sensor |
| `q_current_a`, `d_current_a` | A | Measured phase-current components |
| `bus_voltage_v` | V | Controller input voltage |
| `electrical_power_w` | W | Positive into motor, negative into DC bus |
| `controller_temperature_c` | °C | r4.11 board temperature |
| `motor_temperature_c` | °C | Valid only with a configured motor thermistor |
| `home_state`, `trajectory_complete` | enum/bool | Encoder reference and trajectory state |

The logger requests floating-point position, velocity and torque plus sufficient
resolution for current, voltage, power and temperature. The exact query format
is stored in run metadata.

### Independent and manual observations

`events.csv` records timestamped facts not supplied by moteus:

- applied force or load-cell reading and lever-arm length;
- external winding/case temperature and ambient temperature;
- physical stop, contact, noise or vibration;
- operator abort and reason;
- configuration changes such as enabling anti-cogging.

If an external sensor later gains a programmatic interface, its samples receive
their own raw file and clock description rather than being disguised as moteus
telemetry.

## 4. Run metadata and derived results

`run.json` is written before motion and finalized after stop. It records:

- run ID, wall-clock start/end and exact CLI invocation;
- experiment name and all resolved limits/setpoints;
- motor, controller, CAN adapter, supply and mechanical-fixture identifiers;
- arrow/flywheel/load configuration, mass and known inertia;
- ambient conditions, cooling state and external instruments;
- Python and `moteus` versions, Git revision and dirty state;
- controller ID, firmware information and query resolution;
- result (`completed`, `aborted`, `fault`, `timeout`) and reason.

`controller-config.txt` is a read-only configuration snapshot captured before
the run. This is essential because PID gains, motor `Kv`, current limits,
position scaling, anti-cogging and thermal settings change the meaning of the
telemetry.

Offline analysis writes `summary.json` and plots, never modifies raw files.
Typical derived signals are RPM, position/velocity error, angular acceleration,
mechanical power $\tau\omega$, estimated copper loss, latency percentiles,
temperature rise and fitted friction/cogging curves. The analysis must label
reported torque as model-derived and use external force data for the torque
constant result.

## 5. Repository and data layout

```text
hardware/firmware/moteus-host/
  pyproject.toml
  bench.toml                    tracked conservative defaults and stop limits
  moteus_host/
    clock.py
    bench/
      cli.py                    command selection and operator interaction
      servo.py                  moteus query format and command cycle
      recorder.py               run metadata, CSV and event writing
      safety.py                 preflight, limits, fault/timeout stop path
      experiments/
        inspect.py
        position_step.py
        friction.py
        coast_down.py
        torque_constant.py
        torque_speed.py
        thermal_hold.py
      analysis/
        summarize.py
        plots.py
  tests/
    test_clock.py
    bench/
  logs/                         ignored; never committed
    bench/
      20260920T143012Z_position-step/
        run.json
        controller-config.txt
        telemetry.csv
        events.csv
        summary.json            generated by `bench analyze`
        friction-map.csv        matched directional position bins
        plots/                  generated PNG/SVG files
```

CSV plus JSON is preferred initially: the data volume is small, files are easy
to inspect, and no dataframe dependency is required in the hardware control
path. A column/schema version in `run.json` allows a later Parquet export
without changing the raw acquisition format.

## 6. Experiment sequence and gates

1. **Inspect and idle baseline.** Confirm identity, firmware, home state, bus
   voltage, zero current, temperatures and configuration snapshot without
   motion.
2. **Position steps.** Reuse the safe clock-scale motion to validate timestamps,
   watchdog behavior, tracking, repeatability and latency.
3. **Slow bidirectional friction/cogging.** Traverse several revolutions in both
   directions at very low speed. Fit repeatable torque versus wrapped position
   and direction. Back up configuration before changing anti-cogging settings.
4. **Independent torque constant.** Use a measured lever arm and scale/load
   cell at several low currents in both directions. moteus-reported torque is a
   comparison channel, not the reference measurement.
5. **Coast-down and acceleration.** Only with a known, retained inertia and
   guard. Estimate velocity-dependent losses and check torque-to-acceleration.
6. **Torque-speed envelope.** Sweep only bus voltages and speeds relevant to
   the 12.0–16.8 V design range. A credible torque-at-speed result requires a
   known inertia or external load; an unloaded speed sweep alone measures only
   the no-load boundary.
7. **Thermal hold.** Requires a motor thermistor or external temperature probe
   and a restraint rated for the commanded torque. Increase current/torque in
   conservative plateaus with configured absolute temperature, temperature-rise,
   temperature-rate, board-temperature, voltage and time stop conditions.
8. **Publish.** Transfer fitted values with uncertainty and run IDs into
   `docs/physics/phase-0-feasibility.md` and `software/sim/phase0.toml`; keep raw
   logs local and uncommitted.

Tier B starts only after the retained rotating assembly and guard exist.
Tier C starts only after its fixture and independent instrumentation are
documented and checked.
