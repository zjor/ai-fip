# moteus modes: what we use and what we learned

The moteus controller reports its state as `mode` in every reply (register
0x000; the `mode` column in bench telemetry). This note lists the modes and,
more importantly, the project-specific facts measured on the bench. For the
full protocol see the upstream
[reference](https://github.com/mjbots/moteus/blob/main/docs/reference.md).

Controller: moteus r4.11, firmware and Python library `moteus` as pinned in
`hardware/firmware/moteus-host/`.

## Mode list

Numbers and names from the Python library enum
(`moteus/protocol.py`, class `Mode`). Modes 2–4 are not in that enum; their
meaning is from the upstream reference.

| Mode | Name | Behavior | Entered by |
|---:|---|---|---|
| 0 | stopped | power stage off, no current, rotor free (coasts) | host `set_stop()`, power-on |
| 1 | fault | power stage off after an error | firmware |
| 2 | enabling | gate driver start | firmware, leaving 0 |
| 3 | calibrating | current-sensor offset calibration | firmware, leaving 0 |
| 4 | calibration complete | handover to the commanded mode | firmware |
| 5–8 | PWM / voltage / voltage FOC / voltage DQ | direct voltage control, diagnostics | host |
| 9 | current | direct d/q current command | host `set_current()` |
| **10** | **position** | torque = kp·(position error) + kd·(velocity error) + feedforward | host `set_position()` |
| 11 | timeout | command watchdog expired; behavior set by `servo.timeout_mode` | firmware |
| 12 | zero velocity | hold zero velocity (damping) | host, or `timeout_mode` 12 |
| 13 | stay within | feedforward torque only, position loop at the bounds | host `set_stay_within()` |
| 14 | measure inductance | inductance measurement | host |
| 15 | brake | phases shorted | host, or `timeout_mode` 15 |

Modes are fixed in firmware. Other behaviors are parameter sets of mode 10:

| Behavior | position | velocity | kp_scale | kd_scale | feedforward_torque |
|---|---|---|---|---|---|
| Pure torque (pendulum controller, all torque benches) | NaN | 0 | 0 | 0 | τ |
| Velocity loop (coast-down spin-up) | NaN | v | 0 | > 0 | 0 |
| Position hold | θ | 0 | > 0 | > 0 | 0 |

## Project facts

- **Torque is commanded through mode 10, not mode 9.** Every bench sends
  `set_position(position=NaN, kp_scale=0, kd_scale=0, ilimit_scale=0,
  feedforward_torque=τ)`, including plans written in amps, which are converted
  with the firmware K_t (`moteus_host/bench/experiments/response_timing.py`,
  `torque_constant.py`). Reason: same command path as the pendulum controller,
  and mode 10 keeps `maximum_torque`, position bounds and velocity limits.
  Cost: the torque passes through the firmware K_t, which is 12.5 % high
  ([fixture spec §10](specs/motor-torque-fixture.md#10-simulator-facing-parameters)).
- **Leaving mode 0 costs about 15 ms.** The firmware passes through modes
  2–3 before any torque appears; at ~183 Hz the host typically sees
  0 → 3 → 10 over two to three samples. The pendulum controller must stay in
  mode 10 while balancing; do not stop between commands
  ([fixture spec §6.1, §9.3](specs/motor-torque-fixture.md#61-rise-time-mode)).
- **Mode 11 is latched.** After the watchdog expires the firmware ignores
  further commands until `set_stop()`
  ([coast-down spec](specs/coast-down.md)). Benches therefore confirm a stop
  after every pulse, and the next trial re-enters mode 10 from 0.
- **`servo.timeout_mode` is 12 on our controller** (zero velocity: a watchdog
  trip brakes the rotor). Benches that must not brake (coast-down,
  torque-speed, response timing) set it to 0 (torque off) as a volatile
  override, read it back, and restore 12 after a confirmed stop; nothing is
  written to flash ([response-timing spec](specs/response-timing.md)).
  The stop reply carries the pre-stop mode, so a stop is confirmed with a
  follow-up query.
- **Reported q-current is not zero with the power stage off.** In mode 0
  (and in mode 11 with `timeout_mode` 0) the controller reports about −1.2 A
  with open phases, likely an uncalibrated offset while PWM is off
  ([log, PNDLM-11 entries of 28.10 and 01.11.2026](../project/log.md)).
  Figure A of the
  [response-delay draft](../articles/2026-11-08-response-delay-draft.md) shows
  the same −1.2 A after the watchdog trip. Do not read current outside an
  active mode.
- **Reported q-current includes the anticogging term in mode 10.** At zero
  torque command the current equals the `motor.cogging_dq_comp` table value at
  the rotor position (up to −1.12…+0.90 A on this motor); during motion it
  changes with the position
  ([fixture spec §9.2–9.3](specs/motor-torque-fixture.md#93-rise-time)).
- **Reported velocity is filtered.** Noise up to ±0.025 rev/s at rest and
  ≈6 ms lag behind position
  ([torque-pulse report](desk-stand-torque-pulse-timing.md)).
