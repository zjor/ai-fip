"""Tier-B coast-down: free coast-downs of the retained test disc from a speed ladder,
plus torque-step spin-ups from rest with --accelerate.

Design: docs/hardware/specs/coast-down.md. The 24 V bench supply cannot sink
current, so the motor never brakes: every deceleration is set_stop() and
friction. For the run the firmware watchdog is set to torque-off
(servo.timeout_mode 0), the position bounds are opened (NaN),
servo.max_velocity derates torque above the Tier B speed and the flux brake
burns any regeneration in the windings below the supply's limit; all of them
are volatile and restored with a readback in `finally`. A safety stop, a
cancelled task (what asyncio.run does on Ctrl-C) or a lost reply sends the
stop and then keeps querying until the disc is below coast_stop_rev_s, so the
operator is never left with a blind spinning disc; a second Ctrl-C during that
wait is swallowed with a reminder that only a power disconnect stops it sooner.
"""

from __future__ import annotations

import asyncio
import json
import math
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, BenchConfig, CoastDownConfig, SafetyConfig
from ..recorder import RunRecorder
from ..safety import SafetyStop, check_state, validate_coast_down
from ..servo import create_controller, result_to_state, snapshot_controller
from .common import confirm_stopped, wait_until


CONFIRM_WORD = "GO"
CONNECT_TIMEOUT_S = 5.0
OVERRIDES = ("servo.timeout_mode", "servopos.position_min", "servopos.position_max", "servo.max_velocity")
# Regeneration above this bus voltage is dissipated in the windings instead of
# pushed into the 24 V supply, which cannot sink current. Older firmware exposes
# the threshold directly (flux_brake_min_voltage); 2025 firmware derives it as
# servo.max_voltage - servo.flux_brake_margin_voltage.
FLUX_BRAKE_MIN_VOLTAGE_V = 27.0
FLUX_BRAKE_KEYS = ("servo.flux_brake_min_voltage", "servo.flux_brake_margin_voltage")
# stopped, enabling, calibrating, calibration complete, position, timeout
ALLOWED_MODES = (0, 2, 3, 4, 10, 11)
# After an enable the firmware passes through enabling (2), calibrating (3) and
# calibration complete (4) for ~15 ms before position mode; the real controller
# reported mode 3 on the third reply (2026-10-28). Position mode is required once
# this long has passed since the first command of the step.
ENABLE_SETTLE_S = 0.05
STOP_ATTEMPTS = 3
MAX_LOST_REPLIES = 5
PROGRESS_INTERVAL_S = 1.0


def validate(cfg: CoastDownConfig, limits: SafetyConfig) -> None:
    validate_coast_down(cfg, limits)


def resolve(cfg: CoastDownConfig, *, ladder_rpm: tuple[float, ...] | None = None,
            direction: int | None = None) -> CoastDownConfig:
    """Apply command-line overrides; the result still has to pass validate()."""
    if ladder_rpm is not None:
        cfg = replace(cfg, ladder_rpm=tuple(float(v) for v in ladder_rpm))
    if direction is not None:
        cfg = replace(cfg, direction=int(direction))
    return cfg


def describe(cfg: CoastDownConfig, *, accelerate: bool) -> str:
    caps = (f"host stop thresholds: {cfg.max_velocity_rev_s:g} rev/s ({60 * cfg.max_velocity_rev_s:g} rpm; "
            f"servo.max_velocity derates torque above it), {cfg.max_torque_nm:g} Nm; {cfg.sample_rate_hz:g} Hz requested; "
            f"coast ends below {cfg.coast_stop_rev_s:g} rev/s ({60 * cfg.coast_stop_rev_s:g} rpm), "
            f"coast timeout {cfg.coast_timeout_s:g} s. The motor never brakes: every stop is a free "
            "coast. Volatile overrides: servo.timeout_mode 0, position bounds NaN, servo.max_velocity, "
            f"servo.flux_brake_min_voltage {FLUX_BRAKE_MIN_VOLTAGE_V:g} V; restored with readback. "
            "Operator types GO per step.")
    if accelerate:
        torques = ", ".join(f"{t:g}" for t in cfg.accel_torques_nm)
        return (f"Coast-down, acceleration mode: feedforward torque steps of {torques} Nm from rest, "
                f"both directions, zero gains, until {cfg.accel_target_rev_s:g} rev/s "
                f"({60 * cfg.accel_target_rev_s:g} rpm) or {cfg.accel_timeout_s:g} s, then free coast. {caps}")
    ladder = ", ".join(f"{rpm:g}" for rpm in cfg.ladder_rpm)
    return (f"Coast-down, ladder mode: {ladder} rpm, direction {cfg.direction:+d}; spin-up in velocity "
            f"mode at {cfg.spin_up_torque_nm:g} Nm with a {cfg.accel_limit_rev_s2:g} rev/s² ramp "
            f"(timeout {cfg.spin_up_timeout_s:g} s), {cfg.plateau_s:g} s plateau, then free coast. {caps}")


def load_inertia(path: Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text())
    try:
        inertia = float(record["assembly"]["I_z_kg_m2"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: assembly.I_z_kg_m2 missing") from exc
    if not math.isfinite(inertia) or inertia <= 0.0:
        raise ValueError(f"{path}: assembly.I_z_kg_m2 must be finite and positive")
    try:
        source = str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        source = str(path)
    printed = record.get("printed") or {}
    return {"I_z_kg_m2": inertia, "source": source, "printed_mass_g": printed.get("mass_g")}


def plan_steps(cfg: CoastDownConfig, *, accelerate: bool) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if accelerate:
        for torque in cfg.accel_torques_nm:
            for direction in (1, -1):
                steps.append({"trial": len(steps), "kind": "accel",
                              "target_rev_s": direction * cfg.accel_target_rev_s,
                              "torque_nm": direction * torque, "direction": direction})
        return steps
    for rpm in cfg.ladder_rpm:
        steps.append({"trial": len(steps), "kind": "ladder",
                      "target_rev_s": cfg.direction * rpm / 60.0,
                      "torque_nm": cfg.spin_up_torque_nm, "direction": cfg.direction})
    return steps


def velocity_arguments(target_rev_s: float, cfg: CoastDownConfig, watchdog_s: float) -> dict[str, Any]:
    # kp_scale 0: a pure velocity (kd) loop. With position tracking the lag built up during
    # the ramp would be corrected by a braking pulse at the end of it.
    return dict(position=math.nan, velocity=target_rev_s, kp_scale=0.0,
                maximum_torque=cfg.spin_up_torque_nm, velocity_limit=cfg.max_velocity_rev_s,
                accel_limit=cfg.accel_limit_rev_s2, watchdog_timeout=watchdog_s, query=True)


def torque_arguments(torque_nm: float, cfg: CoastDownConfig, watchdog_s: float) -> dict[str, Any]:
    return dict(position=math.nan, velocity=0.0, kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0,
                feedforward_torque=torque_nm, maximum_torque=abs(torque_nm),
                watchdog_timeout=watchdog_s, query=True)


def flux_brake_override(settings: dict[str, str]) -> tuple[str, str]:
    """(key, value) that makes flux braking start at FLUX_BRAKE_MIN_VOLTAGE_V on this firmware."""
    if "servo.flux_brake_min_voltage" in settings:
        return "servo.flux_brake_min_voltage", f"{FLUX_BRAKE_MIN_VOLTAGE_V:g}"
    if "servo.flux_brake_margin_voltage" in settings:
        try:
            max_voltage = float(settings["servo.max_voltage"])
        except (KeyError, ValueError) as exc:
            raise SafetyStop("servo.max_voltage missing; cannot place the flux brake threshold") from exc
        margin = max(max_voltage - FLUX_BRAKE_MIN_VOLTAGE_V, 0.0)
        return "servo.flux_brake_margin_voltage", f"{margin:g}"
    raise SafetyStop("no flux brake key (" + " or ".join(FLUX_BRAKE_KEYS)
                     + ") in the controller configuration; regeneration could reach the supply")


def spin_up_tolerance(target_rev_s: float, cfg: CoastDownConfig) -> float:
    """How close to the target counts as reached: at least 10 % of it (a pure velocity loop
    settles below the target under load), never more than 30 % so low steps still ramp."""
    target = abs(target_rev_s)
    return min(max(cfg.plateau_tolerance_rev_s, 0.1 * target), 0.3 * target)


def _same(reported: str, expected: str) -> bool:
    try:
        a, b = float(reported), float(expected)
    except ValueError:
        return reported.strip() == expected.strip()
    return a == b or (math.isnan(a) and math.isnan(b))


async def run(config: BenchConfig, output_root: Path, controller_id: int, *,
              accelerate: bool = False, ladder_rpm: tuple[float, ...] | None = None,
              direction: int | None = None) -> str:
    cfg = resolve(config.coast_down, ladder_rpm=ladder_rpm, direction=direction)
    validate(cfg, config.safety)
    inertia = load_inertia(REPO_ROOT / cfg.inertia_record)
    steps = plan_steps(cfg, accelerate=accelerate)
    watchdog_s = config.common.watchdog_timeout_s
    period = 1.0 / cfg.sample_rate_hz
    recorder = RunRecorder(output_root=output_root, experiment="coast-down",
                           resolved_config={"controller_id": controller_id, **asdict(cfg),
                                            "accelerate": accelerate, "inertia": inertia, "steps": steps})
    controller = stream = moteus = None
    originals: dict[str, str] = {}
    overrides_set = False
    motion_commanded = False
    step_command_started = None   # perf_counter of the first command of the current step
    status, reason = "failed", None
    start = time.perf_counter()
    sample = 0
    completed_steps = 0

    async def bounded(awaitable, timeout=None):
        return await asyncio.wait_for(awaitable, timeout or cfg.response_timeout_s)

    async def stopped(timeout=None):
        return await confirm_stopped(controller, moteus, timeout_s=timeout or cfg.response_timeout_s)

    def write_row(state, *, phase, trial, command):
        nonlocal sample
        now = time.perf_counter() - start
        recorder.write_sample({"sample": sample, "host_time_s": now, "scheduled_time_s": now,
                               "request_time_s": now, "response_time_s": now,
                               "experiment_phase": phase, "trial_id": trial, **command, **state})
        sample += 1

    async def record(phase, trial, arguments=None, *, meta=None):
        """One query (arguments None) or one command per sample; checks Tier B limits."""
        nonlocal motion_commanded, step_command_started
        if arguments is None:
            result = await bounded(controller.query())
            command = {"command_mode": "query"}
        else:
            motion_commanded = True
            if step_command_started is None:
                step_command_started = time.perf_counter()
            result = await bounded(controller.set_position(**arguments))
            command = meta
        state = result_to_state(result, moteus)
        write_row(state, phase=phase, trial=trial, command=command)
        check_state(state, config.safety, max_abs_velocity_rev_s=cfg.max_velocity_rev_s,
                    max_torque_nm=cfg.max_torque_nm)
        if state["mode"] not in ALLOWED_MODES:
            raise SafetyStop(f"unexpected controller mode {state['mode']}")
        for field in ("velocity_rev_s", "q_current_a", "bus_voltage_v"):
            if not math.isfinite(state[field]):
                raise SafetyStop(f"non-finite {field}")
        return state

    def settled_below_stop():
        """done() for coasts: |v| < coast_stop_rev_s continuously for coast_settle_s."""
        since = None

        def done(state, elapsed):
            nonlocal since
            if abs(state["velocity_rev_s"]) < cfg.coast_stop_rev_s:
                since = elapsed if since is None else since
                return elapsed - since >= cfg.coast_settle_s
            since = None
            return False
        return done

    async def phase_loop(phase, trial, arguments, meta, *, done, timeout_s, timeout_message):
        phase_start = time.perf_counter()
        while True:
            tick = time.perf_counter()
            state = await record(phase, trial, arguments, meta=meta)
            settled = (arguments is not None and step_command_started is not None
                       and time.perf_counter() - step_command_started > ENABLE_SETTLE_S)
            if settled and state["mode"] != 10:
                # mode 11 = firmware watchdog fired; the disc is already coasting and
                # further commands are ignored, so these rows must not pass as data
                raise SafetyStop(f"controller left position mode during {phase}: mode {state['mode']}")
            if done(state, tick - phase_start):
                return state
            if time.perf_counter() - phase_start > timeout_s:
                raise SafetyStop(timeout_message)
            await wait_until(tick + period)

    async def coast(trial):
        await stopped()
        recorder.event(time.perf_counter() - start, "coast_start", value=trial)
        return await phase_loop("coast", trial, None, None, done=settled_below_stop(),
                                timeout_s=cfg.coast_timeout_s,
                                timeout_message=f"coast timeout after {cfg.coast_timeout_s:g} s")

    async def stop_with_retries():
        last = None
        for attempt in range(STOP_ATTEMPTS):
            try:
                return await stopped()
            except (TimeoutError, SafetyStop) as exc:
                last = exc
                await wait_until(time.perf_counter() + period)
        raise SafetyStop(f"stop not confirmed after {STOP_ATTEMPTS} attempts: {last}")

    async def wait_for_standstill(phase="standstill_wait", trial=None):
        """After any stop: keep the operator informed until the disc is slow for coast_settle_s."""
        wait_start = time.perf_counter()
        deadline = wait_start + cfg.coast_timeout_s
        last_print = -math.inf
        lost = 0
        done = settled_below_stop()
        while True:
            tick = time.perf_counter()
            try:
                state = result_to_state(await bounded(controller.query()), moteus)
            except TimeoutError:
                lost += 1
                if lost >= MAX_LOST_REPLIES:
                    raise SafetyStop(f"{lost} replies lost in a row while the disc may be turning")
                await wait_until(tick + period)
                continue
            lost = 0
            write_row(state, phase=phase, trial=trial, command={"command_mode": "query"})
            speed = abs(state["velocity_rev_s"]) if state["velocity_rev_s"] is not None else math.inf
            if done(state, tick - wait_start):
                recorder.event(tick - start, "standstill", value=speed, unit="rev/s")
                return
            if speed >= cfg.coast_stop_rev_s and tick - last_print >= PROGRESS_INTERVAL_S:
                print(f"coasting: {60 * speed:.0f} rpm", flush=True)
                last_print = tick
            if tick > deadline:
                raise SafetyStop("disc still turning after the coast timeout; keep clear until it stops")
            await wait_until(tick + period)

    async def conf(key, value):
        await asyncio.wait_for(stream.command(f"conf set {key} {value}".encode()), 2)
        readback = await asyncio.wait_for(stream.command(f"conf get {key}".encode(), allow_any_response=True), 2)
        if not _same(readback.decode(), str(value)):
            raise SafetyStop(f"{key} readback {readback.decode().strip()!r} != {value!r}")

    try:
        controller, moteus = create_controller(controller_id, timing=True)
        await stopped(timeout=CONNECT_TIMEOUT_S)
        print("Stopped; capturing controller configuration...", flush=True)
        configuration, device = await asyncio.wait_for(snapshot_controller(controller, moteus), 60)
        recorder.attach_controller(configuration, device)
        settings = dict(line.split(maxsplit=1) for line in configuration.splitlines() if " " in line)
        flux_key, flux_value = flux_brake_override(settings)
        try:
            originals = {key: settings[key] for key in OVERRIDES + (flux_key,)}
        except KeyError as exc:
            raise SafetyStop(f"{exc.args[0]} missing from the controller configuration") from exc
        state = await stopped()
        check_state(state, config.safety)
        if not math.isfinite(state["position_rev"]):
            raise SafetyStop("finite rotor position required")
        print(describe(cfg, accelerate=accelerate), flush=True)
        print(f"Inertia {inertia['I_z_kg_m2']:.6g} kg·m² from {inertia['source']}", flush=True)
        print(f"Run directory: {recorder.path}", flush=True)
        stream = moteus.Stream(controller)
        await asyncio.wait_for(stream.flush_read(), 2)
        overrides_set = True   # mark before sending: a lost response must still trigger restoration
        await conf("servo.timeout_mode", 0)
        await conf("servopos.position_min", "nan")
        await conf("servopos.position_max", "nan")
        await conf("servo.max_velocity", f"{cfg.max_velocity_rev_s:g}")
        await conf(flux_key, flux_value)
        recorder.event(time.perf_counter() - start, "overrides_set",
                       note="; ".join(f"{k}={v}" for k, v in originals.items()) + "; volatile only; never conf write")
        for index, step in enumerate(steps):
            trial = step["trial"]
            step_command_started = None
            await stopped()
            # the previous coast ended settled below the threshold; this absorbs a late blip
            await wait_for_standstill(phase="preflight", trial=trial)
            if step["kind"] == "ladder":
                prompt = (f"Step {index + 1}/{len(steps)}: {60 * abs(step['target_rev_s']):g} rpm "
                          f"({step['target_rev_s']:+.2f} rev/s), spin-up at {step['torque_nm']:g} Nm, "
                          f"plateau, then free coast. Guard closed? type {CONFIRM_WORD}: ")
            else:
                prompt = (f"Step {index + 1}/{len(steps)}: torque step {step['torque_nm']:+g} Nm from rest "
                          f"to {60 * abs(step['target_rev_s']):g} rpm, then free coast. Guard closed? "
                          f"type {CONFIRM_WORD}: ")
            if input(prompt).strip() != CONFIRM_WORD:
                recorder.event(time.perf_counter() - start, "operator_ended", value=trial)
                break
            recorder.event(time.perf_counter() - start, "operator_confirmed", value=trial)
            if step["kind"] == "ladder":
                target = step["target_rev_s"]
                tolerance = spin_up_tolerance(target, cfg)
                arguments = velocity_arguments(target, cfg, watchdog_s)
                meta = dict(command_mode="velocity", command_velocity_rev_s=target,
                            maximum_torque_nm=cfg.spin_up_torque_nm, velocity_limit_rev_s=cfg.max_velocity_rev_s,
                            accel_limit_rev_s2=cfg.accel_limit_rev_s2, watchdog_timeout_s=watchdog_s)
                await phase_loop("spin_up", trial, arguments, meta,
                                 done=lambda s, _: abs(s["velocity_rev_s"] - target) <= tolerance,
                                 timeout_s=cfg.spin_up_timeout_s,
                                 timeout_message=f"spin-up timeout: {60 * abs(target):g} rpm not reached in {cfg.spin_up_timeout_s:g} s")
                await phase_loop("plateau", trial, arguments, meta,
                                 done=lambda _, elapsed: elapsed >= cfg.plateau_s,
                                 timeout_s=cfg.plateau_s + 1.0, timeout_message="plateau overran")
            else:
                torque = step["torque_nm"]
                arguments = torque_arguments(torque, cfg, watchdog_s)
                meta = dict(command_mode="torque", command_velocity_rev_s=0.0, feedforward_torque_nm=torque,
                            maximum_torque_nm=abs(torque), watchdog_timeout_s=watchdog_s,
                            kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0)
                await phase_loop("accel", trial, arguments, meta,
                                 done=lambda s, _: abs(s["velocity_rev_s"]) >= abs(step["target_rev_s"]),
                                 timeout_s=cfg.accel_timeout_s,
                                 timeout_message=f"acceleration timeout: {60 * abs(step['target_rev_s']):g} rpm not reached in {cfg.accel_timeout_s:g} s")
            await coast(trial)
            completed_steps += 1
            recorder.event(time.perf_counter() - start, "step_complete", value=trial)
        status = "completed" if completed_steps else "aborted"
        if not completed_steps:
            reason = "operator did not confirm the first step"
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError)):
            status = "interrupted"
        elif isinstance(exc, (SafetyStop, TimeoutError)):
            status = "safety_stop"
        else:
            status = "failed"
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        previous_handler = None
        try:
            try:
                previous_handler = signal.signal(signal.SIGINT, _second_interrupt)
            except ValueError:
                previous_handler = None   # not the main thread (tests); nothing to guard
            if controller is not None:
                await stop_with_retries()
                await wait_for_standstill()
            if overrides_set:
                for key, value in originals.items():
                    await conf(key, value)
                recorder.event(time.perf_counter() - start, "overrides_restored",
                               note="; ".join(f"{k}={v}" for k, v in originals.items()))
        except BaseException as exc:
            status = "cleanup_failed"
            reason = f"{reason or ''}; cleanup: {type(exc).__name__}: {exc}"
            if motion_commanded:
                print("STOP/RESTORE NOT CONFIRMED: disconnect power; keep clear of the disc; "
                      "inspect timeout mode, position bounds, max velocity and flux brake before reuse.",
                      flush=True)
            elif overrides_set:
                print("OVERRIDES NOT RESTORED (motion was never commanded): inspect timeout mode, "
                      "position bounds, max velocity and flux brake before reuse.", flush=True)
            else:
                print("No reply from the controller; motion was never commanded. Check power, "
                      "the CAN cable and the controller ID.", flush=True)
            raise
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            recorder.close(status, reason)
    return str(recorder.path)


def _second_interrupt(signum, frame):
    print("still coasting; the stop was sent. Only a power disconnect stops the disc sooner.", flush=True)
