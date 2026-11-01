"""Tier-B coast-down: free coast-downs of the retained test disc from a speed ladder,
plus torque-step spin-ups from rest with --accelerate.

Design: docs/hardware/specs/coast-down.md. The session plumbing (connect,
volatile overrides, stop, standstill wait, restore with readback, SIGINT guard)
lives in tier_b.TierBSession; this module plans the steps, prompts the operator
and drives the session. The motor never brakes: every deceleration is
set_stop() and friction.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, BenchConfig, CoastDownConfig, SafetyConfig
from ..recorder import RunRecorder
from ..safety import validate_coast_down
from . import tier_b
from .tier_b import CONFIRM_WORD, FLUX_BRAKE_MIN_VOLTAGE_V, TierBLimits, TierBSession


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
    return tier_b.torque_arguments(torque_nm, watchdog_s)


def spin_up_tolerance(target_rev_s: float, cfg: CoastDownConfig) -> float:
    """How close to the target counts as reached: at least 10 % of it (a pure velocity loop
    settles below the target under load), never more than 30 % so low steps still ramp."""
    target = abs(target_rev_s)
    return min(max(cfg.plateau_tolerance_rev_s, 0.1 * target), 0.3 * target)


async def run(config: BenchConfig, output_root: Path, controller_id: int, *,
              accelerate: bool = False, ladder_rpm: tuple[float, ...] | None = None,
              direction: int | None = None) -> str:
    cfg = resolve(config.coast_down, ladder_rpm=ladder_rpm, direction=direction)
    validate(cfg, config.safety)
    inertia = load_inertia(REPO_ROOT / cfg.inertia_record)
    steps = plan_steps(cfg, accelerate=accelerate)
    watchdog_s = config.common.watchdog_timeout_s
    recorder = RunRecorder(output_root=output_root, experiment="coast-down",
                           resolved_config={"controller_id": controller_id, **asdict(cfg),
                                            "accelerate": accelerate, "inertia": inertia, "steps": steps})
    limits = TierBLimits(max_velocity_rev_s=cfg.max_velocity_rev_s, max_torque_nm=cfg.max_torque_nm,
                         response_timeout_s=cfg.response_timeout_s, sample_rate_hz=cfg.sample_rate_hz,
                         coast_stop_rev_s=cfg.coast_stop_rev_s, coast_settle_s=cfg.coast_settle_s,
                         coast_timeout_s=cfg.coast_timeout_s)
    async with TierBSession(config, controller_id, recorder, limits) as session:
        print(describe(cfg, accelerate=accelerate), flush=True)
        print(f"Inertia {inertia['I_z_kg_m2']:.6g} kg·m² from {inertia['source']}", flush=True)
        print(f"Run directory: {recorder.path}", flush=True)
        await session.set_overrides(cfg.max_velocity_rev_s)
        completed_steps = 0
        for index, step in enumerate(steps):
            trial = step["trial"]
            session.mark_step_start()
            await session.stopped()
            # the previous coast ended settled below the threshold; this absorbs a late blip
            await session.wait_for_standstill(phase="preflight", trial=trial)
            if step["kind"] == "ladder":
                prompt = (f"Step {index + 1}/{len(steps)}: {60 * abs(step['target_rev_s']):g} rpm "
                          f"({step['target_rev_s']:+.2f} rev/s), spin-up at {step['torque_nm']:g} Nm, "
                          f"plateau, then free coast. Guard closed? type {CONFIRM_WORD}: ")
            else:
                prompt = (f"Step {index + 1}/{len(steps)}: torque step {step['torque_nm']:+g} Nm from rest "
                          f"to {60 * abs(step['target_rev_s']):g} rpm, then free coast. Guard closed? "
                          f"type {CONFIRM_WORD}: ")
            if input(prompt).strip() != CONFIRM_WORD:
                session.event("operator_ended", value=trial)
                break
            session.event("operator_confirmed", value=trial)
            if step["kind"] == "ladder":
                target = step["target_rev_s"]
                tolerance = spin_up_tolerance(target, cfg)
                arguments = velocity_arguments(target, cfg, watchdog_s)
                meta = dict(command_mode="velocity", command_velocity_rev_s=target,
                            maximum_torque_nm=cfg.spin_up_torque_nm, velocity_limit_rev_s=cfg.max_velocity_rev_s,
                            accel_limit_rev_s2=cfg.accel_limit_rev_s2, watchdog_timeout_s=watchdog_s)
                await session.phase_loop("spin_up", trial, arguments, meta,
                                         done=lambda s, _: abs(s["velocity_rev_s"] - target) <= tolerance,
                                         timeout_s=cfg.spin_up_timeout_s,
                                         timeout_message=f"spin-up timeout: {60 * abs(target):g} rpm not reached in {cfg.spin_up_timeout_s:g} s")
                await session.phase_loop("plateau", trial, arguments, meta,
                                         done=lambda _, elapsed: elapsed >= cfg.plateau_s,
                                         timeout_s=cfg.plateau_s + 1.0, timeout_message="plateau overran")
            else:
                torque = step["torque_nm"]
                arguments = tier_b.torque_arguments(torque, watchdog_s)
                meta = dict(command_mode="torque", command_velocity_rev_s=0.0, feedforward_torque_nm=torque,
                            maximum_torque_nm=abs(torque), watchdog_timeout_s=watchdog_s,
                            kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0)
                await session.phase_loop("accel", trial, arguments, meta,
                                         done=lambda s, _: abs(s["velocity_rev_s"]) >= abs(step["target_rev_s"]),
                                         timeout_s=cfg.accel_timeout_s,
                                         timeout_message=f"acceleration timeout: {60 * abs(step['target_rev_s']):g} rpm not reached in {cfg.accel_timeout_s:g} s")
            await session.coast(trial)
            completed_steps += 1
            session.event("step_complete", value=trial)
        if completed_steps:
            session.finish("completed")
        else:
            session.finish("aborted", "operator did not confirm the first step")
    return str(recorder.path)
