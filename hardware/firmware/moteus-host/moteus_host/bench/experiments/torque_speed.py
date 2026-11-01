"""Tier-B torque-speed envelope: at one bus voltage, feedforward torque steps from rest
until the disc saturates against the voltage limit, then a free coast.

Design: docs/hardware/specs/torque-speed.md. This module plans the steps, loads the
inertia and loss records and decides per telemetry sample when and why a step ends
(StepMonitor), and run() drives one run per bus voltage through tier_b.TierBSession.
The motor never brakes: every deceleration is set_stop() and friction; the hold after a
saturated end keeps the same torque command.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..analysis.coast_down import _linear_fit
from ..config import REPO_ROOT, BenchConfig, SafetyConfig, TorqueSpeedConfig
from ..recorder import RunRecorder
from ..safety import SafetyStop, validate_torque_speed
from .tier_b import CONFIRM_WORD, TierBLimits, TierBSession, torque_arguments

# Float slack for comparisons on sample timestamps (elapsed = i / rate is not exact).
_TIME_EPS_S = 1e-9
# Float slack for comparisons of voltages given with 0.1 V resolution.
_VOLT_EPS_V = 1e-6
DEFAULT_DERATE_REV_S = 2.0


def validate(cfg: TorqueSpeedConfig, limits: SafetyConfig) -> None:
    validate_torque_speed(cfg, limits)


def resolve(cfg: TorqueSpeedConfig, *, torques_nm: tuple[float, ...] | None = None,
            direction: int | None = None) -> TorqueSpeedConfig:
    """Apply command-line overrides; the result still has to pass validate()."""
    if torques_nm is not None:
        cfg = replace(cfg, torques_nm=tuple(float(t) for t in torques_nm))
    if direction is not None:
        cfg = replace(cfg, directions=(int(direction),))
    return cfg


def expected_noload_rev_s(cfg: TorqueSpeedConfig, bus_v: float) -> float:
    """No-load speed from the nominal Kv; sizes the low-speed band and the operator prompt only."""
    return cfg.expected_noload_rpm_per_v * bus_v / 60.0


def describe(cfg: TorqueSpeedConfig, bus_v: float, expected_noload_rev_s: float) -> str:
    torques = ", ".join(f"{t:g}" for t in cfg.torques_nm)
    directions = ", ".join(f"{d:+d}" for d in cfg.directions)
    return (f"Torque-speed at {bus_v:g} V bus: feedforward torque steps of {torques} Nm from rest, "
            f"directions {directions}, zero gains; each step runs until saturated (windowed "
            f"acceleration below {cfg.saturation_alpha_rev_s2:g} rev/s² for {cfg.saturation_settle_s:g} s "
            f"above {cfg.min_saturation_rev_s:g} rev/s), velocity-capped "
            f"({cfg.max_velocity_rev_s:g} rev/s = {60 * cfg.max_velocity_rev_s:g} rpm cap, host stop "
            f"{cfg.cap_margin_rev_s:g} rev/s below it), supply-limited (bus more than "
            f"{cfg.sag_stop_v:g} V below {bus_v:g} V) or {cfg.step_timeout_s:g} s, then free coast. "
            f"Expected no-load speed ≈ {60 * expected_noload_rev_s:g} rpm. "
            f"Torque cap {cfg.max_torque_nm:g} Nm; {cfg.sample_rate_hz:g} Hz requested. The motor "
            "never brakes: every stop is a free coast. Each step waits for the operator's GO.")


def _finite(record: dict[str, Any], path: Path, *keys: str) -> float:
    node: Any = record
    try:
        for key in keys:
            node = node[key]
        value = float(node)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {'.'.join(keys)} missing") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: {'.'.join(keys)} must be finite")
    return value


def load_records(loss_record: Path) -> dict[str, Any]:
    """Effective inertia and pooled loss fit from the rotating-losses measurement record."""
    path = Path(loss_record)
    record = json.loads(path.read_text())
    inertia = _finite(record, path, "inertia", "effective_total_kg_m2")
    if inertia <= 0.0:
        raise ValueError(f"{path}: inertia.effective_total_kg_m2 must be positive")
    loss_fit = {key: _finite(record, path, "loss_model", "pooled", key)
                for key in ("a_nm", "b_nm_s", "c_nm_s2")}
    pooled = record["loss_model"]["pooled"]
    if "rmse_nm" in pooled:
        loss_fit["rmse_nm"] = _finite(record, path, "loss_model", "pooled", "rmse_nm")
    try:
        source = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source = str(path)
    return {"inertia_kg_m2": inertia,
            "inertia_rel_sd": _finite(record, path, "inertia", "effective_rel_sd"),
            "loss_fit": loss_fit, "source": source}


def plan_steps(cfg: TorqueSpeedConfig) -> list[dict[str, Any]]:
    """Torques ascending, each in cfg.directions order; torque_nm carries the sign."""
    steps: list[dict[str, Any]] = []
    for torque in sorted(cfg.torques_nm):
        for direction in cfg.directions:
            steps.append({"trial": len(steps) + 1, "torque_nm": direction * torque,
                          "direction": direction})
    return steps


class StepMonitor:
    """Per-sample end rule of a torque step; usable as phase_loop's done(state, elapsed).

    Rules in order: supply sag, cap margin, timeout, windowed-slope saturation. The first
    reason sticks. A None velocity or voltage gives no information to the rules that need it.
    """

    def __init__(self, cfg: TorqueSpeedConfig, bus_v: float, derate_rev_s: float,
                 window_s: float = 0.25, min_window_samples: int = 5) -> None:
        self.cfg = cfg
        self.bus_v = bus_v
        self.derate_rev_s = derate_rev_s
        self.window_s = window_s
        self.min_window_samples = min_window_samples
        self.reason: str | None = None
        self.alpha_rev_s2: float | None = None
        self._window: deque[tuple[float, float]] = deque()
        self._settle_start: float | None = None

    def _end(self, reason: str) -> bool:
        self.reason = reason
        return True

    def __call__(self, state: dict[str, Any], elapsed: float) -> bool:
        if self.reason is not None:
            return True
        cfg = self.cfg
        bus = state.get("bus_voltage_v")
        v = state.get("velocity_rev_s")
        if bus is not None and bus < self.bus_v - cfg.sag_stop_v:
            return self._end("supply-limited")
        if v is not None and abs(v) >= cfg.max_velocity_rev_s - cfg.cap_margin_rev_s:
            return self._end("velocity-capped")
        if elapsed >= cfg.step_timeout_s:
            return self._end("timeout")
        if v is None:
            return False
        window = self._window
        window.append((elapsed, v))
        while window and window[0][0] < elapsed - self.window_s - _TIME_EPS_S:
            window.popleft()
        if (len(window) < self.min_window_samples
                or window[-1][0] - window[0][0] < 0.9 * self.window_s):
            self._settle_start = None
            return False
        self.alpha_rev_s2, _ = _linear_fit([t for t, _ in window], [x for _, x in window])
        if abs(v) > cfg.min_saturation_rev_s and abs(self.alpha_rev_s2) < cfg.saturation_alpha_rev_s2:
            if self._settle_start is None:
                self._settle_start = elapsed
        else:
            self._settle_start = None
            return False
        if elapsed - self._settle_start >= cfg.saturation_settle_s - _TIME_EPS_S:
            if abs(v) > cfg.max_velocity_rev_s - self.derate_rev_s:
                return self._end("velocity-capped")
            return self._end("saturated")
        return False


def check_bus(cfg: TorqueSpeedConfig, bus_v: float, allow_any_bus: bool = False, *,
              min_bus_voltage_v: float) -> None:
    """--bus must name an entry of bus_voltages_v unless --allow-any-bus; with or without the flag
    its sag stop must sit one 0.1 V telemetry step above the safety floor."""
    if not math.isfinite(bus_v) or bus_v <= 0:
        raise ValueError(f"--bus must be finite and positive, got {bus_v!r}")
    if bus_v - cfg.sag_stop_v + _VOLT_EPS_V < min_bus_voltage_v + 0.1:
        raise ValueError(f"--bus {bus_v:g} V minus sag_stop_v {cfg.sag_stop_v:g} V is {bus_v - cfg.sag_stop_v:.2f} V, "
                         f"not one 0.1 V telemetry step above safety.min_bus_voltage_v {min_bus_voltage_v:g} V")
    if allow_any_bus:
        return
    if not any(abs(bus_v - v) <= _VOLT_EPS_V for v in cfg.bus_voltages_v):
        listed = ", ".join(f"{v:g}" for v in cfg.bus_voltages_v)
        raise ValueError(f"--bus {bus_v:g} V is not one of {listed} V; add --allow-any-bus to run it anyway")


def bus_preflight(state: dict[str, Any], bus_v: float, cfg: TorqueSpeedConfig) -> None:
    """Refuse to start when the measured bus is not the --bus the operator named."""
    measured = state.get("bus_voltage_v")
    tol = cfg.bus_tolerance_v
    if measured is None or not math.isfinite(measured) or abs(measured - bus_v) > tol + _VOLT_EPS_V:
        shown = math.nan if measured is None else measured
        raise SafetyStop(f"bus {shown:.1f} V is not {bus_v:g} ± {tol:g} V; set the supply first")


def ramp_warning(output_root: Path, bus_v: float, cfg: TorqueSpeedConfig) -> str | None:
    """Warn when --bus skips a listed voltage below it that has no completed run yet."""
    completed: list[float] = []
    for path in sorted(Path(output_root).glob("*_torque-speed*/run.json")):
        try:
            record = json.loads(path.read_text())
            if record.get("experiment") == "torque-speed" and record.get("status") == "completed":
                completed.append(float(record["resolved_config"]["bus_v"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    lower = max(completed) if completed else -math.inf
    skipped = [v for v in cfg.bus_voltages_v
               if lower + _VOLT_EPS_V < v < bus_v - _VOLT_EPS_V
               and not any(abs(v - done) <= _VOLT_EPS_V for done in completed)]
    if not skipped:
        return None
    names = ", ".join(f"{v:g}" for v in sorted(skipped))
    return (f"Warning: --bus {bus_v:g} V skips {names} V with no completed torque-speed run in "
            f"{output_root}; the ramp goes up one voltage at a time.")


async def run(config: BenchConfig, output_root: Path, controller_id: int, *, bus_v: float,
              allow_any_bus: bool = False, torques_nm: tuple[float, ...] | None = None,
              direction: int | None = None) -> str:
    cfg = resolve(config.torque_speed, torques_nm=torques_nm, direction=direction)
    validate(cfg, config.safety)
    records = load_records(REPO_ROOT / cfg.loss_record)
    steps = plan_steps(cfg)
    warning = ramp_warning(output_root, bus_v, cfg)
    if warning:
        print(warning, flush=True)
    check_bus(cfg, bus_v, allow_any_bus, min_bus_voltage_v=config.safety.min_bus_voltage_v)
    noload_rev_s = expected_noload_rev_s(cfg, bus_v)
    watchdog_s = config.common.watchdog_timeout_s
    resolved = {"controller_id": controller_id, **asdict(cfg), "bus_v": bus_v,
                "allow_any_bus": allow_any_bus, "records": records, "derate_rev_s": None,
                "steps": steps}
    recorder = RunRecorder(output_root=output_root, experiment="torque-speed", resolved_config=resolved)
    limits = TierBLimits(max_velocity_rev_s=cfg.max_velocity_rev_s, max_torque_nm=cfg.max_torque_nm,
                         response_timeout_s=cfg.response_timeout_s, sample_rate_hz=cfg.sample_rate_hz,
                         coast_stop_rev_s=cfg.coast_stop_rev_s, coast_settle_s=cfg.coast_settle_s,
                         coast_timeout_s=cfg.coast_timeout_s)

    def sag(state: dict[str, Any]) -> bool:
        bus = state.get("bus_voltage_v")
        return bus is not None and bus < bus_v - cfg.sag_stop_v

    async with TierBSession(config, controller_id, recorder, limits) as session:
        derate_rev_s = float(session.settings.get("servo.max_velocity_derate", DEFAULT_DERATE_REV_S))
        resolved["derate_rev_s"] = derate_rev_s
        bus_preflight(session.initial_state, bus_v, cfg)
        print(describe(cfg, bus_v, noload_rev_s), flush=True)
        print(f"Inertia {records['inertia_kg_m2']:.6g} kg·m² from {records['source']}", flush=True)
        print(f"Run directory: {recorder.path}", flush=True)
        await session.set_overrides(cfg.max_velocity_rev_s)
        completed_steps = 0
        for index, step in enumerate(steps):
            trial = step["trial"]
            session.mark_step_start()
            await session.stopped()
            # the previous coast ended settled below the threshold; this absorbs a late blip
            await session.wait_for_standstill(phase="preflight", trial=trial)
            prompt = (f"Step {index + 1}/{len(steps)}: torque {step['torque_nm']:+g} Nm from rest until "
                      f"saturation (expected no-load ≈ {60 * noload_rev_s:.0f} rpm at {bus_v:g} V), "
                      f"then free coast. Guard closed? type {CONFIRM_WORD}: ")
            if input(prompt).strip() != CONFIRM_WORD:
                session.event("operator_ended", value=trial)
                break
            session.event("operator_confirmed", value=trial)
            torque = step["torque_nm"]
            arguments = torque_arguments(torque, watchdog_s)
            meta = dict(command_mode="torque", command_velocity_rev_s=0.0, feedforward_torque_nm=torque,
                        maximum_torque_nm=abs(torque), watchdog_timeout_s=watchdog_s,
                        kp_scale=0.0, kd_scale=0.0, ilimit_scale=0.0)
            monitor = StepMonitor(cfg, bus_v, derate_rev_s)
            session.event("step_start", value=trial, unit="Nm", note=f"{torque:+g} Nm")
            await session.phase_loop("torque", trial, arguments, meta, done=monitor,
                                     timeout_s=cfg.step_timeout_s + 1.0,
                                     timeout_message=f"torque step overran its {cfg.step_timeout_s:g} s timeout")
            step["reason"] = monitor.reason
            session.event("step_end", value=trial, note=monitor.reason)
            if monitor.reason == "saturated":
                session.event("hold_start", value=trial)
                held = await session.phase_loop("hold", trial, arguments, meta,
                                                done=lambda s, e: e >= cfg.hold_s or sag(s),
                                                timeout_s=cfg.hold_s + 1.0, timeout_message="hold overran")
                if sag(held):
                    session.event("hold_end", value=trial, note="supply-limited")
            await session.coast(trial)
            completed_steps += 1
            session.event("step_complete", value=trial)
        if completed_steps:
            session.finish("completed")
        else:
            session.finish("aborted", "operator did not confirm the first step")
    return str(recorder.path)
