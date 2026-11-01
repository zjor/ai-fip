"""Tier-B torque-speed envelope: at one bus voltage, feedforward torque steps from rest
until the disc saturates against the voltage limit, then a free coast.

Design: docs/hardware/specs/torque-speed.md. This module plans the steps, loads the
inertia and loss records and decides per telemetry sample when and why a step ends
(StepMonitor). The motor never brakes: every deceleration is set_stop() and friction.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..analysis.coast_down import _linear_fit
from ..config import REPO_ROOT, SafetyConfig, TorqueSpeedConfig
from ..safety import validate_torque_speed

# Float slack for comparisons on sample timestamps (elapsed = i / rate is not exact).
_TIME_EPS_S = 1e-9


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
