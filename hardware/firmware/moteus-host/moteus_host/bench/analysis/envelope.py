"""Envelope across bus voltages: no-load speed line and the moteus voltage-circle r_max.

Pools the per-run summaries of several torque-speed runs (one lab-supply voltage each).
Physics as in software/sim/app/phase0.py::max_q_current_a:
    v_per_hz = 60 / (sqrt3 * Kv)              phase volts per mechanical rev/s
    V_eff    = 0.5 * V_bus * r_max * (1 - m)
    (v_per_hz f + R i)^2 + (pi p L i f)^2 <= V_eff^2
r_max is degenerate with Kv, so it is reported for two Kv candidates. Its samples are the
saturated no-load points (i_q = the plateau q-current) and the knees, each at its measured bus
voltage (the nominal --bus when none was measured, marked bus_source "nominal"). For the Kv
derived from K_t, the K_t uncertainty (KT_REL_SD) is propagated as kt_term and added to the
scatter standard error in quadrature.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .plots import Series, write_line_plot

MODULATION_MARGIN = 0.15
POLES = 14
KT_TIMES_KV = 8.27
# PNDLM-9 static K_t uncertainty (±3 %), docs/hardware/specs/torque-speed.md §5
KT_REL_SD = 0.03
GROUPS = ("no_load", "knee")


def v_per_hz(kv_rpm_per_v: float) -> float:
    return 60.0 / (math.sqrt(3.0) * kv_rpm_per_v)


def circle_speed_rev_s(v_bus: float, i_q: float, *, kv_rpm_per_v: float, r_max: float,
                       m: float = MODULATION_MARGIN, r_ohm: float, l_h: float, poles: int = POLES) -> float:
    """Mechanical speed (rev/s) where q current i_q reaches the voltage circle."""
    v_eff = 0.5 * v_bus * r_max * (1.0 - m)
    k = v_per_hz(kv_rpm_per_v)
    r_i = r_ohm * abs(i_q)
    omega_l = math.pi * poles * l_h * abs(i_q)
    a = k * k + omega_l * omega_l
    b = 2.0 * k * r_i
    c = r_i * r_i - v_eff * v_eff
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return 0.0
    return max(0.0, (-b + math.sqrt(disc)) / (2.0 * a))


def r_max_for_point(v_bus: float, i_q: float, f_rev_s: float, *, kv_rpm_per_v: float,
                    m: float = MODULATION_MARGIN, r_ohm: float, l_h: float, poles: int = POLES) -> float:
    """r_max that puts (i_q, f) exactly on the voltage circle."""
    i, f = abs(i_q), abs(f_rev_s)
    voltage = math.hypot(v_per_hz(kv_rpm_per_v) * f + r_ohm * i, math.pi * poles * l_h * i * f)
    return 2.0 * voltage / (v_bus * (1.0 - m))


def fit_line(xs: list[float], ys: list[float]) -> dict[str, float | int | None]:
    """Ordinary least squares y = slope x + intercept; standard errors from residuals, n - 2 dof."""
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys differ in length")
    if n < 2:
        raise ValueError("a line fit needs at least two points")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("a line fit needs at least two distinct x values")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    sse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    if n > 2:
        s2 = sse / (n - 2)
        slope_se = math.sqrt(s2 / sxx)
        intercept_se = math.sqrt(s2 * (1.0 / n + mean_x * mean_x / sxx))
    else:
        slope_se = intercept_se = None
    return {"slope": slope, "intercept": intercept, "slope_se": slope_se, "intercept_se": intercept_se,
            "n": n, "rmse": math.sqrt(sse / n)}


def _controller_config(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            values[parts[0]] = parts[1].strip()
    return values


def _load(run_directory: Path) -> dict[str, Any]:
    summary = json.loads((run_directory / "summary.json").read_text())
    experiment = (summary.get("source") or {}).get("experiment")
    if experiment is None and (run_directory / "run.json").exists():
        experiment = json.loads((run_directory / "run.json").read_text()).get("experiment")
    if experiment != "torque-speed":
        raise ValueError(f"{run_directory} is not a torque-speed summary (experiment {experiment!r})")
    return summary


def _r_max_values(samples: list[dict], kv: float, r_ohm: float, l_h: float) -> list[float]:
    return [r_max_for_point(s["v_bus"], s["i_q"], s["f"], kv_rpm_per_v=kv, r_ohm=r_ohm, l_h=l_h) for s in samples]


def _stats(samples: list[dict], kv: float, r_ohm: float, l_h: float, *, kt_rel_sd: float | None) -> dict[str, Any]:
    """r_max over the pooled samples at kv; with kt_rel_sd, kt_term is half the change of the mean
    between kv*(1 + kt_rel_sd) and kv*(1 - kt_rel_sd), added to the scatter se in quadrature."""
    values = _r_max_values(samples, kv, r_ohm, l_h)
    n = len(values)
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if n > 1 else None
    se_scatter = sd / math.sqrt(n) if sd is not None else None
    kt_term = None
    if kt_rel_sd is not None and values:
        high = statistics.fmean(_r_max_values(samples, kv * (1.0 + kt_rel_sd), r_ohm, l_h))
        low = statistics.fmean(_r_max_values(samples, kv * (1.0 - kt_rel_sd), r_ohm, l_h))
        kt_term = abs(high - low) / 2.0
    if se_scatter is None:
        se = kt_term
    else:
        se = math.hypot(se_scatter, kt_term) if kt_term is not None else se_scatter
    by_group = {}
    for group in GROUPS:
        selected = [v for v, s in zip(values, samples) if s["group"] == group]
        by_group[group] = {"mean": statistics.fmean(selected) if selected else None, "n": len(selected)}
    return {"kv": kv, "mean": mean, "sd": sd, "se": se, "se_scatter": se_scatter, "kt_term": kt_term, "n": n,
            "above_one": mean is not None and mean > 1.0, "by_group": by_group}


def _bus(measured: Any, nominal: float) -> tuple[float, str]:
    if measured is None:
        return nominal, "nominal"
    return float(measured), "measured"


def _output_directory(root: Path) -> Path:
    base = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_envelope"
    path, suffix = root / base, 1
    while path.exists():
        suffix += 1
        path = root / f"{base}_{suffix}"
    path.mkdir(parents=True)
    return path


def analyze(run_directories: list[Path], *, output_root: Path | None = None) -> Path:
    if not run_directories:
        raise ValueError("envelope needs at least one run directory")
    run_directories = [Path(p) for p in run_directories]
    first = run_directories[0]
    controller = _controller_config(first / "controller-config.txt")
    r_ohm = float(controller["motor.resistance_ohm"])
    l_h = float(controller["motor.inductance_q_H"])
    kv_firmware = float(controller["motor.Kv"])
    kt = float(json.loads((first / "run.json").read_text())["resolved_config"]["kt_nm_per_a"])
    candidates = {"kv_from_kt": KT_TIMES_KV / kt, "kv_firmware": kv_firmware}

    def r_max(v_bus: float, i_q: float, f: float) -> dict[str, float]:
        return {name: r_max_for_point(v_bus, i_q, f, kv_rpm_per_v=kv, r_ohm=r_ohm, l_h=l_h)
                for name, kv in candidates.items()}

    # bus_v of a point is the measured bus (no_load.bus_v, knee.bus_v), the nominal --bus as fallback;
    # runs[] keeps the nominal as the label
    runs, points, lower_bounds, knees, samples = [], [], [], [], []
    for run_directory in run_directories:
        summary = _load(run_directory)
        nominal = float(summary["bus_v"])
        run_id = summary["source"].get("run_id") or run_directory.name
        runs.append({"run_id": run_id, "bus_v": nominal})
        for step in summary["steps"]:
            base = {"run_id": run_id, "trial": step["trial"], "direction": step["direction"]}
            no_load = step.get("no_load")
            if no_load is not None:
                v_bus, source = _bus(no_load.get("bus_v"), nominal)
                f = abs(no_load["speed_rev_s"])
                entry = {**base, "bus_v": v_bus, "bus_source": source, "speed_rpm": 60.0 * f,
                         "lower_bound": bool(no_load.get("lower_bound"))}
                if not entry["lower_bound"] and no_load.get("q_current_a") is not None:
                    i_q = abs(no_load["q_current_a"])
                    entry.update(q_current_a=i_q, r_max=r_max(v_bus, i_q, f))
                    samples.append({"group": "no_load", "v_bus": v_bus, "i_q": i_q, "f": f})
                (lower_bounds if entry["lower_bound"] else points).append(entry)
            knee = step.get("knee")
            if knee is not None and knee.get("q_current_a") is not None:
                v_bus, source = _bus(knee.get("bus_v"), nominal)
                f, i_q = abs(knee["speed_rev_s"]), abs(knee["q_current_a"])
                knees.append({**base, "bus_v": v_bus, "bus_source": source, "speed_rpm": 60.0 * f,
                              "q_current_a": i_q, "r_max": r_max(v_bus, i_q, f)})
                samples.append({"group": "knee", "v_bus": v_bus, "i_q": i_q, "f": f})

    def line(selected: list[dict]) -> dict[str, Any] | None:
        xs = [p["bus_v"] for p in selected]
        if len(set(xs)) < 2:
            return None
        return fit_line(xs, [p["speed_rpm"] for p in selected])

    result = {
        "schema_version": 1,
        "runs": runs,
        "no_load_line": {
            "units": "rpm over bus V",
            "fit": line(points),
            "by_direction": {name: line([p for p in points if p["direction"] == d])
                             for name, d in (("positive", 1), ("negative", -1))},
            "points": points,
        },
        "lower_bounds": lower_bounds,
        "r_max": {name: _stats(samples, kv, r_ohm, l_h, kt_rel_sd=KT_REL_SD if name == "kv_from_kt" else None)
                  for name, kv in candidates.items()},
        "inputs": {"m": MODULATION_MARGIN, "r_ohm": r_ohm, "l_h": l_h, "poles": POLES, "kt_nm_per_a": kt,
                   "kt_rel_sd": KT_REL_SD},
        "knees": knees,
    }
    directory = _output_directory(Path(output_root) if output_root is not None else first.parent)
    result["artifacts"] = {"plots": _write_plot(directory, result)}
    output = directory / "summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output


def _write_plot(directory: Path, result: dict[str, Any]) -> list[str]:
    series = []
    points = sorted((p["bus_v"], p["speed_rpm"]) for p in result["no_load_line"]["points"])
    if points:
        series.append(Series("no-load", "#2563eb", points))
    fit = result["no_load_line"]["fit"]
    if fit is not None:
        xs = [points[0][0], points[-1][0]]
        series.append(Series(f"line {fit['slope']:.1f} rpm/V", "#111827",
                             [(x, fit["slope"] * x + fit["intercept"]) for x in xs]))
    bounds = sorted((p["bus_v"], p["speed_rpm"]) for p in result["lower_bounds"])
    if bounds:
        series.append(Series("no-load lower bound", "#9ca3af", bounds))
    knees = sorted((k["bus_v"], k["speed_rpm"]) for k in result["knees"])
    if knees:
        series.append(Series("knees", "#dc2626", knees))
    if not series:
        return []
    (directory / "plots").mkdir()
    write_line_plot(directory / "plots" / "envelope.svg", title="Torque-speed envelope",
                    x_label="bus voltage (V)", y_label="speed (rpm)", series=series)
    return ["plots/envelope.svg"]
