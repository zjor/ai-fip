"""Loss-versus-speed, plateau cross-check and effective inertia from a coast-down run.

Design: docs/hardware/specs/coast-down.md §5. Velocity is smoothed with a
centred linear fit over WINDOW_S windows; loss torque is -I * alpha during a
free coast. Everything is stdlib: the 3x3 normal equations of the quadratic
loss model are solved by Gaussian elimination.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .plots import Series, write_line_plot


ANALYSIS_VERSION = 1
WINDOW_S = 0.25
MIN_WINDOW_SAMPLES = 5
BIN_REV_S = 1.0
MIN_FIT_POINTS = 6
TWO_PI = 2.0 * math.pi


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares slope and intercept of ys over xs."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    return slope, mean_y - slope * mean_x


def _windows(times: list[float], values: list[float]) -> list[tuple[float, float, float]]:
    """(mean time, mean value, slope) over WINDOW_S windows stepping by half a window."""
    if not times:
        return []
    result = []
    start = times[0]
    end = times[-1]
    while start <= end:
        stop = start + WINDOW_S
        xs = [t for t in times if start <= t < stop]
        if len(xs) >= MIN_WINDOW_SAMPLES:
            ys = [v for t, v in zip(times, values) if start <= t < stop]
            slope, _ = _linear_fit(xs, ys)
            result.append((sum(xs) / len(xs), sum(ys) / len(ys), slope))
        start += WINDOW_S / 2.0
    return result


def coast_points(times: list[float], velocities: list[float], inertia: float) -> list[tuple[float, float]]:
    """(omega rad/s, loss torque Nm) per window of a free coast; |omega| and tau > 0 when slowing."""
    points = []
    for _, mean_v, slope in _windows(times, velocities):
        if mean_v == 0.0:
            continue
        omega = TWO_PI * abs(mean_v)
        tau = -inertia * TWO_PI * slope * math.copysign(1.0, mean_v)
        points.append((omega, tau))
    return points


def loss_map(points: list[tuple[float, float]], bin_rev_s: float = BIN_REV_S) -> list[dict[str, float | int]]:
    bins: dict[int, list[float]] = defaultdict(list)
    for omega, tau in points:
        bins[int((omega / TWO_PI) // bin_rev_s)].append(tau)
    result = []
    for index in sorted(bins):
        taus = bins[index]
        result.append({
            "speed_rev_s": (index + 0.5) * bin_rev_s,
            "n": len(taus),
            "tau_mean_nm": statistics.fmean(taus),
            "tau_sd_nm": statistics.stdev(taus) if len(taus) > 1 else 0.0,
        })
    return result


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular normal equations")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                factor = a[r][col] / a[col][col]
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    columns = [_solve(matrix, [1.0 if r == c else 0.0 for r in range(n)]) for c in range(n)]
    return [[columns[c][r] for c in range(n)] for r in range(n)]


def fit_loss_model(points: list[tuple[float, float]]) -> dict[str, float | int] | None:
    """tau = a + b*omega + c*omega^2 by least squares, with parameter standard errors."""
    if len(points) < MIN_FIT_POINTS:
        return None
    rows = [[1.0, omega, omega * omega] for omega, _ in points]
    taus = [tau for _, tau in points]
    normal = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    rhs = [sum(r[i] * t for r, t in zip(rows, taus)) for i in range(3)]
    try:
        a, b, c = _solve(normal, rhs)
        covariance = _inverse(normal)
    except ValueError:
        return None
    residuals = [t - (a + b * r[1] + c * r[2]) for r, t in zip(rows, taus)]
    dof = max(len(points) - 3, 1)
    variance = sum(e * e for e in residuals) / dof
    return {
        "a_nm": a, "b_nm_s": b, "c_nm_s2": c,
        "a_se": math.sqrt(max(variance * covariance[0][0], 0.0)),
        "b_se": math.sqrt(max(variance * covariance[1][1], 0.0)),
        "c_se": math.sqrt(max(variance * covariance[2][2], 0.0)),
        "rmse_nm": math.sqrt(sum(e * e for e in residuals) / len(points)),
        "n": len(points),
    }


def _model(fit: dict[str, float | int] | None, omega: float) -> float:
    if fit is None:
        return 0.0
    return float(fit["a_nm"]) + float(fit["b_nm_s"]) * omega + float(fit["c_nm_s2"]) * omega * omega


def plateau_losses(rows_by_trial: dict[int, list[dict[str, str]]], kt: float) -> list[dict[str, float | int]]:
    result = []
    for trial in sorted(rows_by_trial):
        plateau = [r for r in rows_by_trial[trial] if r["experiment_phase"] == "plateau"]
        if not plateau:
            continue
        speeds = [abs(float(r["velocity_rev_s"])) for r in plateau]
        currents = [abs(float(r["q_current_a"])) for r in plateau]
        result.append({
            "trial": trial,
            "speed_rev_s": statistics.fmean(speeds),
            "tau_nm": kt * statistics.fmean(currents),
            "q_current_sd_a": statistics.stdev(currents) if len(currents) > 1 else 0.0,
        })
    return result


def effective_inertia(
    rows_by_trial: dict[int, list[dict[str, str]]],
    kt: float,
    steps: list[dict[str, Any]],
    fit: dict[str, float | int] | None,
) -> dict[str, Any]:
    """I_eff per accel trial from (kt*i - tau_loss(omega)) / alpha over windows of the step phase."""
    torque_by_trial = {int(s["trial"]): float(s["torque_nm"]) for s in steps if s["kind"] == "accel"}
    trials = []
    for trial in sorted(torque_by_trial):
        accel = [r for r in rows_by_trial.get(trial, []) if r["experiment_phase"] == "accel"]
        times = [float(r["host_time_s"]) for r in accel]
        velocities = [float(r["velocity_rev_s"]) for r in accel]
        currents = [float(r["q_current_a"]) for r in accel]
        estimates = []
        for mean_t, mean_v, slope in _windows(times, velocities):
            if slope == 0.0:
                continue
            window = [i for t, i in zip(times, currents) if mean_t - WINDOW_S / 2 <= t <= mean_t + WINDOW_S / 2]
            if not window:
                continue
            net = kt * abs(statistics.fmean(window)) - _model(fit, TWO_PI * abs(mean_v))
            estimates.append(net / (TWO_PI * abs(slope)))
        if estimates:
            trials.append({
                "trial": trial,
                "torque_nm": torque_by_trial[trial],
                "I_eff_kg_m2": statistics.fmean(estimates),
                "windows": len(estimates),
            })
    values = [t["I_eff_kg_m2"] for t in trials]
    return {
        "trials": trials,
        "mean": statistics.fmean(values) if values else None,
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "method": "per 0.25 s window: (kt*|i_q| - fitted loss at omega) / alpha; losses from the coast fit",
    }


def _write_loss_map(path: Path, bins: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=("speed_rev_s", "n", "tau_mean_nm", "tau_sd_nm"))
        writer.writeheader()
        writer.writerows(bins)


def _write_plots(run_directory: Path, rows_by_trial: dict[int, list[dict[str, str]]],
                 points: list[tuple[float, float]], fit: dict[str, float | int] | None) -> list[str]:
    plots = run_directory / "plots"
    plots.mkdir(exist_ok=True)
    palette = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f")
    speed_series = []
    for index, trial in enumerate(sorted(rows_by_trial)):
        coast = [r for r in rows_by_trial[trial] if r["experiment_phase"] == "coast"]
        if coast:
            t0 = float(coast[0]["host_time_s"])
            speed_series.append(Series(
                f"trial {trial}", palette[index % len(palette)],
                [(float(r["host_time_s"]) - t0, 60.0 * abs(float(r["velocity_rev_s"]))) for r in coast],
            ))
    written = []
    if speed_series:
        write_line_plot(plots / "speed-vs-time.svg", title="Coast-down speed",
                        x_label="time since release (s)", y_label="speed (rpm)", series=speed_series)
        written.append("plots/speed-vs-time.svg")
    if points:
        loss_series = [Series("windowed -I*alpha", "#1f77b4",
                              sorted((omega / TWO_PI * 60.0, 1000.0 * tau) for omega, tau in points))]
        if fit is not None:
            omega_max = max(omega for omega, _ in points)
            loss_series.append(Series("fit a+b*w+c*w^2", "#d62728", [
                (omega / TWO_PI * 60.0, 1000.0 * _model(fit, omega))
                for omega in (omega_max * i / 50.0 for i in range(51))
            ]))
        write_line_plot(plots / "loss-vs-speed.svg", title="Loss torque versus speed",
                        x_label="speed (rpm)", y_label="loss torque (mNm)", series=loss_series)
        written.append("plots/loss-vs-speed.svg")
    return written


def analyze(run_directory: Path) -> Path:
    run_directory = run_directory.resolve()
    metadata = json.loads((run_directory / "run.json").read_text())
    if metadata.get("experiment") != "coast-down":
        raise ValueError("coast-down analysis requires a coast-down run")
    if metadata.get("status") != "completed":
        raise ValueError(f"coast-down analysis requires a completed run, got {metadata.get('status')!r}")
    resolved = metadata["resolved_config"]
    inertia = float(resolved["inertia"]["I_z_kg_m2"])
    kt = float(resolved["kt_nm_per_a"])
    steps = list(resolved.get("steps", []))
    with (run_directory / "telemetry.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    rows_by_trial: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("trial_id") not in (None, ""):
            rows_by_trial[int(float(row["trial_id"]))].append(row)

    points: list[tuple[float, float]] = []
    for trial in sorted(rows_by_trial):
        coast = [r for r in rows_by_trial[trial] if r["experiment_phase"] == "coast"]
        points.extend(coast_points(
            [float(r["host_time_s"]) for r in coast],
            [float(r["velocity_rev_s"]) for r in coast],
            inertia,
        ))
    fit = fit_loss_model(points)
    bins = loss_map(points)
    voltages = [float(r["bus_voltage_v"]) for r in rows if r.get("bus_voltage_v")]
    temperatures = [float(r["controller_temperature_c"]) for r in rows if r.get("controller_temperature_c")]
    currents = [abs(float(r["q_current_a"])) for r in rows if r.get("q_current_a")]
    speeds = [abs(float(r["velocity_rev_s"])) for r in rows if r.get("velocity_rev_s")]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "source": {
            "run_id": metadata.get("run_id"),
            "experiment": metadata["experiment"],
            "status": metadata["status"],
            "sample_count": metadata.get("sample_count"),
        },
        "method": {
            "window_s": WINDOW_S,
            "min_window_samples": MIN_WINDOW_SAMPLES,
            "bin_rev_s": BIN_REV_S,
            "loss_torque": "-I * 2*pi * d(velocity)/dt over centred windows of the free coast",
            "fit": "least squares tau = a + b*omega + c*omega^2 (omega in rad/s); None below 6 points",
            "plateau": "kt * mean |q_current| while held at speed (velocity mode)",
        },
        "inertia": resolved["inertia"],
        "kt_nm_per_a": kt,
        "loss_map_bins": len(bins),
        "fit": fit,
        "plateau": plateau_losses(rows_by_trial, kt),
        "effective_inertia": effective_inertia(rows_by_trial, kt, steps, fit),
        "operating_envelope": {
            "bus_voltage_v": {"min": min(voltages), "max": max(voltages)} if voltages else None,
            "controller_temperature_c": {"min": min(temperatures), "max": max(temperatures)} if temperatures else None,
            "peak_abs_q_current_a": max(currents) if currents else None,
            "peak_speed_rpm": 60.0 * max(speeds) if speeds else None,
            "fault_samples": sum(float(r["fault"]) != 0 for r in rows if r.get("fault")),
        },
        "artifacts": {"loss_map": "loss-map.csv", "plots": []},
    }
    _write_loss_map(run_directory / "loss-map.csv", bins)
    summary["artifacts"]["plots"] = _write_plots(run_directory, rows_by_trial, points, fit)
    temporary = run_directory / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output = run_directory / "summary.json"
    temporary.replace(output)
    return output
