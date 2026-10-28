"""Loss-versus-speed, plateau cross-check and effective inertia from a coast-down run.

Design: docs/hardware/specs/coast-down.md §5. Velocity is smoothed with a
linear fit over non-overlapping WINDOW_S windows; during a free coast the loss
torque is -I * alpha. The rotating inertia is the disc plus the motor rotor,
which the disc record does not know: when acceleration trials exist, the
total is recovered without a loss model from matched speeds,
I = kt*i / (alpha_accel + |alpha_coast|), and that value scales the loss
map. Everything is stdlib: the 3x3 normal equations of the quadratic loss
model are solved by Gaussian elimination.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
from typing import Any

from .plots import Series, write_line_plot


ANALYSIS_VERSION = 2
WINDOW_S = 0.25
MIN_WINDOW_SAMPLES = 5
BIN_REV_S = 1.0
MIN_FIT_POINTS = 6
ANALYZABLE_STATUS = ("completed", "safety_stop", "interrupted")
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


def _windows(times: list[float], values: list[float]) -> list[tuple[int, int, float, float, float]]:
    """Non-overlapping WINDOW_S windows as (lo, hi, mean time, mean value, slope); hi exclusive."""
    result = []
    lo = 0
    n = len(times)
    while lo < n:
        stop = times[lo] + WINDOW_S
        hi = lo
        while hi < n and times[hi] < stop:
            hi += 1
        if hi - lo >= MIN_WINDOW_SAMPLES:
            xs, ys = times[lo:hi], values[lo:hi]
            slope, _ = _linear_fit(xs, ys)
            result.append((lo, hi, sum(xs) / len(xs), sum(ys) / len(ys), slope))
        lo = hi if hi > lo else lo + 1
    return result


def coast_slopes(times: list[float], velocities: list[float]) -> list[tuple[float, float]]:
    """(|speed| rev/s, |slope| rev/s^2) per window of a free coast: pure kinematics, no inertia."""
    return [(abs(mean_v), abs(slope)) for _, _, _, mean_v, slope in _windows(times, velocities)
            if mean_v != 0.0]


def coast_points(times: list[float], velocities: list[float], inertia: float) -> list[tuple[float, float]]:
    """(omega rad/s, loss torque Nm) per window of a free coast; tau > 0 when slowing."""
    return [(TWO_PI * speed, inertia * TWO_PI * slope) for speed, slope in coast_slopes(times, velocities)]


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
    """tau = a + b*omega + c*omega^2 by least squares, with statistical parameter standard errors."""
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


class _AlphaMap:
    """Coast deceleration versus speed, binned and linearly interpolated between bin centres."""

    def __init__(self, slopes: list[tuple[float, float]]):
        bins: dict[int, list[float]] = defaultdict(list)
        for speed, slope in slopes:
            bins[int(speed // BIN_REV_S)].append(slope)
        self.speeds = [(index + 0.5) * BIN_REV_S for index in sorted(bins)]
        self.alphas = [statistics.fmean(bins[index]) for index in sorted(bins)]

    def __call__(self, speed: float) -> float | None:
        if not self.speeds or speed < self.speeds[0] - BIN_REV_S or speed > self.speeds[-1] + BIN_REV_S:
            return None
        i = bisect_left(self.speeds, speed)
        if i == 0:
            return self.alphas[0]
        if i == len(self.speeds):
            return self.alphas[-1]
        s0, s1 = self.speeds[i - 1], self.speeds[i]
        f = (speed - s0) / (s1 - s0)
        return self.alphas[i - 1] * (1.0 - f) + self.alphas[i] * f


def plateau_losses(rows_by_trial: dict[int, list[dict[str, str]]], kt: float,
                   fit: dict[str, float | int] | None = None) -> list[dict[str, float | int | None]]:
    result = []
    for trial in sorted(rows_by_trial):
        plateau = [r for r in rows_by_trial[trial] if r["experiment_phase"] == "plateau"]
        if not plateau:
            continue
        speeds = [abs(float(r["velocity_rev_s"])) for r in plateau]
        currents = [abs(float(r["q_current_a"])) for r in plateau]
        speed = statistics.fmean(speeds)
        result.append({
            "trial": trial,
            "speed_rev_s": speed,
            "tau_nm": kt * statistics.fmean(currents),
            "q_current_sd_a": statistics.stdev(currents) if len(currents) > 1 else 0.0,
            "coast_tau_nm": _model(fit, TWO_PI * speed) if fit is not None else None,
        })
    return result


def effective_inertia(rows_by_trial: dict[int, list[dict[str, str]]], kt: float,
                      steps: list[dict[str, Any]], coast_alpha: _AlphaMap) -> dict[str, Any]:
    """I per accel trial from matched speeds: kt*|i| / (2*pi*(|alpha_accel| + |alpha_coast|))."""
    torque_by_trial = {int(s["trial"]): float(s["torque_nm"]) for s in steps if s["kind"] == "accel"}
    trials = []
    for trial in sorted(torque_by_trial):
        if trial not in rows_by_trial:
            continue
        accel = [r for r in rows_by_trial[trial] if r["experiment_phase"] == "accel"]
        times = [float(r["host_time_s"]) for r in accel]
        velocities = [float(r["velocity_rev_s"]) for r in accel]
        currents = [float(r["q_current_a"]) for r in accel]
        estimates = []
        for lo, hi, _, mean_v, slope in _windows(times, velocities):
            alpha_coast = coast_alpha(abs(mean_v))
            if slope == 0.0 or alpha_coast is None:
                continue
            current = abs(statistics.fmean(currents[lo:hi]))
            estimates.append(kt * current / (TWO_PI * (abs(slope) + alpha_coast)))
        if estimates:
            trials.append({
                "trial": trial,
                "torque_nm": torque_by_trial[trial],
                "I_eff_kg_m2": statistics.fmean(estimates),
                "windows": len(estimates),
            })
    values = [t["I_eff_kg_m2"] for t in trials]
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "trials": trials,
        "mean": mean,
        "sd": sd,
        "rel_sd": (sd / mean) if mean else None,
        "method": "per 0.25 s window of the torque step: kt*|i_q| / (2*pi*(|alpha_accel| + |alpha_coast| at the same speed)); "
                  "no loss model needed, the coast at matched speed supplies the losses",
    }


def _completed_trials(run_directory: Path, status: str, trials: list[int]) -> list[int]:
    """Trials whose coast finished: all of them for a completed run, else those with step_complete."""
    events_path = run_directory / "events.csv"
    completed: set[int] = set()
    if events_path.is_file():
        with events_path.open(newline="") as source:
            for event in csv.DictReader(source):
                if event.get("event") == "step_complete" and event.get("value") not in (None, ""):
                    completed.add(int(float(event["value"])))
    if status == "completed" and not completed:
        return trials
    return [t for t in trials if t in completed]


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
    status = metadata.get("status")
    if status not in ANALYZABLE_STATUS:
        raise ValueError(f"coast-down analysis needs a completed, safety-stopped or interrupted run, got {status!r}")
    resolved = metadata["resolved_config"]
    disc_inertia = float(resolved["inertia"]["I_z_kg_m2"])
    kt = float(resolved["kt_nm_per_a"])
    steps = list(resolved.get("steps", []))
    with (run_directory / "telemetry.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    all_rows_by_trial: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("trial_id") not in (None, ""):
            all_rows_by_trial[int(float(row["trial_id"]))].append(row)
    completed = _completed_trials(run_directory, status, sorted(all_rows_by_trial))
    if not completed:
        raise ValueError(f"{status!r} run has no completed step to analyze")
    rows_by_trial = {trial: all_rows_by_trial[trial] for trial in completed}

    slopes: list[tuple[float, float]] = []
    for trial in sorted(rows_by_trial):
        coast = [r for r in rows_by_trial[trial] if r["experiment_phase"] == "coast"]
        slopes.extend(coast_slopes([float(r["host_time_s"]) for r in coast],
                                   [float(r["velocity_rev_s"]) for r in coast]))
    inertia_eff = effective_inertia(rows_by_trial, kt, steps, _AlphaMap(slopes))
    if inertia_eff["mean"]:
        basis, inertia = "effective", float(inertia_eff["mean"])
    else:
        basis, inertia = "disc_only", disc_inertia
    points = [(TWO_PI * speed, inertia * TWO_PI * slope) for speed, slope in slopes]
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
            "status": status,
            "sample_count": metadata.get("sample_count"),
            "completed_trials": completed,
        },
        "method": {
            "window_s": WINDOW_S,
            "min_window_samples": MIN_WINDOW_SAMPLES,
            "bin_rev_s": BIN_REV_S,
            "windows": "non-overlapping, so the fit's standard errors are not inflated by shared samples",
            "loss_torque": "I * 2*pi * |d(velocity)/dt| over windows of the free coast, with I = effective "
                           "(disc + rotor, from the acceleration trials) when available, else the disc record",
            "fit": "least squares tau = a + b*omega + c*omega^2 (omega in rad/s); None below 6 points; "
                   "standard errors are statistical only and exclude the inertia uncertainty (see inertia.effective.rel_sd)",
            "plateau": "kt * mean |q_current| while held at speed (velocity mode), next to the coast fit at that speed",
        },
        "inertia": {
            "disc": resolved["inertia"],
            "effective": {k: inertia_eff[k] for k in ("mean", "sd", "rel_sd")},
            "basis": basis,
            "used_kg_m2": inertia,
            "ratio_eff_to_disc": (float(inertia_eff["mean"]) / disc_inertia) if inertia_eff["mean"] else None,
        },
        "kt_nm_per_a": kt,
        "loss_map_bins": len(bins),
        "fit": fit,
        "plateau": plateau_losses(rows_by_trial, kt, fit),
        "effective_inertia": inertia_eff,
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
