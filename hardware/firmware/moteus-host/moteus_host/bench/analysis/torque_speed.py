"""Delivered torque versus speed, knee and no-load speed from one torque-speed run.

Design: docs/hardware/specs/torque-speed.md §5. Per non-overlapping WINDOW_S
window of a torque step, the delivered torque is tau = I_eff * 2*pi * alpha +
tau_loss(2*pi * speed), with I_eff and the pooled loss fit taken from the
rotating-losses record. The nominal torque command is never used as a torque
value (the firmware's K_t differs from the measured one by about 14 %); it
appears in the outputs only as a label. Windows whose mean bus voltage sags
below bus_v - sag_threshold_v are flagged, kept in the table and excluded from
the knee and the no-load slope. The low-speed reference is fitted over the raw
rows of the band (rows with a sagging bus excluded), not over windows.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .coast_down import ANALYZABLE_STATUS, TWO_PI, WINDOW_S, MIN_WINDOW_SAMPLES, _linear_fit, _model, _windows
from .plots import Series, write_line_plot


ANALYSIS_VERSION = 1
KNEE_FRACTION = 0.9
BAND_FRACTION = 0.5
MIN_TAPER_WINDOWS = 3
CSV_FIELDS = ("trial", "torque_cmd_nm", "direction", "time_s", "speed_rev_s", "alpha_rev_s2",
              "tau_nm", "tau_se_nm", "q_current_a", "bus_v", "supply_limited")
PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf")
FLAGGED_COLOR = "#b0b0b0"


def _value(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    return float(value)


def delivered_torque_windows(rows: list[dict], *, inertia: float, inertia_rel_sd: float, loss_fit: dict,
                             bus_v: float, sag_threshold_v: float) -> list[dict]:
    """Windows of one trial's torque-phase rows; time_s is from the first row of the phase."""
    usable = [r for r in rows if _value(r, "host_time_s") is not None and _value(r, "velocity_rev_s") is not None]
    if not usable:
        return []
    times = [float(r["host_time_s"]) for r in usable]
    velocities = [float(r["velocity_rev_s"]) for r in usable]
    rmse = float(loss_fit.get("rmse_nm", 0.0) or 0.0)
    result = []
    for lo, hi, mean_t, mean_v, slope in _windows(times, velocities):
        speed, alpha = abs(mean_v), abs(slope)
        tau = inertia * TWO_PI * alpha + _model(loss_fit, TWO_PI * speed)
        currents = [c for c in (_value(r, "q_current_a") for r in usable[lo:hi]) if c is not None]
        voltages = [v for v in (_value(r, "bus_voltage_v") for r in usable[lo:hi]) if v is not None]
        mean_bus = statistics.fmean(voltages) if voltages else None
        result.append({
            "time_s": mean_t - times[0],
            "speed_rev_s": speed,
            "alpha_rev_s2": alpha,
            "tau_nm": tau,
            "tau_se_nm": math.hypot(tau * inertia_rel_sd, rmse),
            "q_current_a": abs(statistics.fmean(currents)) if currents else None,
            "bus_v": mean_bus,
            "supply_limited": mean_bus is not None and mean_bus < bus_v - sag_threshold_v,
        })
    return result


def low_speed_torque(rows: list[dict], *, min_rev_s: float, expected_noload_rev_s: float, inertia: float,
                     inertia_rel_sd: float, loss_fit: dict, bus_v: float, sag_threshold_v: float) -> dict | None:
    """Delivered torque over the raw torque-phase rows in the low-speed band.

    Rows count when their bus is not sagging (bus >= bus_v - sag_threshold_v, or not reported) and
    min_rev_s < |v| < BAND_FRACTION * expected_noload_rev_s. One line fit of |v| against time over
    those rows gives alpha; tau = I_eff * 2*pi * alpha + loss(2*pi * mean |v|). Raw rows instead of
    WINDOW_S windows, because a high-torque step crosses the band in less than two windows.
    """
    upper = BAND_FRACTION * expected_noload_rev_s
    times, speeds, currents = [], [], []
    for row in rows:
        t, v, bus = _value(row, "host_time_s"), _value(row, "velocity_rev_s"), _value(row, "bus_voltage_v")
        if t is None or v is None or (bus is not None and bus < bus_v - sag_threshold_v):
            continue
        if not min_rev_s < abs(v) < upper:
            continue
        times.append(t)
        speeds.append(abs(v))
        current = _value(row, "q_current_a")
        if current is not None:
            currents.append(abs(current))
    if len(speeds) < 2 * MIN_WINDOW_SAMPLES:
        return None
    alpha, _ = _linear_fit(times, speeds)
    speed = statistics.fmean(speeds)
    tau = inertia * TWO_PI * abs(alpha) + _model(loss_fit, TWO_PI * speed)
    rmse = float(loss_fit.get("rmse_nm", 0.0) or 0.0)
    return {
        "tau_nm": tau,
        "tau_se_nm": math.hypot(tau * inertia_rel_sd, rmse),
        "q_current_a": statistics.fmean(currents) if currents else None,
        "n_rows": len(speeds),
        "speed_rev_s": speed,
    }


def knee(windows: list[dict], reference_tau_nm: float | None, *, after_rev_s: float = 0.0) -> dict | None:
    """Speed where the delivered torque crosses 0.9*reference, after the band (speed >= after_rev_s).

    The first unflagged window below 0.9*reference and the last unflagged window before it at or above
    0.9*reference bracket the crossing; speed is interpolated linearly in tau between them, with half
    their speed distance as the uncertainty and the mean of their q-currents and bus voltages (the
    measured bus, for the voltage-circle fit). With no window at or above
    0.9*reference after the band, the first window below it is used as is, with half its fitted speed
    span (alpha * WINDOW_S / 2) as the uncertainty.
    """
    if reference_tau_nm is None:
        return None
    threshold = KNEE_FRACTION * reference_tau_nm
    above: dict | None = None
    for window in windows:
        if window["supply_limited"] or window["speed_rev_s"] < after_rev_s:
            continue
        if window["tau_nm"] >= threshold:
            above = window
            continue
        if above is None:
            return {
                "speed_rev_s": window["speed_rev_s"],
                "speed_se_rev_s": window["alpha_rev_s2"] * WINDOW_S / 2.0,
                "q_current_a": window["q_current_a"],
                "tau_nm": window["tau_nm"],
                "bus_v": window.get("bus_v"),
            }
        fraction = (above["tau_nm"] - threshold) / (above["tau_nm"] - window["tau_nm"])
        currents = [w["q_current_a"] for w in (above, window) if w["q_current_a"] is not None]
        voltages = [w.get("bus_v") for w in (above, window) if w.get("bus_v") is not None]
        return {
            "speed_rev_s": above["speed_rev_s"] + fraction * (window["speed_rev_s"] - above["speed_rev_s"]),
            "speed_se_rev_s": abs(window["speed_rev_s"] - above["speed_rev_s"]) / 2.0,
            "q_current_a": statistics.fmean(currents) if currents else None,
            "tau_nm": threshold,
            "bus_v": statistics.fmean(voltages) if voltages else None,
        }
    return None


def _taper_slope(windows: list[dict], knee_point: dict | None) -> float | None:
    """d(tau)/d(speed) over the unflagged windows from the knee to the end of the step."""
    if knee_point is None:
        return None
    taper = [w for w in windows if not w["supply_limited"] and w["speed_rev_s"] >= knee_point["speed_rev_s"]]
    if len(taper) < MIN_TAPER_WINDOWS:
        return None
    slope, _ = _linear_fit([w["speed_rev_s"] for w in taper], [w["tau_nm"] for w in taper])
    return slope


def no_load(hold_rows: list[dict], reason: str | None, windows: list[dict], knee: dict | None,
            *, loss_fit: dict | None = None, torque_rows: list[dict] | None = None,
            bus_v: float | None = None, sag_threshold_v: float | None = None) -> dict | None:
    """No-load speed of one step: the hold mean when saturated, a lower bound when velocity-capped.

    With bus_v and sag_threshold_v, hold rows whose bus is below bus_v - sag_threshold_v are
    dropped (the supply folded back and the speed with it); no row left gives None.
    """
    if reason == "velocity-capped":
        speeds = [abs(v) for v in (_value(r, "velocity_rev_s") for r in torque_rows or []) if v is not None]
        if not speeds:
            return None
        return {"speed_rev_s": max(speeds), "lower_bound": True}
    if reason != "saturated":
        return None
    total = len(hold_rows)
    if bus_v is not None and sag_threshold_v is not None:
        hold_rows = [r for r in hold_rows
                     if _value(r, "bus_voltage_v") is None or _value(r, "bus_voltage_v") >= bus_v - sag_threshold_v]
    dropped = total - len(hold_rows)
    speeds = [abs(v) for v in (_value(r, "velocity_rev_s") for r in hold_rows) if v is not None]
    if not speeds:
        return None
    speed = statistics.fmean(speeds)
    currents = [abs(c) for c in (_value(r, "q_current_a") for r in hold_rows) if c is not None]
    voltages = [v for v in (_value(r, "bus_voltage_v") for r in hold_rows) if v is not None]
    slope = _taper_slope(windows, knee)
    bias = None
    if slope:
        bias = _model(loss_fit, TWO_PI * speed) / abs(slope)
    return {
        "speed_rev_s": speed,
        "speed_sd_rev_s": statistics.stdev(speeds) if len(speeds) > 1 else 0.0,
        "lower_bound": False,
        "q_current_a": statistics.fmean(currents) if currents else None,
        "bus_v": statistics.fmean(voltages) if voltages else None,
        "loss_bias_rev_s": bias,
        "hold_rows_dropped": dropped,
    }


def _by_direction(steps: list[dict]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, direction in (("positive", 1), ("negative", -1)):
        speeds = [s["no_load"]["speed_rev_s"] for s in steps
                  if s["direction"] == direction and s["reason"] == "saturated" and s["no_load"] is not None]
        result[name] = {"speed_rev_s": statistics.fmean(speeds), "steps": len(speeds)} if speeds else None
    if result["positive"] and result["negative"]:
        result["difference_rev_s"] = result["positive"]["speed_rev_s"] - result["negative"]["speed_rev_s"]
    else:
        result["difference_rev_s"] = None
    return result


def _write_csv(path: Path, table: list[dict]) -> None:
    with path.open("w", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(table)


def _write_plots(run_directory: Path, analyzed: list[tuple[dict, list[dict], list[dict]]]) -> list[str]:
    """analyzed: (step, windows, trial rows from the torque phase on) per analyzed step."""
    plots = run_directory / "plots"
    plots.mkdir(exist_ok=True)
    torque_series, speed_series = [], []
    for index, (step, windows, rows) in enumerate(analyzed):
        label = f"{float(step['torque_nm']):+g} Nm"
        color = PALETTE[index % len(PALETTE)]
        clean = [(60.0 * w["speed_rev_s"], 1000.0 * w["tau_nm"]) for w in windows if not w["supply_limited"]]
        flagged = [(60.0 * w["speed_rev_s"], 1000.0 * w["tau_nm"]) for w in windows if w["supply_limited"]]
        if clean:
            torque_series.append(Series(label, color, clean))
        if flagged:
            torque_series.append(Series(f"{label} supply-limited", FLAGGED_COLOR, flagged))
        timed = [r for r in rows if _value(r, "host_time_s") is not None and _value(r, "velocity_rev_s") is not None]
        if timed:
            t0 = float(timed[0]["host_time_s"])
            speed_series.append(Series(label, color, [
                (float(r["host_time_s"]) - t0, 60.0 * abs(float(r["velocity_rev_s"]))) for r in timed]))
    written = []
    if torque_series:
        write_line_plot(plots / "torque-vs-speed.svg", title="Delivered torque versus speed",
                        x_label="speed (rpm)", y_label="delivered torque I*alpha + loss (mNm)",
                        series=torque_series)
        written.append("plots/torque-vs-speed.svg")
    if speed_series:
        write_line_plot(plots / "speed-vs-time.svg", title="Torque step, hold and coast",
                        x_label="time since step start (s)", y_label="speed (rpm)", series=speed_series)
        written.append("plots/speed-vs-time.svg")
    return written


def analyze(run_directory: Path) -> Path:
    run_directory = Path(run_directory).resolve()
    metadata = json.loads((run_directory / "run.json").read_text())
    if metadata.get("experiment") != "torque-speed":
        raise ValueError("torque-speed analysis requires a torque-speed run")
    status = metadata.get("status")
    if status not in ANALYZABLE_STATUS:
        raise ValueError(f"torque-speed analysis needs a completed, safety-stopped or interrupted run, got {status!r}")
    resolved = metadata["resolved_config"]
    bus_v = float(resolved["bus_v"])
    records = resolved["records"]
    inertia = float(records["inertia_kg_m2"])
    inertia_rel_sd = float(records["inertia_rel_sd"])
    loss_fit = records["loss_fit"]
    sag_threshold_v = float(resolved["sag_threshold_v"])
    min_rev_s = float(resolved["min_saturation_rev_s"])
    expected_noload_rev_s = float(resolved["expected_noload_rpm_per_v"]) * bus_v / 60.0
    band_upper_rev_s = BAND_FRACTION * expected_noload_rev_s

    with (run_directory / "telemetry.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    rows_by_trial: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("trial_id") not in (None, ""):
            rows_by_trial[int(float(row["trial_id"]))].append(row)

    step_summaries: list[dict[str, Any]] = []
    skipped: list[int] = []
    table: list[dict[str, Any]] = []
    analyzed: list[tuple[dict, list[dict], list[dict]]] = []
    for step in resolved.get("steps", []):
        trial = int(step["trial"])
        reason = step.get("reason")
        if not reason:
            skipped.append(trial)
            continue
        trial_rows = rows_by_trial.get(trial, [])
        torque_rows = [r for r in trial_rows if r["experiment_phase"] == "torque"]
        hold_rows = [r for r in trial_rows if r["experiment_phase"] == "hold"]
        windows = delivered_torque_windows(torque_rows, inertia=inertia, inertia_rel_sd=inertia_rel_sd,
                                           loss_fit=loss_fit, bus_v=bus_v, sag_threshold_v=sag_threshold_v)
        low = low_speed_torque(torque_rows, min_rev_s=min_rev_s, expected_noload_rev_s=expected_noload_rev_s,
                               inertia=inertia, inertia_rel_sd=inertia_rel_sd, loss_fit=loss_fit, bus_v=bus_v,
                               sag_threshold_v=sag_threshold_v)
        knee_point = knee(windows, low["tau_nm"] if low else None, after_rev_s=band_upper_rev_s)
        torque_cmd = float(step["torque_nm"])
        direction = int(step.get("direction", 1 if torque_cmd > 0 else -1))
        step_summaries.append({
            "trial": trial,
            "torque_cmd_nm": torque_cmd,
            "direction": direction,
            "reason": reason,
            "low_speed": low,
            "knee": knee_point,
            "no_load": no_load(hold_rows, reason, windows, knee_point, loss_fit=loss_fit, torque_rows=torque_rows,
                               bus_v=bus_v, sag_threshold_v=sag_threshold_v),
            "windows_flagged": sum(w["supply_limited"] for w in windows),
        })
        table.extend({"trial": trial, "torque_cmd_nm": torque_cmd, "direction": direction, **w} for w in windows)
        analyzed.append((step, windows, [r for r in trial_rows if r["experiment_phase"] in ("torque", "hold", "coast")]))

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
            "analyzed_trials": [s["trial"] for s in step_summaries],
        },
        "method": {
            "window_s": WINDOW_S,
            "min_window_samples": MIN_WINDOW_SAMPLES,
            "delivered_torque": "I_eff * 2*pi * |alpha| + (a + b*omega + c*omega^2) at the window's mean speed; "
                                "the nominal command is a label only",
            "tau_se": "tau * inertia_rel_sd, with the loss fit rmse added in quadrature when recorded",
            "supply_limited": f"window mean bus voltage < bus_v - sag_threshold_v ({bus_v - sag_threshold_v:g} V); "
                              "kept in the table, excluded from the reference, knee and no-load slope",
            "low_speed_band_rev_s": [min_rev_s, band_upper_rev_s],
            "low_speed": "one line fit of |v| over time across the raw torque-phase rows in the band with bus >= "
                         "bus_v - sag_threshold_v; tau at their mean speed; null below 2 * min_window_samples rows",
            "knee": f"speed at tau = {KNEE_FRACTION:g} * low-speed torque, interpolated between the unflagged "
                    f"windows at or above {band_upper_rev_s:g} rev/s that bracket it; uncertainty half their "
                    "speed distance (half the window's span when no window above the threshold precedes it)",
            "no_load": "mean |v| over the hold rows of a saturated step with bus >= bus_v - sag_threshold_v "
                       "(sagging rows dropped and counted); max |v| as a lower bound when velocity-capped; "
                       "loss_bias = loss(omega_noload) / |d tau / d speed| over the taper",
        },
        "bus_v": bus_v,
        "records": records,
        "steps": step_summaries,
        "no_load_by_direction": _by_direction(step_summaries),
        "skipped_trials": skipped,
        "operating_envelope": {
            "bus_voltage_v": {"min": min(voltages), "max": max(voltages)} if voltages else None,
            "controller_temperature_c": {"min": min(temperatures), "max": max(temperatures)} if temperatures else None,
            "peak_abs_q_current_a": max(currents) if currents else None,
            "peak_speed_rpm": 60.0 * max(speeds) if speeds else None,
            "fault_samples": sum(float(r["fault"]) != 0 for r in rows if r.get("fault")),
        },
        "artifacts": {"torque_vs_speed": "torque-vs-speed.csv", "plots": []},
    }
    _write_csv(run_directory / "torque-vs-speed.csv", table)
    summary["artifacts"]["plots"] = _write_plots(run_directory, analyzed)
    temporary = run_directory / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output = run_directory / "summary.json"
    temporary.replace(output)
    return output
