"""Summarize a completed bidirectional friction run."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .plots import Series, write_line_plot


ANALYSIS_VERSION = 1
BIN_COUNT = 100


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"telemetry field {field!r} is missing")
    return float(value)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p05": _quantile(values, 0.05),
        "p95": _quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "rms": math.sqrt(statistics.mean(value * value for value in values)),
    }


def _phase_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    positions = [_number(row, "position_rev") for row in rows]
    velocity = [_number(row, "velocity_rev_s") for row in rows]
    torque = [_number(row, "torque_nm") for row in rows]
    return {
        "samples": len(rows),
        "duration_s": _number(rows[-1], "host_time_s")
        - _number(rows[0], "host_time_s"),
        "position_rev": {
            "min": min(positions),
            "max": max(positions),
            "span": max(positions) - min(positions),
        },
        "velocity_rev_s": _stats(velocity),
        "reported_torque_nm": _stats(torque),
        "samples_above_abs_velocity": {
            "0.05_rev_s": sum(abs(value) > 0.05 for value in velocity),
            "0.10_rev_s": sum(abs(value) > 0.10 for value in velocity),
            "0.20_rev_s": sum(abs(value) > 0.20 for value in velocity),
        },
    }


def _parse_controller_value(configuration: str, key: str) -> float | None:
    prefix = f"{key} "
    for line in configuration.splitlines():
        if line.startswith(prefix):
            value = float(line[len(prefix) :])
            return None if math.isnan(value) else value
    return None


def _binned_rows(
    rows: list[dict[str, str]], lower: float, upper: float
) -> list[dict[str, float | int]]:
    width = (upper - lower) / BIN_COUNT
    grouped = {
        phase: [list() for _ in range(BIN_COUNT)]
        for phase in ("forward", "reverse")
    }
    for row in rows:
        phase = row["experiment_phase"]
        if phase not in grouped:
            continue
        position = _number(row, "position_rev")
        if position < lower - width or position > upper + width:
            continue
        index = min(BIN_COUNT - 1, max(0, int((position - lower) / width)))
        grouped[phase][index].append(row)

    result: list[dict[str, float | int]] = []
    for index in range(BIN_COUNT):
        forward = grouped["forward"][index]
        reverse = grouped["reverse"][index]
        if not forward or not reverse:
            raise ValueError(
                f"position bin {index} has no samples in both directions"
            )
        forward_torque = statistics.median(
            _number(row, "torque_nm") for row in forward
        )
        reverse_torque = statistics.median(
            _number(row, "torque_nm") for row in reverse
        )
        result.append(
            {
                "position_rev": lower + (index + 0.5) * width,
                "forward_samples": len(forward),
                "reverse_samples": len(reverse),
                "forward_torque_nm": forward_torque,
                "reverse_torque_nm": reverse_torque,
                "periodic_torque_nm": (forward_torque + reverse_torque) / 2.0,
                "directional_friction_nm": (forward_torque - reverse_torque)
                / 2.0,
                "forward_velocity_median_rev_s": statistics.median(
                    _number(row, "velocity_rev_s") for row in forward
                ),
                "forward_velocity_min_rev_s": min(
                    _number(row, "velocity_rev_s") for row in forward
                ),
                "forward_velocity_max_rev_s": max(
                    _number(row, "velocity_rev_s") for row in forward
                ),
                "reverse_velocity_median_rev_s": statistics.median(
                    _number(row, "velocity_rev_s") for row in reverse
                ),
                "reverse_velocity_min_rev_s": min(
                    _number(row, "velocity_rev_s") for row in reverse
                ),
                "reverse_velocity_max_rev_s": max(
                    _number(row, "velocity_rev_s") for row in reverse
                ),
            }
        )
    return result


def _harmonics(values: list[float], maximum: int = 30) -> list[dict[str, float | int]]:
    mean = statistics.mean(values)
    centered = [value - mean for value in values]
    count = len(centered)
    result = []
    for harmonic in range(1, maximum + 1):
        cosine = 2.0 / count * sum(
            value * math.cos(2.0 * math.pi * harmonic * (index + 0.5) / count)
            for index, value in enumerate(centered)
        )
        sine = 2.0 / count * sum(
            value * math.sin(2.0 * math.pi * harmonic * (index + 0.5) / count)
            for index, value in enumerate(centered)
        )
        result.append(
            {
                "cycles_per_revolution": harmonic,
                "amplitude_nm": math.hypot(cosine, sine),
                "cosine_nm": cosine,
                "sine_nm": sine,
            }
        )
    return sorted(result, key=lambda item: item["amplitude_nm"], reverse=True)


def _write_map(path: Path, bins: list[dict[str, float | int]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)
    temporary.replace(path)


def _pairs(
    bins: list[dict[str, float | int]], field: str
) -> list[tuple[float, float]]:
    return [
        (float(row["position_rev"]), float(row[field]))
        for row in bins
    ]


def _write_plots(run_directory: Path, bins: list[dict[str, float | int]]) -> None:
    plots = run_directory / "plots"
    plots.mkdir(exist_ok=True)
    write_line_plot(
        plots / "torque-vs-position.svg",
        title="Reported torque vs rotor position",
        x_label="Position (rev)",
        y_label="moteus-reported torque (Nm)",
        series=[
            Series("forward median", "#2563eb", _pairs(bins, "forward_torque_nm")),
            Series("reverse median", "#dc2626", _pairs(bins, "reverse_torque_nm")),
        ],
    )
    write_line_plot(
        plots / "friction-decomposition.svg",
        title="Bidirectional torque decomposition",
        x_label="Position (rev)",
        y_label="Estimated torque (Nm)",
        series=[
            Series("position-periodic", "#7c3aed", _pairs(bins, "periodic_torque_nm")),
            Series("directional friction", "#059669", _pairs(bins, "directional_friction_nm")),
        ],
    )
    velocity_series = []
    colors = {
        "forward_velocity_median_rev_s": "#2563eb",
        "forward_velocity_min_rev_s": "#93c5fd",
        "forward_velocity_max_rev_s": "#60a5fa",
        "reverse_velocity_median_rev_s": "#dc2626",
        "reverse_velocity_min_rev_s": "#fca5a5",
        "reverse_velocity_max_rev_s": "#f87171",
    }
    for field, color in colors.items():
        velocity_series.append(Series(field.replace("_", " "), color, _pairs(bins, field)))
    write_line_plot(
        plots / "velocity-vs-position.svg",
        title="Measured velocity and stick-slip envelope",
        x_label="Position (rev)",
        y_label="Velocity (rev/s)",
        series=velocity_series,
    )


def analyze(run_directory: Path) -> Path:
    run_directory = run_directory.resolve()
    metadata_path = run_directory / "run.json"
    telemetry_path = run_directory / "telemetry.csv"
    configuration_path = run_directory / "controller-config.txt"
    for required in (metadata_path, telemetry_path, configuration_path):
        if not required.is_file():
            raise ValueError(f"missing required run file: {required}")

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("experiment") != "friction":
        raise ValueError("friction analysis requires a friction run")
    if metadata.get("status") != "completed":
        raise ValueError(
            f"friction analysis requires a completed run, got {metadata.get('status')!r}"
        )
    with telemetry_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    phases = {
        phase: [row for row in rows if row["experiment_phase"] == phase]
        for phase in ("forward", "reverse")
    }
    if not all(phases.values()):
        raise ValueError("completed friction run lacks forward or reverse samples")

    resolved = metadata["resolved_config"]
    center = float(resolved["sweep_center_rev"])
    travel = float(resolved["travel_rev"])
    lower, upper = center - travel / 2.0, center + travel / 2.0
    bins = _binned_rows(rows, lower, upper)
    periodic = [float(row["periodic_torque_nm"]) for row in bins]
    directional = [float(row["directional_friction_nm"]) for row in bins]
    configuration = configuration_path.read_text()

    all_voltage = [_number(row, "bus_voltage_v") for row in rows]
    all_temperature = [_number(row, "controller_temperature_c") for row in rows]
    all_q_current = [_number(row, "q_current_a") for row in rows]
    all_d_current = [_number(row, "d_current_a") for row in rows]
    round_trip_ms = [
        1000.0 * (_number(row, "response_time_s") - _number(row, "request_time_s"))
        for row in rows
    ]
    lateness_ms = [
        1000.0 * (_number(row, "response_time_s") - _number(row, "scheduled_time_s"))
        for row in rows
    ]
    harmonics = _harmonics(periodic)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "source": {
            "run_id": metadata["run_id"],
            "experiment": metadata["experiment"],
            "status": metadata["status"],
            "sample_count": metadata["sample_count"],
        },
        "method": {
            "position_bins": BIN_COUNT,
            "sweep_range_rev": [lower, upper],
            "bin_statistic": "median",
            "periodic_torque": "(forward + reverse) / 2",
            "directional_friction": "(forward - reverse) / 2",
            "torque_provenance": "moteus model-derived estimate; not an independent torque measurement",
        },
        "phases": {phase: _phase_summary(values) for phase, values in phases.items()},
        "decomposition": {
            "directional_friction_nm": _stats(directional),
            "position_periodic_torque_nm": {
                **_stats(periodic),
                "peak_to_peak": max(periodic) - min(periodic),
            },
            "dominant_spatial_harmonics": harmonics[:10],
        },
        "operating_envelope": {
            "bus_voltage_v": {"min": min(all_voltage), "max": max(all_voltage)},
            "controller_temperature_c": {
                "min": min(all_temperature),
                "max": max(all_temperature),
            },
            "peak_abs_q_current_a": max(abs(value) for value in all_q_current),
            "peak_abs_d_current_a": max(abs(value) for value in all_d_current),
            "fault_samples": sum(_number(row, "fault") != 0 for row in rows),
        },
        "timing": {
            "round_trip_ms": _stats(round_trip_ms),
            "schedule_lateness_ms": _stats(lateness_ms),
        },
        "controller": {
            "motor_poles": _parse_controller_value(configuration, "motor.poles"),
            "cogging_dq_scale": _parse_controller_value(
                configuration, "motor.cogging_dq_scale"
            ),
        },
        "artifacts": {
            "friction_map": "friction-map.csv",
            "plots": [
                "plots/torque-vs-position.svg",
                "plots/friction-decomposition.svg",
                "plots/velocity-vs-position.svg",
            ],
        },
    }
    _write_map(run_directory / "friction-map.csv", bins)
    _write_plots(run_directory, bins)
    temporary = run_directory / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output = run_directory / "summary.json"
    temporary.replace(output)
    return output
