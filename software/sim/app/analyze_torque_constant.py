"""Analyze torque-constant fixture recordings without connecting to hardware.

Physical torque from the compression scale at the measured contact radius is
the reference; moteus-reported torque is a comparison channel. Rise times are
host-sampled: a value equal to the resolution is an upper bound.

Several run directories can be pooled into one analysis (for example the
positive side and the negative side recorded in separate sessions); every
plateau and step is tagged with its run ID and the provenance lists every run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from .analyze_position_step import stats


G = 9.80665
INTERPRETATION = (
    "Physical torque from the compression scale at the measured contact radius is the "
    "reference; moteus-reported torque is a comparison channel. Rise times are "
    "host-sampled; a value equal to the resolution is an upper bound, not a measurement."
)


def torque_from_grams(grams: float, radius_mm: float) -> float:
    return grams / 1000.0 * G * radius_mm / 1000.0


def fit_line(x: list[float], y: list[float]) -> dict:
    n = len(x)
    if n < 3 or len(y) != n:
        raise ValueError("fit requires at least three paired points")
    x_mean, y_mean = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((xi - x_mean) ** 2 for xi in x)
    if sxx == 0.0:
        raise ValueError("fit requires at least two distinct x values")
    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / sxx
    intercept = y_mean - slope * x_mean
    residuals = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
    sse = sum(r * r for r in residuals)
    variance = sse / (n - 2)
    return dict(slope=slope, intercept=intercept,
                slope_se=math.sqrt(variance / sxx),
                intercept_se=math.sqrt(variance * (1.0 / n + x_mean ** 2 / sxx)),
                residual_rms=math.sqrt(sse / n), n=n)


def rise_time(samples: list[dict], t0: float, *, settle_fraction: float = 0.5,
              start_a: float = 0.0) -> dict:
    """Time to 63 % and 95 % of the change from `start_a` (the pre-hold level, 0 if the
    step was applied from rest) to the settled current."""
    times = [s["response_time_s"] for s in samples]
    currents = [s["q_current_a"] for s in samples]
    tail = currents[max(0, int(len(currents) * (1.0 - settle_fraction))):] or currents
    steady = statistics.fmean(tail)
    change = steady - start_a
    spacing = [b - a for a, b in zip(times, times[1:])]
    resolution_ms = round(1000.0 * statistics.median(spacing), 3) if spacing else None
    result = dict(steady_a=steady, start_a=start_a, t63_ms=None, t95_ms=None,
                  resolution_ms=resolution_ms, status="not_reached")
    if change == 0.0:
        return result
    for key, fraction in (("t63_ms", 0.63), ("t95_ms", 0.95)):
        for index, value in enumerate(currents):
            if (value - start_a) * change >= fraction * change * change:
                result[key] = 1000.0 * (times[index] - t0)
                if key == "t95_ms":
                    result["status"] = "within_one_sample" if index == 0 else "resolved"
                break
    return result


def parse_note(note: str) -> dict[str, str]:
    result = {}
    for part in note.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = value.strip()
    return result


REQUIRED_FIELDS = ("request_time_s", "response_time_s", "q_current_a", "d_current_a",
                   "torque_nm", "mode")
FILES = ("run.json", "telemetry.csv", "events.csv", "controller-config.txt")


def _float(value):
    return float(value) if value not in (None, "") else None


def load_run(run: Path) -> tuple[dict, list[dict], list[dict]]:
    metadata = json.loads((run / "run.json").read_text())
    if metadata["experiment"] != "torque-constant":
        raise ValueError("requires a torque-constant recording")
    with (run / "telemetry.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != metadata["sample_count"]:
        raise ValueError("recorded sample count mismatch")
    for i, row in enumerate(rows):
        if int(row["sample"]) != i:
            raise ValueError("sample sequence mismatch")
        for field in REQUIRED_FIELDS:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f"non-finite {field}; inspect invalid/aborted recording")
        row["trial_id"] = int(row["trial_id"])
        row["command_q_current_a"] = _float(row.get("command_q_current_a"))
        row["feedforward_torque_nm"] = _float(row.get("feedforward_torque_nm"))
        if row["response_time_s"] < row["request_time_s"]:
            raise ValueError("negative response interval")
        if i and row["request_time_s"] < rows[i - 1]["response_time_s"]:
            raise ValueError("overlapping requests")
    with (run / "events.csv").open(newline="") as source:
        events = list(csv.DictReader(source))
    return metadata, rows, events


def _direction(value: float) -> str:
    return "zero" if value == 0.0 else "positive" if value > 0.0 else "negative"


def _plateaus(rows: list[dict], events: list[dict], radius: dict | None) -> list[dict]:
    readings = {}
    for event in events:
        if event["event"] == "scale_reading":
            readings[int(parse_note(event["note"])["plateau"])] = _float(event["value"])
    plateaus = []
    for trial in sorted({r["trial_id"] for r in rows if r["experiment_phase"] == "plateau"}):
        active = [r for r in rows if r["trial_id"] == trial and r["experiment_phase"] == "plateau"]
        baseline = [r for r in rows if r["trial_id"] == trial and r["experiment_phase"] == "baseline"]
        settled = active[len(active) // 5:] or active
        command = active[0]["command_q_current_a"]
        direction = _direction(command)
        grams = readings.get(trial)
        physical = None
        if grams is not None and direction != "zero" and radius:
            sign = 1.0 if direction == "positive" else -1.0
            physical = sign * torque_from_grams(grams, radius[direction])
        gaps = [b["response_time_s"] - a["response_time_s"] for a, b in zip(active, active[1:])]
        plateaus.append(dict(
            trial_id=trial, command_a=command, direction=direction,
            duration_s=active[-1]["response_time_s"] - active[0]["request_time_s"],
            mean_q_current_a=statistics.fmean(r["q_current_a"] for r in settled),
            std_q_current_a=statistics.pstdev([r["q_current_a"] for r in settled]),
            mean_d_current_a=statistics.fmean(r["d_current_a"] for r in settled),
            mean_reported_torque_nm=statistics.fmean(r["torque_nm"] for r in settled),
            baseline_mean_q_current_a=(statistics.fmean(r["q_current_a"] for r in baseline)
                                       if baseline else None),
            scale_g=grams, physical_torque_nm=physical, samples=len(active),
            max_sample_gap_ms=1000.0 * max(gaps) if gaps else None))
    return plateaus


def _fit_or_none(points: list[tuple[float, float]]) -> dict | None:
    if len(points) < 3:
        return None
    try:
        return fit_line([p[0] for p in points], [p[1] for p in points])
    except ValueError:
        return None


def _fits(plateaus: list[dict]) -> dict:
    fits = {}
    for direction in ("positive", "negative"):
        mine = [p for p in plateaus if p["direction"] == direction]
        read = [p for p in mine if p["physical_torque_nm"] is not None]
        physical = [(p["mean_q_current_a"], p["physical_torque_nm"]) for p in read]
        # Per commanded amp: what the firmware delivers for a torque request, loop shortfall included.
        commanded = [(p["command_a"], p["physical_torque_nm"]) for p in read]
        reported = [(p["mean_q_current_a"], p["mean_reported_torque_nm"]) for p in mine]
        # Selected by commanded magnitude: a zero-current offset must not drop the 1 A plateaus.
        low = [(p["mean_q_current_a"], p["physical_torque_nm"]) for p in read
               if abs(p["command_a"]) <= 1.0]
        # Linear region, per commanded amp: above the ~0.3 A seating dead band and the
        # constant fixture/cogging offset that dominate the readings below 2 A (2026-10-26).
        linear = [(p["command_a"], p["physical_torque_nm"]) for p in read
                  if abs(p["command_a"]) >= 2.0]
        fits[direction] = dict(physical=_fit_or_none(physical), reported=_fit_or_none(reported),
                               low_current=_fit_or_none(low), commanded=_fit_or_none(commanded),
                               linear_region=_fit_or_none(linear))
    return fits


def _acceptance(plateaus: list[dict], fits: dict, status: str) -> dict:
    result = {}
    spread_ok = True
    magnitudes_ok = True
    for direction in ("positive", "negative"):
        read = [p for p in plateaus if p["direction"] == direction and p["physical_torque_nm"] is not None]
        groups: dict[float, list[float]] = {}
        for p in read:
            groups.setdefault(abs(p["command_a"]), []).append(abs(p["physical_torque_nm"]))
        means = [statistics.fmean(groups[k]) for k in sorted(groups)]
        result[f"monotonic_{direction}"] = bool(read) and all(a <= b for a, b in zip(means, means[1:]))
        for values in groups.values():
            mean = statistics.fmean(values)
            if max(abs(v - mean) for v in values) > max(0.005, 0.05 * abs(mean)):
                spread_ok = False
        magnitudes_ok &= len([k for k in groups if k > 0.0]) >= 4
    result["repeat_spread_ok"] = spread_ok
    result["four_magnitudes_per_direction"] = magnitudes_ok
    for key, flag in (("physical", "kt_symmetry_within_5pct"),
                      ("linear_region", "kt_symmetry_within_5pct_linear_region")):
        slopes = [fits[d][key]["slope"] for d in ("positive", "negative") if fits[d][key]]
        result[flag] = (len(slopes) == 2 and abs(abs(slopes[0]) - abs(slopes[1]))
                        <= 0.05 * statistics.fmean(abs(s) for s in slopes))
    result["no_fault_or_stop"] = status == "completed"
    return result


def _rise_times(rows: list[dict]) -> list[dict]:
    results = []
    for trial in sorted({r["trial_id"] for r in rows if r["experiment_phase"] == "step"}):
        step = [r for r in rows if r["trial_id"] == trial and r["experiment_phase"] == "step"]
        prehold = [r for r in rows if r["trial_id"] == trial and r["experiment_phase"] == "prehold"]
        settled = prehold[len(prehold) // 2:]
        start = statistics.fmean(r["q_current_a"] for r in settled) if settled else 0.0
        results.append(dict(trial_id=trial, torque_nm=step[0]["feedforward_torque_nm"],
                            prehold_torque_nm=prehold[0]["feedforward_torque_nm"] if prehold else None,
                            **rise_time(step, step[0]["request_time_s"], start_a=start)))
    return results


def _firmware_kt(run: Path) -> float | None:
    for line in (run / "controller-config.txt").read_text().splitlines():
        if line.startswith("motor.Kv "):
            kv = float(line.split()[1])
            return 8.27 / kv if kv > 0 else None
    return None


def _source(run: Path, metadata: dict, rows: list[dict]) -> dict:
    return dict(run_id=metadata["run_id"], status=metadata["status"], reason=metadata.get("reason"),
                mode=metadata["resolved_config"].get("mode"), sample_count=len(rows),
                resolved_config=metadata["resolved_config"],
                sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                        for name in FILES})


def analyze(run: Path | list[Path]) -> dict:
    runs = [run] if isinstance(run, Path) else list(run)
    if not runs:
        raise ValueError("at least one run directory is required")
    sources, radii_by_run, plateaus, rises, all_rows = [], {}, [], [], []
    for path in runs:
        metadata, rows, events = load_run(path)
        radii = {e["event"]: _float(e["value"]) for e in events
                 if e["event"] in ("radius_positive_mm", "radius_negative_mm")}
        radius = (dict(positive=radii["radius_positive_mm"], negative=radii["radius_negative_mm"])
                  if len(radii) == 2 else None)
        run_id = metadata["run_id"]
        sources.append(_source(path, metadata, rows))
        radii_by_run[run_id] = radius
        plateaus += [dict(run_id=run_id, **p) for p in _plateaus(rows, events, radius)]
        rises += [dict(run_id=run_id, **r) for r in _rise_times(rows)]
        all_rows += rows
    run, rows = runs[0], all_rows
    radius = radii_by_run[sources[0]["run_id"]] if len(runs) == 1 else radii_by_run
    status = "completed" if all(s["status"] == "completed" for s in sources) else "incomplete"
    fits = _fits(plateaus)
    step_rows = [r for r in rows if r["experiment_phase"] == "step"]
    spacing = [b["response_time_s"] - a["response_time_s"]
               for a, b in zip(step_rows, step_rows[1:])
               if a["trial_id"] == b["trial_id"]]

    def mean_abs_slope(key: str) -> float | None:
        slopes = [fits[d][key]["slope"] for d in ("positive", "negative") if fits[d][key]]
        return statistics.fmean(abs(s) for s in slopes) if len(slopes) == 2 else None

    if not rises:
        rate_limit = None
    elif all(r["status"] == "within_one_sample" for r in rises):
        rate_limit = "none within one sample"
    else:
        reached = [r["t95_ms"] for r in rises if r["t95_ms"] is not None]
        rate_limit = max(reached) if reached else "not reached"
    return dict(
        analysis_version=2,
        source=sources[0] if len(sources) == 1 else dict(
            runs=sources, status=status, sample_count=len(rows),
            mode=sorted({s["mode"] for s in sources})),
        radius_mm=radius, plateaus=plateaus, fits=fits,
        acceptance=_acceptance(plateaus, fits, status),
        rise_time=rises,
        simulator=dict(kt_nm_per_a=mean_abs_slope("physical"),
                       kt_per_commanded_a_nm_per_a=mean_abs_slope("commanded"),
                       kt_linear_region_per_commanded_a_nm_per_a=mean_abs_slope("linear_region"),
                       kt_low_current_nm_per_a=mean_abs_slope("low_current"),
                       kt_firmware_nm_per_a=_firmware_kt(run),
                       torque_rate_limit=rate_limit,
                       resolution_ms=round(1000.0 * statistics.median(spacing), 3) if spacing else None),
        round_trip_ms=stats([1000.0 * (r["response_time_s"] - r["request_time_s"]) for r in rows])
        if rows else None,
        interpretation=INTERPRETATION)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="+",
                        help="one or more run directories; several are pooled into one fit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
