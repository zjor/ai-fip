"""Offline desk-stand analysis; standard library only, never connects to hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def stats(values):
    ordered = sorted(values)

    def percentile(p):
        index = (len(ordered) - 1) * p
        low = int(index)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (index - low)

    return dict(min=min(values), median=statistics.median(values),
                p95=percentile(.95), p99=percentile(.99), max=max(values))


def first_sustained(flags, count=3):
    return next((i for i in range(len(flags) - count + 1)
                 if all(flags[i:i + count])), None)


def final_entry(flags):
    """Entry after the last violation, or None if the recording ends outside."""
    last_bad = max((i for i, value in enumerate(flags) if not value), default=-1)
    return last_bad + 1 if last_bad + 1 < len(flags) else None


def analyze(run: Path):
    metadata = json.loads((run / 'run.json').read_text())
    if metadata['experiment'] != 'position-step' or metadata['status'] != 'completed':
        raise ValueError('requires a completed position-step run')
    with (run / 'telemetry.csv').open(newline='') as source:
        rows = [{k: (float(v) if v and k not in ('experiment_phase', 'command_mode') else v)
                 for k, v in row.items()} for row in csv.DictReader(source)]
    if len(rows) != metadata['sample_count']:
        raise ValueError('sample count mismatch')
    for i, row in enumerate(rows):
        if row['sample'] != i or row['response_time_s'] < row['request_time_s']:
            raise ValueError('invalid sample sequence or timestamps')
        if i and row['request_time_s'] < rows[i - 1]['response_time_s']:
            raise ValueError('overlapping requests')
    starts = [i for i in range(1, len(rows))
              if rows[i]['command_position_rev'] != rows[i - 1]['command_position_rev']]
    moves = []
    for start, end in zip(starts, starts[1:] + [len(rows)]):
        segment = rows[start:end]
        t0 = segment[0]['request_time_s']
        baseline = [r for r in rows[:start] if r['response_time_s'] >= t0 - .1]
        origin = statistics.median(r['position_rev'] for r in baseline)
        deviation_deg = max(abs(r['position_rev'] - origin) * 360 for r in baseline)
        direction = 1 if segment[0]['command_position_rev'] > rows[start - 1]['command_position_rev'] else -1
        target = segment[0]['command_position_rev']
        elapsed = lambda i: None if i is None else 1000 * (segment[i]['response_time_s'] - t0)
        thresholds = sorted(set([.1, .25, .5, max(.1, 2 * deviation_deg)]))
        detections = []
        for threshold in thresholds:
            flags = [direction * (r['position_rev'] - origin) * 360 > threshold for r in segment]
            index = first_sustained(flags)
            detections.append(dict(
                threshold_deg=threshold,
                exceeds_baseline=threshold > deviation_deg,
                first_of_three_response_ms=elapsed(index),
                confirmation_response_ms=elapsed(None if index is None else index + 2),
                # An observation bracket, NOT a controller-side physical onset interval.
                previous_response_ms=elapsed(None if index is None or index == 0 else index - 1)))
        inside = [abs(r['position_rev'] - target) <= .005 and abs(r['velocity_rev_s']) <= .02
                  for r in segment]
        first = next((i for i, value in enumerate(inside) if value), None)
        final = final_entry(inside)
        tail = [r for r in segment if r['response_time_s'] >= segment[-1]['response_time_s'] - .1]
        moves.append(dict(
            sample=start, phase=segment[0]['experiment_phase'], direction=direction,
            baseline_samples=len(baseline), baseline_max_deviation_deg=deviation_deg,
            baseline_peak_velocity_rev_s=max(abs(r['velocity_rev_s']) for r in baseline),
            adaptive_threshold_deg=max(.1, 2 * deviation_deg), detections=detections,
            first_entry_ms=elapsed(first), final_entry_ms=elapsed(final),
            final_dwell_ms=None if final is None else elapsed(len(segment) - 1) - elapsed(final),
            final_100ms_position_error_deg=stats([(r['position_rev'] - target) * 360 for r in tail]),
            final_100ms_peak_velocity_rev_s=max(abs(r['velocity_rev_s']) for r in tail),
            final_100ms_inside_fraction=sum(abs(r['position_rev'] - target) <= .005 and abs(r['velocity_rev_s']) <= .02 for r in tail) / len(tail),
            peak_velocity_rev_s=max(abs(r['velocity_rev_s']) for r in segment),
            overshoot_deg=max(0, max(direction * (r['position_rev'] - target) * 360 for r in segment))))
    metrics = {
        'round_trip_ms': [1000 * (r['response_time_s'] - r['request_time_s']) for r in rows],
        'request_lateness_ms': [1000 * (r['request_time_s'] - r['scheduled_time_s']) for r in rows],
        'response_lateness_ms': [1000 * (r['response_time_s'] - r['scheduled_time_s']) for r in rows],
        'response_gap_ms': [1000 * (b['response_time_s'] - a['response_time_s']) for a, b in zip(rows, rows[1:])],
    }
    return dict(
        analysis_version=1,
        source=dict(run_id=metadata['run_id'], software=metadata['software'],
                    resolved_config=metadata['resolved_config'], sample_count=len(rows),
                    sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                            for name in ('run.json', 'telemetry.csv', 'events.csv', 'controller-config.txt')}),
        method=dict(baseline_window_s=.1, consecutive_detection_samples=3,
                    adaptive_threshold='max(0.1 degree, 2 * maximum baseline deviation from median)',
                    tracking_position_tolerance_rev=.005, tracking_velocity_tolerance_rev_s=.02,
                    timestamps='host request/response; no controller sample timestamps',
                    percentiles='linear interpolation; empirical, not guaranteed bounds'),
        timing={key: dict(**stats(values), count_above_5ms=sum(v > 5 for v in values),
                          count_above_10ms=sum(v > 10 for v in values)) for key, values in metrics.items()},
        largest_rtt_samples=[dict(sample=int(r['sample']), phase=r['experiment_phase'],
                                  rtt_ms=1000 * (r['response_time_s'] - r['request_time_s']))
                             for r in sorted(rows, key=lambda r: r['response_time_s'] - r['request_time_s'], reverse=True)[:5]],
        fault_samples=sum(r['fault'] != 0 for r in rows), moves=moves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
