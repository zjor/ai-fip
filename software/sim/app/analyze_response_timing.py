"""Analyze response-timing CAN recordings without connecting to hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from .analyze_position_step import stats


def detect(rows, field, baseline, floor, t0, max_gap):
    center = statistics.median(baseline)
    threshold = max(floor, 2 * max(abs(v - center) for v in baseline))
    for i in range(len(rows) - 1):
        a, b = rows[i:i + 2]
        if (abs(a[field] - center) > threshold and abs(b[field] - center) > threshold
                and (a[field] - center) * (b[field] - center) > 0
                and b['response_time_s'] - a['response_time_s'] <= max_gap):
            return dict(status='detected', baseline=center, threshold=threshold,
                        first_response_ms=1000 * (a['response_time_s'] - t0),
                        confirmation_response_ms=1000 * (b['response_time_s'] - t0),
                        signed_change=a[field] - center)
    return dict(status='not_detected', baseline=center, threshold=threshold,
                first_response_ms=None, confirmation_response_ms=None)


def analyze(run):
    metadata = json.loads((run / 'run.json').read_text())
    if metadata['experiment'] != 'response-timing':
        raise ValueError('requires a response-timing recording')
    cfg = metadata['resolved_config']
    fields = ('request_time_s', 'response_time_s', 'position_rev', 'velocity_rev_s',
              'q_current_a', 'mode')
    with (run / 'telemetry.csv').open(newline='') as source:
        rows = list(csv.DictReader(source))
    if len(rows) != metadata['sample_count']:
        raise ValueError('recorded sample count mismatch')
    for i, row in enumerate(rows):
        if int(row['sample']) != i:
            raise ValueError('sample sequence mismatch')
        for field in fields:
            row[field] = float(row[field])
            if not math.isfinite(row[field]):
                raise ValueError(f'non-finite {field}; inspect invalid/aborted recording')
        if row['response_time_s'] < row['request_time_s']:
            raise ValueError('negative response interval')
        if i and row['request_time_s'] < rows[i - 1]['response_time_s']:
            raise ValueError('overlapping requests')
    trials = []
    for pulse in (r for r in rows if r['experiment_phase'] == 'pulse'):
        trial = pulse['trial_id']
        baseline = [r for r in rows if r['trial_id'] == trial and r['experiment_phase'] == 'baseline']
        if len(baseline) < 3:
            raise ValueError('pulse lacks baseline')
        # Preserve the sample immediately before the quiet-window boundary.
        while len(baseline) > 1 and baseline[-1]['response_time_s'] - baseline[1]['response_time_s'] >= cfg['quiet_s']:
            baseline.pop(0)
        observation = [r for r in rows if r['trial_id'] == trial and r['experiment_phase'] in ('pulse', 'observe')]
        t0 = pulse['request_time_s']
        max_gap = 2.5 / cfg['sample_rate_hz']
        # Current detection only while the controller still reports position/torque mode.
        active = [r for r in observation if r['mode'] == 10]
        signals = {}
        for label, field, floor, samples in (
            ('current_a', 'q_current_a', .05, active),
            ('position_rev', 'position_rev', .1 / 360, observation),
            ('velocity_rev_s', 'velocity_rev_s', .02, observation),
        ):
            signals[label] = detect(samples, field, [r[field] for r in baseline], floor, t0, max_gap)
        gaps = [b['response_time_s'] - a['response_time_s']
                for a, b in zip([baseline[-1]] + observation, observation)]
        trials.append(dict(trial_id=int(trial), pulse_torque_nm=float(pulse['feedforward_torque_nm']),
                           baseline_samples=len(baseline), pulse_active_samples=len(active),
                           max_observation_gap_ms=1000 * max(gaps),
                           timing_gap_exceeds_2_5_periods=max(gaps) > max_gap,
                           watchdog_timeout_observed=any(r['mode'] == 11 for r in observation),
                           signals=signals))
    return dict(analysis_version=1, source=dict(run_id=metadata['run_id'], status=metadata['status'],
                reason=metadata['reason'], sample_count=len(rows), resolved_config=cfg,
                sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                        for name in ('run.json', 'telemetry.csv', 'events.csv', 'controller-config.txt')}),
                interpretation='Host command-to-observation detection, NOT pure actuator dead time. '
                'Two same-sign consecutive threshold crossings; baseline median and twice maximum deviation. '
                'Not detected is not zero delay. Partial runs and timing gaps require review. '
                'No device sample timestamp or one-way transport estimate.',
                round_trip_ms=stats([1000 * (r['response_time_s'] - r['request_time_s']) for r in rows]) if rows else None,
                trials=trials)


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
