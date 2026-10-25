"""Tier-A pilot: single watchdog-bounded torque pulses, with query-only observation."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict

from ..recorder import RunRecorder
from ..safety import SafetyStop, check_state
from ..servo import create_controller, result_to_state, snapshot_controller
from .common import wait_until


CONNECT_TIMEOUT_S = 5.0


def validate(cfg, limits):
    for name, value in asdict(cfg).items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"response_timing.{name} must be finite and positive")
    if not isinstance(cfg.repetitions_per_direction, int) or cfg.repetitions_per_direction > 50:
        raise ValueError("repetitions_per_direction must be an integer in 1..50")
    if not 100 <= cfg.sample_rate_hz <= 500:
        raise ValueError("response timing requires 100..500 Hz requested sampling")
    if cfg.pulse_torque_nm > min(.02, limits.max_torque_nm):
        raise ValueError("pilot pulse torque exceeds 0.02 Nm / Tier-A limit")
    if not 3 / cfg.sample_rate_hz <= cfg.pulse_s <= .03:
        raise ValueError("pulse must span at least three periods and be <=30 ms")
    if cfg.max_velocity_rev_s > min(.8, limits.max_abs_velocity_rev_s):
        raise ValueError("pilot velocity limit exceeds 0.8 rev/s / Tier-A limit")
    if cfg.max_excursion_rev > min(.05, limits.max_position_excursion_rev):
        raise ValueError("pilot excursion exceeds 0.05 rev / Tier-A limit")
    if cfg.quiet_velocity_rev_s >= cfg.max_velocity_rev_s or cfg.quiet_span_rev >= cfg.max_excursion_rev:
        raise ValueError("quiet thresholds must be below live limits")
    if cfg.quiet_s < .1 or cfg.quiet_timeout_s <= cfg.quiet_s:
        raise ValueError("quiet window must be >=100 ms and below quiet timeout")
    if cfg.observe_s < cfg.pulse_s + 3 / cfg.sample_rate_hz:
        raise ValueError("observation must include at least three post-pulse periods")
    if not 1 / cfg.sample_rate_hz < cfg.response_timeout_s <= .1:
        raise ValueError("response timeout must exceed sample period and be <=100 ms")


def describe(cfg):
    return (f"Response timing: {2 * cfg.repetitions_per_direction} alternating pulses, "
            f"±{cfg.pulse_torque_nm:g} Nm for {1000 * cfg.pulse_s:g} ms; "
            f"{cfg.sample_rate_hz:g} Hz requested; stop thresholds: speed {cfg.max_velocity_rev_s:g} rev/s, "
            f"total excursion {cfg.max_excursion_rev:g} rev. "
            "No automatic position return. Timeout mode temporarily set to torque-off (0).")


def quiet_window(rows, cfg):
    if len(rows) < 3 or rows[-1]['response_time_s'] - rows[0]['response_time_s'] < cfg.quiet_s:
        return False
    if any(b['response_time_s'] - a['response_time_s'] > 2.5 / cfg.sample_rate_hz
           for a, b in zip(rows, rows[1:])):
        return False
    positions = [r['position_rev'] for r in rows]
    return (max(positions) - min(positions) <= cfg.quiet_span_rev
            and all(abs(r['velocity_rev_s']) <= cfg.quiet_velocity_rev_s for r in rows)
            and all(r['mode'] == 10 for r in rows))


def torque_arguments(torque, cfg):
    return dict(position=math.nan, velocity=0.0, kp_scale=0.0, kd_scale=0.0,
                ilimit_scale=0.0, feedforward_torque=torque,
                maximum_torque=cfg.pulse_torque_nm, watchdog_timeout=cfg.pulse_s,
                query=True)


async def run(config, output_root, controller_id):
    cfg = config.response_timing
    validate(cfg, config.safety)
    recorder = RunRecorder(output_root=output_root, experiment='response-timing',
                           resolved_config={'controller_id': controller_id, **asdict(cfg)})
    controller = stream = None
    original_timeout = None
    timeout_changed = False
    status, reason = 'failed', None
    start = time.perf_counter()
    sample = 0
    origin = None

    async def bounded(awaitable, timeout=None):
        return await asyncio.wait_for(awaitable, timeout or cfg.response_timeout_s)

    async def stopped(timeout=None):
        result = await bounded(controller.set_stop(query=True), timeout)
        state = result_to_state(result, moteus)
        if state['mode'] != 0:
            # The stop reply carries the mode sampled before the command is
            # applied (measured: mode 2 right after a position command, 0 on
            # the next query). Confirm with a follow-up query.
            await asyncio.sleep(0.005)
            state = result_to_state(await bounded(controller.query()), moteus)
        if state['mode'] != 0:
            raise SafetyStop('stop was not confirmed by controller')
        return state

    async def record(phase, trial, torque=None):
        nonlocal sample
        # No catch-up bursts: a late response shifts the next request.
        scheduled = time.perf_counter() - start
        request = time.perf_counter() - start
        if torque is None:
            result = await bounded(controller.query())
            command = {'command_mode': 'query'}
        else:
            result = await bounded(controller.set_position(**torque_arguments(torque, cfg)))
            command = dict(command_mode='torque', command_velocity_rev_s=0.0,
                           feedforward_torque_nm=torque, maximum_torque_nm=cfg.pulse_torque_nm,
                           watchdog_timeout_s=cfg.pulse_s, kp_scale=0.0, kd_scale=0.0,
                           ilimit_scale=0.0)
        response = time.perf_counter() - start
        state = result_to_state(result, moteus)
        row = dict(sample=sample, host_time_s=response, scheduled_time_s=scheduled,
                   request_time_s=request, response_time_s=response,
                   experiment_phase=phase, trial_id=trial, **command, **state)
        recorder.write_sample(row)
        sample += 1
        check_state(state, config.safety, origin_position_rev=origin,
                    max_position_excursion_rev=cfg.max_excursion_rev,
                    max_abs_velocity_rev_s=cfg.max_velocity_rev_s)
        for field in ('position_rev', 'velocity_rev_s', 'q_current_a', 'd_current_a',
                      'torque_nm', 'controller_temperature_c', 'bus_voltage_v'):
            if not math.isfinite(state[field]):
                raise SafetyStop(f'non-finite {field}')
        if state['mode'] not in (0, 2, 3, 10, 11):
            raise SafetyStop(f'unexpected controller mode {state["mode"]}')
        return row

    try:
        controller, moteus = create_controller(controller_id, timing=True)
        # The first transaction also opens the fdcanusb transport (~80 ms
        # measured), so it is not held to the per-sample response bound.
        await stopped(timeout=CONNECT_TIMEOUT_S)
        print('Stopped; capturing controller configuration...', flush=True)
        configuration, device = await asyncio.wait_for(snapshot_controller(controller, moteus), 60)
        recorder.attach_controller(configuration, device)
        settings = dict(line.split(maxsplit=1) for line in configuration.splitlines() if ' ' in line)
        original_timeout = int(settings['servo.timeout_mode'])
        state = await stopped()
        check_state(state, config.safety)
        origin = state['position_rev']
        if not math.isfinite(origin) or state.get('home_state') not in (1, 2):
            raise SafetyStop('finite referenced rotor position required')
        lower, upper = (float(settings[k]) for k in ('servopos.position_min', 'servopos.position_max'))
        if not (math.isfinite(lower) and math.isfinite(upper)
                and lower < origin - cfg.max_excursion_rev
                and origin + cfg.max_excursion_rev < upper):
            raise SafetyStop('finite controller position bounds with full excursion clearance required')
        print(describe(cfg), flush=True)
        print(f'Run directory: {recorder.path}', flush=True)
        if input('Secure stand and light arrow, clear travel, prepare power disconnect; type PULSE: ').strip() != 'PULSE':
            status, reason = 'aborted', 'operator did not confirm PULSE'
            return str(recorder.path)
        recorder.event(time.perf_counter() - start, 'operator_confirmed', value='PULSE')
        stream = moteus.Stream(controller)
        # A fresh Stream's first read runs the library's flow-control probe,
        # which swallows the first command response (measured: 3 s timeout).
        # flush_read() bypasses the probe and drains stale diagnostic data.
        await asyncio.wait_for(stream.flush_read(), 2)
        # Mark before sending: a lost response must still trigger restoration.
        timeout_changed = True
        await asyncio.wait_for(stream.command(b'conf set servo.timeout_mode 0'), 2)
        readback = await asyncio.wait_for(stream.command(b'conf get servo.timeout_mode', allow_any_response=True), 2)
        if int(readback.decode().strip()) != 0:
            raise SafetyStop('torque-off timeout mode readback failed')
        recorder.event(time.perf_counter() - start, 'timeout_mode_override', value=0,
                       note=f'original={original_timeout}; volatile only; never conf write')
        for trial in range(2 * cfg.repetitions_per_direction):
            direction = 1 if trial % 2 == 0 else -1
            print(f'Trial {trial + 1}: waiting for quiet zero-torque baseline...', flush=True)
            rows = []
            deadline = time.perf_counter() + cfg.quiet_timeout_s
            while True:
                if time.perf_counter() >= deadline:
                    raise SafetyStop('quiet baseline timeout; no pulse issued for this trial')
                tick = time.perf_counter()
                row = await record('baseline', trial, 0.0)
                rows.append(row)
                while len(rows) > 1 and rows[-1]['response_time_s'] - rows[1]['response_time_s'] >= cfg.quiet_s:
                    rows.pop(0)
                if quiet_window(rows, cfg):
                    break
                await wait_until(tick + 1 / cfg.sample_rate_hz)
            recorder.event(time.perf_counter() - start, 'baseline_accepted', value=trial,
                           note=f'{len(rows)} samples; span {rows[-1]["response_time_s"] - rows[0]["response_time_s"]:.6f}s')
            pulse_start = time.perf_counter()
            pulse = await record('pulse', trial, direction * cfg.pulse_torque_nm)
            # This is the ONLY nonzero command. Queries do not refresh the watchdog.
            observed_timeout = pulse['mode'] == 11
            while time.perf_counter() - pulse_start < cfg.observe_s:
                await wait_until(pulse_start + 1 / cfg.sample_rate_hz)
                tick = time.perf_counter()
                row = await record('observe', trial)
                observed_timeout |= row['mode'] == 11
                await wait_until(tick + 1 / cfg.sample_rate_hz)
            await stopped()
            if not observed_timeout:
                raise SafetyStop('watchdog timeout state was not observed after pulse')
            recorder.event(time.perf_counter() - start, 'trial_complete', value=trial)
        status = 'completed'
    except BaseException as exc:
        status = 'safety_stop' if isinstance(exc, (SafetyStop, TimeoutError)) else 'failed'
        reason = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            if controller is not None:
                await stopped()
            if timeout_changed:
                await asyncio.wait_for(stream.command(f'conf set servo.timeout_mode {original_timeout}'.encode()), 2)
                restored = await asyncio.wait_for(stream.command(b'conf get servo.timeout_mode', allow_any_response=True), 2)
                if int(restored.decode().strip()) != original_timeout:
                    raise SafetyStop('timeout mode restoration readback failed')
                recorder.event(time.perf_counter() - start, 'timeout_mode_restored', value=original_timeout)
        except BaseException as exc:
            status = 'cleanup_failed'
            reason = f'{reason or ""}; cleanup: {type(exc).__name__}: {exc}'
            print('STOP/RESTORE NOT CONFIRMED: disconnect power; inspect timeout mode before reuse.', flush=True)
            raise
        finally:
            recorder.close(status, reason)
    return str(recorder.path)
